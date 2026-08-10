# SPDX-License-Identifier: MPL-2.0
"""Questions about the corpus rather than about its contents.

"How many CISAR forms were signed this year?" is not a retrieval question and
no amount of better retrieval answers it. Search returns the handful of
passages most like the question; counting needs every document that matches and
none of their text. Asked of the chat pane, the model got five chunks, correctly
observed they did not contain a total, and said so - and sometimes added that it
could not see the user's files, which is the wrong explanation for the right
refusal.

So the count is computed here, from the index, and handed to the model as a
fact. The model is never asked to tally a list: a language model counting
twenty-nine filenames is a coin toss, and there is no reason to gamble when
SELECT COUNT does it exactly.

What is counted is **documents, not chunks**. A PDF becomes many chunks and
"how many forms" means files. The distinction is invisible until it is wrong by
a factor of four.

Dates come from filenames first and modification time second, because a form
archive names its files by the date on the form - "Signed - A.CISAR.20260225.pdf"
was signed in 2026 whatever the filesystem thinks, and a file copied to a new
machine in 2027 has not changed when it was signed.
"""

from __future__ import annotations

import fnmatch
import os
import re
import time
from collections import Counter

# An 8-digit date embedded in a name, not part of a longer run of digits.
DATE8 = re.compile(r"(?<!\d)((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")
# A bare year, as a fallback.
YEAR4 = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def date_of(path: str, mtime: float | None) -> tuple[str | None, str | None, str]:
    """(year, month, where it came from) for one document."""
    name = os.path.basename(path)
    found = DATE8.search(name)
    if found:
        return found.group(1), f"{found.group(1)}-{found.group(2)}", "filename"
    found = YEAR4.search(name)
    if found:
        return found.group(1), None, "filename"
    if mtime:
        stamp = time.localtime(mtime)
        return str(stamp.tm_year), time.strftime("%Y-%m", stamp), "modified"
    return None, None, "unknown"


def _matches(path: str, match: str | None, path_prefix: str | None) -> bool:
    if path_prefix and not path.lower().startswith(path_prefix.lower()):
        return False
    if not match:
        return True
    haystack = path.lower()
    pattern = match.lower()
    # A pattern with wildcards is a glob; anything else is a plain substring,
    # because "CISAR" should find CISAR without anyone writing "*CISAR*".
    if any(ch in pattern for ch in "*?["):
        return fnmatch.fnmatch(haystack, pattern) or fnmatch.fnmatch(
            os.path.basename(haystack), pattern
        )
    return pattern in haystack


def count(documents: dict[str, dict], match: str | None = None,
          path_prefix: str | None = None, group_by: str = "year",
          examples: int = 5) -> dict:
    """Aggregate over indexed documents.

    `documents` is what Store.indexed_state() returns: path -> {source, mtime}.
    """
    total = len(documents)
    hits: list[tuple[str, dict]] = [
        (path, meta) for path, meta in documents.items()
        if _matches(path, match, path_prefix)
    ]

    dated = [(path, *date_of(path, meta.get("mtime"))) for path, meta in hits]
    basis: Counter[str] = Counter(entry[3] for entry in dated)

    # If any filename carries a date, those are authoritative and a file
    # without one is "undated" rather than dated by its modification time.
    # Mixing the two answers "how many were signed this year" with files that
    # were merely *copied* this year, which is a wrong answer that looks right.
    # Only when nothing has a filename date is mtime used at all.
    trust_filenames = basis.get("filename", 0) > 0

    groups: Counter[str] = Counter()
    for path, year, month, where in dated:
        if trust_filenames and where != "filename":
            year = month = None
        if group_by == "year":
            key = year or "undated"
        elif group_by == "month":
            key = month or year or "undated"
        elif group_by == "folder":
            key = os.path.dirname(path) or "."
        elif group_by == "extension":
            key = os.path.splitext(path)[1].lower() or "(none)"
        else:
            key = ""
        if key:
            groups[key] += 1

    def order(item):
        key = item[0]
        # "undated" always last, whichever way the rest is sorted - it is the
        # residue, not the newest thing in the corpus.
        return (key in ("undated", "(none)"), key)

    ordered = sorted(groups.items(), key=order)
    if group_by in ("year", "month"):
        undated = [kv for kv in ordered if kv[0] == "undated"]
        ordered = sorted(
            [kv for kv in ordered if kv[0] != "undated"],
            key=lambda kv: kv[0], reverse=True,
        ) + undated

    return {
        "matched": len(hits),
        "total_indexed": total,
        "match": match,
        "path_prefix": path_prefix,
        "group_by": group_by,
        "groups": [{"key": key, "count": value} for key, value in ordered],
        # Enough to cite, not enough to drown the prompt.
        "examples": [path for path, _ in sorted(hits)[:examples]],
        "dates_from": dict(basis),
    }


def describe(result: dict) -> str:
    """The same facts as a short block a model can answer and cite from."""
    what = f' matching "{result["match"]}"' if result.get("match") else ""
    where = f' under {result["path_prefix"]}' if result.get("path_prefix") else ""
    lines = [
        f"Corpus count (computed over the index, not estimated):",
        f"{result['matched']} indexed document(s){what}{where}, "
        f"out of {result['total_indexed']} in the whole index.",
    ]
    if result["groups"]:
        source = result.get("dates_from", {})
        note = ""
        if result["group_by"] in ("year", "month"):
            if source.get("filename"):
                note = " (dates read from the filenames)"
                if source.get("modified") or source.get("unknown"):
                    note = (
                        " (dates read from the filenames; files whose name carries"
                        " no date are counted as undated rather than dated by when"
                        " the file was last modified)"
                    )
            elif source.get("modified"):
                note = (
                    " (no filename carries a date, so these are each file's"
                    " modification time - which is when the file was last written,"
                    " not necessarily when the document was made)"
                )
        listed = ", ".join(f"{g['key']}: {g['count']}" for g in result["groups"][:24])
        lines.append(f"By {result['group_by']}{note}: {listed}.")
    if result["examples"]:
        lines.append("Examples: " + "; ".join(result["examples"]))
    # The model has no clock, and "this year" is unanswerable without one.
    lines.append(f"Today's date is {time.strftime('%Y-%m-%d')}.")
    return "\n".join(lines)
