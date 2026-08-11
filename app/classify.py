# SPDX-License-Identifier: MPL-2.0
"""Deciding the same question about every row of a large spreadsheet.

The shape of the problem: 16,000 failure modes, one policy document defining
what makes a failure mode built-in-test applicable, and a verdict needed for
each row. Done a row at a time through a retrieval-and-ask loop it cost about
1.5 seconds a row - roughly six and three quarter hours, and that is before
anyone disagrees with a verdict and wants it re-run.

Almost none of that time was doing anything useful, and it is worth being
precise about where it went, because it decides what to fix. Measured on this
stack, embedding one row costs 27ms and searching for it a few more - under two
per cent of the 1.5 seconds. The rest is a language model writing one verdict.
So a faster embedder, or a GPU under the embedder, buys nothing here. Three
things do:

**The criteria never change.** Every row was retrieving substantially the same
policy passages. Retrieved once and reused, that work disappears - and every
row is then judged against identical wording, which matters more for a
determination someone has to defend than the speed does.

**Failure modes repeat.** The same mode recurs across many line items with
different identifiers, part numbers and effects. Decided once per distinct
case and mapped back, the model sees a fraction of the rows.

**A verdict is small.** One call can carry many rows, and the fixed cost of a
call - the criteria, the instructions, the model's own preamble - is then paid
once per batch rather than once per row.

Batching is not free, and the cost is accuracy rather than money: past some
width a model starts pattern-matching across rows instead of judging each on
its merits. Where that begins depends on the model and the rows, so this
reports enough to find it (`agreement` against a truth column) rather than
picking a number and hoping. Treat batch size as an accuracy dial that happens
to also make things faster.

Everything here degrades rather than fails. A batch whose reply cannot be
parsed, or comes back missing rows, is retried in halves and finally one row at
a time; a row that fails alone is recorded as an error with its reason, never
silently dropped. Sixteen thousand determinations with four unexplained gaps is
a worse outcome than one that took longer.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Callable

log = logging.getLogger("rag.classify")

# Enough of a verdict to act on and to audit: what was decided, on which
# criterion, and how sure. `criterion` is what makes a spreadsheet of 16,000
# verdicts defensible - "applicable" alone is an opinion, "applicable, by the
# continuous-monitoring clause" is a determination.
VERDICT_FIELDS = ("id", "label", "criterion", "confidence")

_WHITESPACE = re.compile(r"\s+")
_UNINFORMATIVE = re.compile(r"[^a-z0-9 ]+")


def normalise(text: str) -> str:
    """A key for deciding two rows pose the same question.

    Case, punctuation and spacing carry no meaning for whether a failure mode
    is BIT applicable, and treating "Loss of Signal - Ch 3" and "loss of
    signal, ch 3" as different questions doubles the work for nothing.
    """
    lowered = _UNINFORMATIVE.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", lowered).strip()


class Case:
    """One distinct question, and every row that asks it."""

    __slots__ = ("id", "key", "text", "rows", "label", "criterion",
                 "confidence", "error")

    def __init__(self, ident: str, key: str, text: str) -> None:
        # A short unique handle the model copies back. Assigned rather than
        # derived from the text, because derived ids collide: the first twelve
        # characters of "loss of signal on channel 1" and "...channel 2" are
        # the same string, and two cases sharing an id means one of them
        # quietly overwrites the other in the lookup and is never asked about
        # at all. That is worse than a wrong answer - it is a missing one, and
        # in a sheet of sixteen thousand rows nobody notices which.
        self.id = ident
        self.key = key
        self.text = text
        self.rows: list[int] = []
        self.label: str | None = None
        self.criterion: str = ""
        self.confidence: str = ""
        self.error: str | None = None


def build_cases(rows: list[dict], columns: list[str]) -> list[Case]:
    """Group rows by the question they actually pose."""
    cases: dict[str, Case] = {}
    for index, row in enumerate(rows):
        parts = [str(row.get(column, "") or "").strip() for column in columns]
        text = " | ".join(part for part in parts if part)
        key = normalise(text)
        if not key:
            key = f"__blank__{index}"
        case = cases.get(key)
        if case is None:
            case = cases[key] = Case(f"c{len(cases) + 1}", key, text)
        case.rows.append(index)
    return list(cases.values())


PROMPT = """You are applying one fixed set of criteria to a list of items, for
an engineering record that someone will have to defend later.

The criteria, retrieved from the project's own documents:

{criteria}

Decide each item below against those criteria and nothing else. Where the
criteria do not settle an item, say so with the label {unclear!r} rather than
guessing - an item flagged for a human is cheap, a wrong verdict buried in
sixteen thousand rows is not.

Reply with one JSON object per line and nothing else. No preamble, no summary,
no markdown fence. Each object:

{{"id": "<the item's id, copied exactly>", "label": "<one of: {labels}>", "criterion": "<the specific criterion or passage number that decides it, briefly>", "confidence": "<high|medium|low>"}}

One line per item, every id present exactly once, ids copied exactly as given.

Items:
{items}"""


def _render_items(cases: list[Case]) -> str:
    return "\n".join(f"{case.id}. {case.text}" for case in cases)


def _parse(reply: str, wanted: dict[str, Case]) -> tuple[dict[str, dict], list[str]]:
    """Verdicts by id, and the ids that did not come back."""
    found: dict[str, dict] = {}
    for line in reply.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            # Models sometimes wrap the object in prose; take the first object
            # on the line if there is one rather than discarding the verdict.
            start = line.find("{")
            if start < 0:
                continue
            line = line[start:]
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("id", "")).strip()
        if key in wanted and key not in found:
            found[key] = entry
    missing = [key for key in wanted if key not in found]
    return found, missing


def classify_cases(
    cases: list[Case],
    criteria: str,
    labels: list[str],
    ask: Callable[[str], str],
    batch_size: int = 25,
    unclear: str = "unclear",
    on_progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    concurrency: int = 1,
) -> None:
    """Fill in each case's verdict, in batches, retrying narrower on trouble.

    Batches run concurrently when `concurrency` is above one. The work is
    entirely waiting on a model, so threads are the right tool; what decides a
    useful number is the server at the other end - one model instance gains
    little from being asked four things at once, while several GPUs serving
    several instances gain nearly linearly. Higher is not automatically better:
    past what the server can actually run in parallel, requests queue and the
    only thing that grows is the time each one appears to take.
    """
    allowed = {label.lower(): label for label in labels}
    done = 0
    total = len(cases)
    counter = threading.Lock()

    def run(group: list[Case], width: int) -> None:
        nonlocal done
        if not group:
            return
        if should_stop and should_stop():
            raise Cancelled()

        wanted = {case.id: case for case in group}
        # Cheap, and it is the invariant the whole retry scheme rests on: an
        # id seen twice means a case is about to be asked about zero times.
        if len(wanted) != len(group):
            raise RuntimeError("duplicate case ids - verdicts would be lost")
        prompt = PROMPT.format(
            criteria=criteria,
            labels=", ".join(labels),
            unclear=unclear,
            items=_render_items(group),
        )
        try:
            reply = ask(prompt)
            verdicts, missing = _parse(reply, wanted)
        except Exception as exc:  # a dead model must not lose the whole run
            reply, verdicts, missing = "", {}, list(wanted)
            log.warning("classification call failed: %s", exc)

        for key, entry in verdicts.items():
            case = wanted[key]
            raw = str(entry.get("label", "")).strip().lower()
            # An unrecognised label is not a verdict. Better to mark it unclear
            # and let a human look than to record a label nobody defined.
            case.label = allowed.get(raw, unclear)
            case.criterion = str(entry.get("criterion", ""))[:300]
            case.confidence = str(entry.get("confidence", ""))[:12]
            with counter:
                done += 1

        if missing:
            stragglers = [wanted[key] for key in missing]
            if width > 1:
                # Halve and retry: a batch too wide for the model to keep
                # straight usually succeeds split, and splitting is cheaper
                # than dropping to one row at a time immediately.
                half = max(1, width // 2)
                for start in range(0, len(stragglers), half):
                    run(stragglers[start:start + half], half)
            else:
                for case in stragglers:
                    case.error = "no verdict returned for this row"
                    with counter:
                        done += 1
        if on_progress:
            on_progress(min(done, total), total)

    batches = [cases[start:start + batch_size]
               for start in range(0, total, batch_size)]

    if concurrency <= 1:
        for group in batches:
            run(group, batch_size)
        return

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="classify") as pool:
        futures = [pool.submit(run, group, batch_size) for group in batches]
        for future in futures:
            # Surfaced rather than swallowed: a cancellation has to unwind, and
            # anything else is a bug worth seeing rather than a quietly short
            # results file.
            future.result()


class Cancelled(RuntimeError):
    """Raised to unwind a job someone asked to stop."""


def results(rows: list[dict], cases: list[Case], id_column: str | None,
            truth_column: str | None) -> dict:
    """Per-row verdicts, plus agreement when the sheet already has answers."""
    per_row: list[dict] = [{} for _ in rows]
    for case in cases:
        for index in case.rows:
            per_row[index] = {
                "row": index + 2,  # 1-based, plus the header line
                "id": rows[index].get(id_column) if id_column else index + 1,
                "label": case.label,
                "criterion": case.criterion,
                "confidence": case.confidence,
                "error": case.error,
            }

    agreement = None
    if truth_column:
        checked = matched = 0
        for index, row in enumerate(rows):
            truth = str(row.get(truth_column, "") or "").strip().lower()
            got = (per_row[index].get("label") or "").strip().lower()
            if not truth or not got:
                continue
            checked += 1
            matched += truth == got
        if checked:
            agreement = {
                "checked": checked,
                "agreed": matched,
                "rate": round(matched / checked, 4),
            }

    labels: dict[str, int] = {}
    for entry in per_row:
        labels[entry.get("label") or "error"] = labels.get(entry.get("label") or "error", 0) + 1

    return {
        "rows": len(rows),
        "distinct_cases": len(cases),
        # The number that explains the runtime: how much of the sheet was
        # actually a repeat of something already decided.
        "deduplication": round(1 - len(cases) / max(len(rows), 1), 4),
        "labels": labels,
        "errors": sum(1 for case in cases if case.error),
        "agreement": agreement,
        "verdicts": per_row,
    }
