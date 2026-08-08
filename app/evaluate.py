# SPDX-License-Identifier: MPL-2.0
"""Measuring whether a retrieval change helped.

Every knob in this project - reranking, hybrid fusion, hop count, query
expansion, what goes into the embedded text - was chosen by running a few
questions and looking at the results. That works for finding out whether
something is catastrophically broken and not much else: five questions cannot
tell you that a change helped the corpus, only that it helped those five, and
the ones it quietly broke are by definition the ones nobody typed.

So: a list of questions with the documents that ought to answer them, and two
numbers.

    hit@k   the share of questions with a right answer in the top k. What a
            reader actually experiences, since nobody reads past the first
            screen.
    MRR     mean reciprocal rank - 1.0 if the right document is always first,
            0.5 if always second. Rewards moving a hit from rank 5 to rank 2,
            which hit@k cannot see.

No language model is involved and none is needed. This grades *retrieval* -
whether the right passage was found - which is the half that has to work before
any generator has a chance. Judging the prose an LLM writes is a different
problem needing a different (and much more expensive) apparatus, and it is not
this one.

Expectations are substrings matched against a result's path, not exact
citations, because line ranges move whenever chunking changes and a golden set
that has to be rewritten after every change will not be maintained. "Ch 011"
keeps meaning what it meant.

    python -m app.evaluate                          measure
    python -m app.evaluate --save before.json       record
    python -m app.evaluate --compare before.json    what a change did
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_FILE = "rag-eval.json"
DEFAULT_API = "http://127.0.0.1:49404"
CUTOFFS = (1, 3, 5, 10)


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    cases = data["questions"] if isinstance(data, dict) else data

    for index, case in enumerate(cases, start=1):
        if not case.get("question"):
            raise ValueError(f"case {index} has no question")
        # One spelling, checked once, rather than a KeyError per case later.
        if not case.get("expect"):
            raise ValueError(f"case {index} ({case['question'][:40]!r}) has no expect")
        if isinstance(case["expect"], str):
            case["expect"] = [case["expect"]]
    return cases


def search(api: str, question: str, top_k: int, body_extra: dict) -> list[dict]:
    body = {"query": question, "top_k": top_k}
    body.update(body_extra)
    request = urllib.request.Request(
        api.rstrip("/") + "/search",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read()).get("results", [])


def rank_of_first_hit(results: list[dict], expect: list[str]) -> int | None:
    """1-based rank of the first result matching any expectation.

    Matched against the section and symbol as well as the path, so a corpus
    living in one large file can still be graded: there the answer is "the
    Autostart section", not "a different document".
    """
    wanted = [e.lower() for e in expect]
    for rank, hit in enumerate(results, start=1):
        haystack = " ".join(str(hit.get(field, "")) for field in
                            ("path", "citation", "section", "symbol")).lower()
        if any(w in haystack for w in wanted):
            return rank
    return None


def evaluate(cases: list[dict], api: str, top_k: int, body_extra: dict) -> dict:
    rows = []
    for case in cases:
        try:
            results = search(api, case["question"], top_k, body_extra)
            rank = rank_of_first_hit(results, case["expect"])
            error = None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            results, rank, error = [], None, str(exc)
        rows.append({
            "question": case["question"],
            "expect": case["expect"],
            "rank": rank,
            "error": error,
            "top": [hit.get("path", "") for hit in results[:3]],
        })

    found = [row for row in rows if row["rank"]]
    total = len(rows) or 1
    return {
        "cases": len(rows),
        "hit": {
            f"@{k}": round(sum(1 for r in found if r["rank"] <= k) / total, 4)
            for k in CUTOFFS if k <= top_k
        },
        "mrr": round(sum(1 / r["rank"] for r in found) / total, 4),
        "misses": [r["question"] for r in rows if not r["rank"]],
        "rows": rows,
    }


def format_report(result: dict, baseline: dict | None = None) -> str:
    lines = []
    ranks = {r["question"]: r["rank"] for r in (baseline or {}).get("rows", [])}

    lines.append("")
    for row in result["rows"]:
        rank, was = row["rank"], ranks.get(row["question"])
        now = f"#{rank}" if rank else "MISS"
        mark = "  "
        if baseline is not None and row["question"] in ranks:
            if was == rank:
                mark = "  "
            elif rank and (was is None or rank < was):
                mark = "up"
            else:
                mark = "DN"
            shown = f"{'#' + str(was) if was else 'MISS':>5} -> {now:<5}"
        else:
            shown = f"{now:>5}      "
        lines.append(f" {mark} {shown} {row['question'][:62]}")
        if row["error"]:
            lines.append(f"        error: {row['error'][:80]}")
        elif not rank:
            lines.append(f"        wanted {'/'.join(row['expect'])}, got {', '.join(row['top']) or '(nothing)'}")

    lines.append("")
    summary = "  ".join(f"hit{k} {v:.0%}" for k, v in result["hit"].items())
    lines.append(f"  {result['cases']} questions   {summary}   MRR {result['mrr']:.3f}")
    if baseline is not None:
        delta = result["mrr"] - baseline["mrr"]
        moved = sum(
            1 for row in result["rows"]
            if row["question"] in ranks and row["rank"] != ranks[row["question"]]
        )
        was = "  ".join(f"hit{k} {v:.0%}" for k, v in baseline["hit"].items())
        lines.append(f"  was:                 {was}   MRR {baseline['mrr']:.3f}")
        lines.append(f"  MRR {delta:+.3f}, {moved} question(s) changed rank")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--file", default=DEFAULT_FILE, help="golden questions")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save", metavar="FILE", help="record this run")
    parser.add_argument("--compare", metavar="FILE", help="a run to compare against")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    # Anything the search endpoint takes, so a setting can be A/B'd without
    # restarting the server: --set hybrid=false --set expand_query=false
    parser.add_argument("--set", action="append", default=[], metavar="K=V")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"no {args.file}. Copy rag-eval.example.json and edit it: a dozen "
              f"questions you know the answers to is enough to start.", file=sys.stderr)
        return 2

    extra: dict = {}
    for pair in args.set:
        key, _, value = pair.partition("=")
        if value.lower() in ("true", "false"):
            extra[key] = value.lower() == "true"
        elif value.lstrip("-").isdigit():
            extra[key] = int(value)
        else:
            extra[key] = value

    try:
        cases = load_cases(args.file)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"{args.file}: {exc}", file=sys.stderr)
        return 2

    result = evaluate(cases, args.api, args.top_k, extra)

    baseline = None
    if args.compare:
        with open(args.compare, encoding="utf-8") as handle:
            baseline = json.load(handle)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_report(result, baseline))

    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\n  saved to {args.save}")

    # Non-zero when nothing was found at all, so this can gate a change.
    return 0 if result["mrr"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
