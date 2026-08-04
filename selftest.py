#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Offline sanity check for the RAG corpus and chunker.

Runs the real include/exclude rules and the real chunker over a repository
without Docker, Qdrant, or an embedding model — so you can confirm *what* would
be indexed before paying for a first ingest.

    python3 rag/selftest.py [repo_path]
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.chunker import chunk_file  # noqa: E402
from app.config import CONFIG, corpus_warning  # noqa: E402
from app.scanner import iter_source_files, load_text  # noqa: E402


def check_powershell_ascii() -> int:
    """Fail loudly if a .ps1 here has drifted away from pure ASCII.

    Windows PowerShell 5.1 decodes a BOM-less .ps1 as Windows-1252, so a UTF-8
    em-dash becomes three characters ending in a double quote. Inside a
    double-quoted string that silently truncates the string and starts a runaway
    one that swallows following function definitions whole. The symptom is a
    "term is not recognized" error for a function that is plainly there.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    problems = 0
    for name in sorted(os.listdir(here)):
        if not name.endswith(".ps1"):
            continue
        with open(os.path.join(here, name), "rb") as handle:
            raw = handle.read()
        for lineno, line in enumerate(raw.split(b"\n"), 1):
            bad = [b for b in line if b > 127]
            if bad:
                print(f"NON-ASCII: {name}:{lineno} -> bytes {[hex(b) for b in bad[:4]]}")
                problems += 1
    return problems


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    if not os.path.isdir(repo):
        print(f"not a directory: {repo}")
        return 1

    CONFIG.repo_path = repo
    print(f"corpus: {repo}")
    warning = corpus_warning(CONFIG)
    if warning and len(sys.argv) <= 1:
        print(f"WARNING: {warning}")
    print()

    files_by_lang: Counter[str] = Counter()
    chunks_by_lang: Counter[str] = Counter()
    total_chars = 0
    skipped = 0
    sample = None

    for rel_path, language in iter_source_files(CONFIG):
        content = load_text(os.path.join(repo, rel_path), CONFIG.max_file_bytes)
        if content is None:
            skipped += 1
            continue
        chunks = chunk_file(
            rel_path,
            language,
            content,
            chunk_chars=CONFIG.chunk_chars,
            overlap_lines=CONFIG.chunk_overlap_lines,
        )
        if not chunks:
            skipped += 1
            continue
        files_by_lang[language] += 1
        chunks_by_lang[language] += len(chunks)
        total_chars += sum(len(c.text) for c in chunks)
        if sample is None and language == "rs" and len(chunks) > 2:
            sample = chunks[1]

    print(f"{'language':<12} {'files':>7} {'chunks':>8}")
    print("-" * 29)
    for language, count in files_by_lang.most_common():
        print(f"{language:<12} {count:>7} {chunks_by_lang[language]:>8}")
    print("-" * 29)
    print(f"{'TOTAL':<12} {sum(files_by_lang.values()):>7} {sum(chunks_by_lang.values()):>8}")
    print(f"\nskipped (empty/binary/oversized): {skipped}")
    print(f"text to embed: {total_chars / 1_000_000:.1f} MB")

    if sample:
        print(f"\nsample chunk — {sample.path}:{sample.start_line}-{sample.end_line}")
        print(f"symbol: {sample.symbol}")
        print("-" * 60)
        print(sample.text[:600])
        print("-" * 60)

    if check_powershell_ascii():
        print("\nFAIL: a PowerShell script contains non-ASCII bytes (see above).")
        return 1

    if not files_by_lang:
        print("\nWARNING: nothing would be indexed. Check RAG_REPO_MOUNT / exclude rules.")
        return 1
    print("\nPowerShell scripts are pure ASCII: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Piping into `head` closes stdout early; that is not an error.
        os._exit(0)
