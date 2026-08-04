# SPDX-License-Identifier: MPL-2.0
"""Structure-aware chunking.

Chunks are built from *boundaries* rather than blind character windows, so a
retrieved chunk usually starts at a function/impl/heading rather than mid-body.
Three strategies, picked by language:

* Rust    - split at top-level-ish item starts (fn/impl/struct/mod/...).
* Markdown- split at ATX headings.
* Other   - split at blank-line paragraph gaps.

Boundaries are then greedily packed up to ``chunk_chars``; a single boundary
larger than the budget is windowed with line overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

RUST_ITEM = re.compile(
    r"^\s{0,4}(?:#\[[^\]]*\]\s*)?"
    r"(?:pub(?:\([^)]*\))?\s+)?"
    r"(?:default\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?"
    r'(?:extern\s+"[^"]*"\s+)?'
    r"(fn|struct|enum|union|impl|trait|mod|static|type|macro_rules!)\b"
)

MD_HEADING = re.compile(r"^#{1,6}\s+\S")
# A heading at this level or above starts a new chunk outright, rather than
# being packed in with what came before. Without it a short "## Recommendation"
# section gets merged with the unrelated section above it, and the two topics
# average into one muddy vector - the passage is in the index but never wins a
# query, because only a fraction of its embedding is about it.
MD_HARD_HEADING = re.compile(r"^#{1,2}\s+\S")


@dataclass
class Chunk:
    """One embeddable unit of a file."""

    path: str
    language: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str
    symbol: str = ""
    # The real file on disk. Differs from `path` only for archive members,
    # where path is "bundle.zip!doc.pdf" and source is "bundle.zip". Deletion
    # keys on this, so re-indexing a changed archive clears all its members.
    source: str = ""


def looks_like_markdown(lines: list[str]) -> bool:
    """True if the text is structured with headings, whatever its extension.

    Plenty of real documents are Markdown in a .txt file. Going by extension
    alone gives them paragraph splitting and loses every section boundary.
    """
    headings = sum(1 for line in lines[:400] if MD_HEADING.match(line))
    return headings >= 3


def _boundaries(lines: list[str], language: str) -> tuple[list[int], set[int]]:
    """Return (sorted start indices, the subset that must not be packed into).

    The second value marks *hard* boundaries: a chunk always begins there.
    """
    starts: list[int] = [0]
    hard: set[int] = set()

    # Sniff ONLY plain text. "#" starts a comment in TOML, shell, Python, YAML
    # and more, so a heading regex matches their comment lines and would
    # shred every config file into heading-sized pieces.
    if language == "text" and looks_like_markdown(lines):
        language = "markdown"

    if language == "rs":
        in_block_comment = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if in_block_comment:
                if "*/" in stripped:
                    in_block_comment = False
                continue
            if stripped.startswith("/*"):
                if "*/" not in stripped:
                    in_block_comment = True
                continue
            if RUST_ITEM.match(line):
                starts.append(i)
    elif language == "markdown":
        in_fence = False
        for i, line in enumerate(lines):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence and MD_HEADING.match(line):
                starts.append(i)
                if MD_HARD_HEADING.match(line):
                    hard.add(i)
    else:
        blank_run = 0
        for i, line in enumerate(lines):
            if line.strip():
                if blank_run:
                    starts.append(i)
                blank_run = 0
            else:
                blank_run += 1

    # A boundary run can pick up attribute/doc-comment lines that belong to the
    # item below; pull the start upward through them so the chunk keeps its
    # docs.
    adjusted: list[int] = []
    for start in starts:
        i = start
        while i > 0:
            prev = lines[i - 1].strip()
            if prev.startswith("///") or prev.startswith("//!") or prev.startswith("#["):
                i -= 1
            else:
                break
        adjusted.append(i)
        if start in hard and i != start:
            hard.discard(start)
            hard.add(i)

    return sorted(set(adjusted)), hard


def _symbol_for(lines: list[str], language: str) -> str:
    """Best-effort human label for a chunk (shown in results)."""
    for line in lines[:12]:
        stripped = line.strip()
        if not stripped:
            continue
        if language == "markdown" and MD_HEADING.match(stripped):
            return stripped.lstrip("#").strip()[:120]
        if language == "rs" and RUST_ITEM.match(line):
            return stripped.rstrip("{").strip()[:120]
    for line in lines:
        if line.strip():
            return line.strip()[:120]
    return ""


def _window(
    path: str,
    language: str,
    lines: list[str],
    offset: int,
    budget: int,
    overlap_lines: int,
) -> list[Chunk]:
    """Split an oversized boundary block into overlapping line windows."""
    chunks: list[Chunk] = []
    i = 0
    n = len(lines)
    while i < n:
        size = 0
        j = i
        while j < n and (size + len(lines[j]) + 1 <= budget or j == i):
            size += len(lines[j]) + 1
            j += 1
        body = lines[i:j]
        chunks.append(
            Chunk(
                path=path,
                language=language,
                start_line=offset + i + 1,
                end_line=offset + j,
                text="\n".join(body),
                symbol=_symbol_for(body, language),
            )
        )
        if j >= n:
            break
        i = max(j - overlap_lines, i + 1)
    return chunks


def chunk_file(
    path: str,
    language: str,
    content: str,
    chunk_chars: int = 1600,
    overlap_lines: int = 8,
) -> list[Chunk]:
    """Chunk one file's text. Returns [] for empty/whitespace-only files."""
    if not content.strip():
        return []

    lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    starts, hard = _boundaries(lines, language)
    starts.append(len(lines))

    blocks: list[tuple[int, list[str], bool]] = []
    for a, b in zip(starts, starts[1:]):
        if b > a:
            blocks.append((a, lines[a:b], a in hard))

    chunks: list[Chunk] = []
    pending: list[str] = []
    pending_start = 0
    pending_size = 0

    def pending_has_body() -> bool:
        """True if `pending` holds anything beyond headings and blank lines.

        A heading with no body under it belongs with the section that follows,
        not alone: flushing on the next heading would otherwise emit a chunk
        containing only "# Introduction", which can win a result slot while
        telling the reader nothing.
        """
        return any(
            line.strip() and not MD_HEADING.match(line) and line.strip() != "---"
            for line in pending
        )

    def flush() -> None:
        nonlocal pending, pending_start, pending_size
        if not pending:
            return
        body = pending
        chunks.append(
            Chunk(
                path=path,
                language=language,
                start_line=pending_start + 1,
                end_line=pending_start + len(body),
                text="\n".join(body),
                symbol=_symbol_for(body, language),
            )
        )
        pending = []
        pending_size = 0

    for start, body, is_hard in blocks:
        size = sum(len(line) + 1 for line in body)

        # A section heading always starts its own chunk, so one topic does not
        # get averaged together with the previous one - unless what is pending
        # is only a heading, which has nothing to be averaged with anyway.
        if is_hard and pending_has_body():
            flush()

        if size > chunk_chars:
            flush()
            chunks.extend(
                _window(path, language, body, start, chunk_chars, overlap_lines)
            )
            continue

        if pending and pending_size + size > chunk_chars:
            flush()

        if not pending:
            pending_start = start
        pending.extend(body)
        pending_size += size

    flush()
    return [c for c in chunks if c.text.strip()]
