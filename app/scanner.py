# SPDX-License-Identifier: MPL-2.0
"""Corpus selection: which files are indexable, and how to read them.

Deliberately dependency-free (stdlib only) so the include/exclude rules can be
exercised offline — see ``rag/selftest.py`` — without Qdrant, Ollama, or any
third-party package installed.
"""

from __future__ import annotations

import fnmatch
import os
from typing import Iterator

from .config import FILENAME_LANGUAGES, LANGUAGE_BY_EXT, Config


def language_for(filename: str, cfg: Config) -> str | None:
    """Language tag for a path, or None if it is not indexable."""
    base = os.path.basename(filename)
    ext = os.path.splitext(base)[1].lower()
    if base in FILENAME_LANGUAGES:
        language = FILENAME_LANGUAGES[base]
    else:
        # RAG_EXTRA_TEXT_EXTS wins, so a site can override a built-in mapping
        # as well as add new ones - all without touching the code.
        language = cfg.extra_text_exts.get(ext) or LANGUAGE_BY_EXT.get(ext)
    if language is None:
        return None
    if cfg.include_languages and language not in cfg.include_languages:
        return None
    return language


def is_excluded(rel_path: str, cfg: Config) -> bool:
    parts = rel_path.split("/")
    if any(part in cfg.exclude_dirs for part in parts[:-1]):
        return True
    return any(fnmatch.fnmatch(parts[-1], pattern) for pattern in cfg.exclude_globs)


def iter_archive_members(rel_path: str, cfg: Config) -> Iterator[tuple[str, str]]:
    """Yield (virtual path, language) for the indexable files inside an archive."""
    from .archive import SEPARATOR, list_members

    abs_path = os.path.join(cfg.repo_path, rel_path)
    for member, _size in list_members(
        abs_path, cfg.archive_max_bytes, cfg.archive_max_members
    ):
        language = language_for(member, cfg)
        if language is None:
            continue
        yield f"{rel_path}{SEPARATOR}{member}", language


def iter_source_files(cfg: Config) -> Iterator[tuple[str, str]]:
    """Yield (repo-relative path, language) for every indexable file.

    Archive members appear as their own entries under a virtual path, so each
    document inside a zip is indexed and cited individually.
    """
    from .archive import is_archive

    root = cfg.repo_path
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in cfg.exclude_dirs)
        for filename in sorted(filenames):
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
            if is_excluded(rel_path, cfg):
                continue

            if cfg.archives and is_archive(filename):
                yield from iter_archive_members(rel_path, cfg)
                continue

            language = language_for(filename, cfg)
            if language is None:
                continue
            yield rel_path, language


def unsupported_counts(cfg: Config) -> dict[str, int]:
    """Count files whose format has no dependable reader (.doc, .ppt).

    Reported rather than silently skipped: a search index that quietly omits
    every legacy Word document is worse than one that says so, because you only
    discover the gap when a search comes back empty and you trust it.
    """
    from .extract import UNSUPPORTED_EXTS

    counts: dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(cfg.repo_path):
        dirnames[:] = [d for d in dirnames if d not in cfg.exclude_dirs]
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in UNSUPPORTED_EXTS:
                counts[ext] = counts.get(ext, 0) + 1
    return counts


def read_text(abs_path: str, max_bytes: int) -> str | None:
    """Read a file as text, or None if it is missing, binary, or oversized.

    Document formats (PDF, Word, slides, spreadsheets) are binary and are
    handled by ``load_text`` below, which routes them through app.extract.
    """
    try:
        if os.path.getsize(abs_path) > max_bytes:
            return None
        with open(abs_path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    return raw.decode("utf-8", errors="replace")


def load_text(abs_path: str, max_bytes: int) -> str | None:
    """Text of any indexable file, extracting document formats as needed.

    Accepts an archive member's virtual path ("bundle.zip!doc.pdf") as well as
    an ordinary file.
    """
    # Imported lazily: these reach for optional third-party libraries, and
    # selftest.py must be able to walk a corpus with none of them present.
    from .archive import read_member, split_member
    from .extract import extract, is_document

    member = split_member(abs_path)
    if member:
        return read_member(member[0], member[1], max_bytes)

    try:
        if os.path.getsize(abs_path) > max_bytes:
            return None
    except OSError:
        return None

    if is_document(abs_path):
        return extract(abs_path)
    return read_text(abs_path, max_bytes)
