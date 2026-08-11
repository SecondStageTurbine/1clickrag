# SPDX-License-Identifier: MPL-2.0
"""Corpus selection: which files are indexable, and how to read them.

Deliberately dependency-free (stdlib only) so the include/exclude rules can be
exercised offline — see ``rag/selftest.py`` — without Qdrant, Ollama, or any
third-party package installed.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from typing import Iterator

from .config import FILENAME_LANGUAGES, LANGUAGE_BY_EXT, Config

log = logging.getLogger("rag.scanner")

# Windows reserved device names. A file called "nul" is not a file: the path
# resolves to the device \\.\nul, and os.path.relpath then raises ValueError
# because the two paths are on different "mounts". Unhandled, that escapes the
# generator and aborts the entire walk - so one stray file left behind by a
# shell redirect (`something > nul` under a POSIX shell writes a real file)
# takes down the whole ingest, not just itself.
#
# The reservation applies whatever the extension: "nul.txt" is the same device.
WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{n}" for n in range(1, 10)}
    | {f"lpt{n}" for n in range(1, 10)}
)


def is_device_name(filename: str) -> bool:
    """True for a name Windows resolves to a device rather than a file."""
    if os.name != "nt":
        return False
    return os.path.basename(filename).split(".")[0].strip().lower() in WINDOWS_DEVICE_NAMES


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
            if is_device_name(filename):
                log.warning("skipping reserved device name: %s", abs_path)
                continue
            try:
                rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
            except ValueError as exc:
                # Anything else the platform refuses to make relative. Skipping
                # one file is a gap in the index; letting it raise here is no
                # index at all.
                log.warning("skipping unusable path %s: %s", abs_path, exc)
                continue
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

    Raises OSError if the file is there but cannot be read - locked by the
    application still writing it, or on a share that just dropped. That is a
    different thing from "nothing to index here", and the difference decides
    whether the work queue retries it or forgets it: returning None for both
    is how a document saved from Word ends up permanently missing from the
    index, because the read happened while Word still held the handle.
    """
    try:
        size = os.path.getsize(abs_path)
    except OSError:
        return None  # vanished between the walk and the read

    if size > max_bytes:
        return None
    try:
        with open(abs_path, "rb") as handle:
            raw = handle.read()
    except FileNotFoundError:
        return None
    except OSError:
        if os.path.exists(abs_path):
            raise
        return None
    if b"\x00" in raw[:4096]:
        return None
    return raw.decode("utf-8", errors="replace")


_SURROGATES = re.compile("[\ud800-\udfff]")


def scrub(text: str | None) -> str | None:
    """Replace lone surrogates with U+FFFD.

    A str in Python may hold surrogate code points that no UTF-8 encoder will
    accept, and document extractors produce them: a PDF with a malformed
    ToUnicode map hands back half a surrogate pair, which is not a character
    and never was. Everything downstream eventually encodes - the extraction
    cache compresses UTF-8, the tokeniser wants UTF-8, SQLite wants UTF-8 - so
    one such code point anywhere in a 16,000-page corpus aborted the ingest
    with `'utf-8' codec can't encode character '\\ud800'`, naming a codec
    rather than the file, from a stack that had nothing to do with the PDF.

    U+FFFD rather than deletion, for consistency: every other decode here uses
    errors="replace" and therefore already yields U+FFFD for bytes it cannot
    make sense of. Same signal, same character, one meaning - "something was
    here and could not be represented".
    """
    if text is None or not _SURROGATES.search(text):
        return text
    return _SURROGATES.sub("�", text)


def load_text(abs_path: str, max_bytes: int) -> str | None:
    """Text of any indexable file, extracting document formats as needed.

    Accepts an archive member's virtual path ("bundle.zip!doc.pdf") as well as
    an ordinary file.

    Every path out of here is scrubbed of lone surrogates, because this is the
    one funnel all indexable text passes through - guarding the individual
    extractors would mean guarding each new one forever.
    """
    return scrub(_load_text(abs_path, max_bytes))


def _load_text(abs_path: str, max_bytes: int) -> str | None:
    # Imported lazily: these reach for optional third-party libraries, and
    # selftest.py must be able to walk a corpus with none of them present.
    from .archive import read_member, split_member
    from .config import CONFIG
    from .extract import extract, is_document
    from .extract_cache import fingerprint_file, open_cache

    cache = open_cache(CONFIG)
    member = split_member(abs_path)

    # Only the expensive paths are cached: extracting a document, or reaching
    # inside an archive. Plain text is already a single read, so caching it
    # would double the disk it occupies to save nothing.
    if member:
        container, inside = member
        if cache is None:
            return read_member(container, inside, max_bytes)
        # The container is hashed, not the member: opening the zip twice to
        # fingerprint one entry would cost more than it saves, and a changed
        # archive changes the hash for every member it holds.
        stamp = fingerprint_file(container)
        key = f"{stamp}:{inside}" if stamp else None
        if key:
            cached = cache.get(key)
            if cached is not None:
                return cached
        text = read_member(container, inside, max_bytes)
        if key and text:
            cache.put(key, text)
        return text

    try:
        if os.path.getsize(abs_path) > max_bytes:
            return None
    except OSError:
        return None

    if is_document(abs_path):
        if cache is None:
            return extract(abs_path)
        stamp = fingerprint_file(abs_path)
        if stamp is None:
            return extract(abs_path)  # unreadable now; let extract() report it
        cached = cache.get(stamp)
        if cached is not None:
            return cached
        text = extract(abs_path)
        if text:
            cache.put(stamp, text)
        return text
    return read_text(abs_path, max_bytes)
