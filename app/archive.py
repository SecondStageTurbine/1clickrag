# SPDX-License-Identifier: MPL-2.0
"""Look inside archives and index the documents they contain.

A shared drive is full of zipped bundles - a project handover, a tender pack, a
year of invoices - and their contents are invisible to a plain file walk. Each
member is indexed as its own document under a virtual path:

    contracts/2026-tender.zip!schedule-b.pdf

so a hit tells you both which archive and which file inside it.

Deliberately shallow: archives nested inside archives are not opened. One level
covers the real cases, while unbounded recursion is how a zip bomb turns a
search index into a disk-space incident.
"""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile

log = logging.getLogger("rag.archive")

ARCHIVE_EXTS = {".zip"}

# Separates the archive from the member. Only ever interpreted when the left
# side ends in a known archive extension, so a literal "!" in a filename is not
# mistaken for one.
SEPARATOR = "!"


def is_archive(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in ARCHIVE_EXTS


def split_member(path: str) -> tuple[str, str] | None:
    """Split 'a/b.zip!inner/doc.pdf' into ('a/b.zip', 'inner/doc.pdf')."""
    index = path.rfind(SEPARATOR)
    while index > 0:
        outer = path[:index]
        if is_archive(outer):
            return outer, path[index + 1 :]
        index = path.rfind(SEPARATOR, 0, index)
    return None


def list_members(abs_path: str, max_bytes: int, max_members: int) -> list[tuple[str, int]]:
    """Return [(member name, uncompressed size)] for an archive's real files."""
    try:
        if os.path.getsize(abs_path) > max_bytes:
            log.warning("skipping oversized archive %s", abs_path)
            return []
    except OSError:
        return []

    try:
        with zipfile.ZipFile(abs_path) as archive:
            members = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                # Nested archives are not opened; see the module docstring.
                if is_archive(info.filename):
                    continue
                members.append((info.filename, info.file_size))
                if len(members) >= max_members:
                    log.warning(
                        "archive %s has more than %d files - indexing the first %d",
                        abs_path,
                        max_members,
                        max_members,
                    )
                    break
            return members
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        log.warning("could not read archive %s: %s", abs_path, exc)
        return []


def read_member(abs_path: str, member: str, max_bytes: int) -> str | None:
    """Extract one member to a temp file and run the normal readers over it.

    Extraction is to a temp file rather than a stream because the document
    readers (pypdf, python-docx, openpyxl) all want a real path, and because it
    bounds memory for a large member.
    """
    from .scanner import read_text
    from .extract import extract, is_document

    try:
        with zipfile.ZipFile(abs_path) as archive:
            try:
                info = archive.getinfo(member)
            except KeyError:
                return None
            if info.file_size > max_bytes:
                return None

            suffix = os.path.splitext(member)[1]
            # The temp name is ours, never the member's - a member called
            # "../../evil" must not decide where anything is written.
            handle, temp_path = tempfile.mkstemp(suffix=suffix, prefix="rag-member-")
            os.close(handle)
            try:
                with archive.open(member) as source, open(temp_path, "wb") as target:
                    target.write(source.read())
                if is_document(temp_path):
                    return extract(temp_path)
                return read_text(temp_path, max_bytes)
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
    except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError) as exc:
        # NotImplementedError covers an unsupported compression method, and a
        # RuntimeError an encrypted entry: both are "skip this one", not fatal.
        log.warning("could not read %s from %s: %s", member, abs_path, exc)
        return None
