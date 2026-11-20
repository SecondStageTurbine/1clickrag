# SPDX-License-Identifier: MPL-2.0
"""Discover and read spreadsheet files embedded in PDFs.

PDFs carry files in two common places: the document-level EmbeddedFiles name
tree and page-level /FileAttachment annotations.  pypdf's public attachments
mapping covers the former; supplier documents frequently use the latter, so we
inspect both.  Members are returned as stable virtual names and never written
using the supplier-controlled filename.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator

log = logging.getLogger("rag.pdf_attachments")

SUPPORTED_EXTS = frozenset({".xlsx", ".xlsm", ".xls", ".csv"})


@dataclass(frozen=True)
class EmbeddedFile:
    name: str
    data: bytes


class AttachmentStaging:
    """Bounded, ingest-scoped storage for one-pass PDF attachment reads."""

    def __init__(self, max_temp_bytes: int) -> None:
        self.max_temp_bytes = max(0, max_temp_bytes)
        self.temp_bytes = 0
        self._temporary = tempfile.TemporaryDirectory(prefix="rag-pdf-attachments-")
        self.temp_dir = self._temporary.name
        self._batches: dict[str, list[tuple[str, int]]] = {}
        self._files: dict[tuple[str, str], str] = {}
        self._stage_member: Callable[[str, str], bool] = lambda _path, _name: True
        self._temp_limit_warned = False

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def has_batch(self, path: str) -> bool:
        return self._key(path) in self._batches

    def set_member_filter(self, predicate: Callable[[str, str], bool]) -> None:
        self._stage_member = predicate

    def stage(
        self,
        path: str,
        max_member_bytes: int,
        max_members: int,
        max_total_bytes: int,
    ) -> list[tuple[str, int]]:
        key = self._key(path)
        if key in self._batches:
            return list(self._batches[key])

        listed: list[tuple[str, int]] = []
        self._batches[key] = listed
        extracted_bytes = 0
        for item in _iter_embedded_files(path):
            size = len(item.data)
            if size > max_member_bytes:
                log.warning("skipping oversized PDF attachment %s in %s", item.name, path)
                continue
            if len(listed) >= max_members:
                log.warning("PDF %s exceeds the %d attachment limit", path, max_members)
                break
            if extracted_bytes + size > max_total_bytes:
                log.warning(
                    "PDF %s exceeds the %d-byte total attachment limit",
                    path,
                    max_total_bytes,
                )
                break
            listed.append((item.name, size))
            extracted_bytes += size
            if not self._stage_member(path, item.name):
                continue
            if self.temp_bytes + size > self.max_temp_bytes:
                if not self._temp_limit_warned:
                    log.warning(
                        "PDF attachment staging reached its %d-byte "
                        "temporary-storage limit; remaining changed attachments "
                        "will be skipped",
                        self.max_temp_bytes,
                    )
                    self._temp_limit_warned = True
                continue

            suffix = os.path.splitext(item.name)[1]
            handle, staged_path = tempfile.mkstemp(
                suffix=suffix, prefix="member-", dir=self.temp_dir
            )
            try:
                with os.fdopen(handle, "wb") as target:
                    target.write(item.data)
            except Exception:
                try:
                    os.unlink(staged_path)
                except OSError:
                    pass
                raise
            self._files[(key, item.name)] = staged_path
            self.temp_bytes += size
        return list(listed)

    def read(self, path: str, member: str, max_bytes: int) -> bytes | None:
        staged_path = self.path(path, member, max_bytes)
        if staged_path is None:
            return None
        try:
            with open(staged_path, "rb") as source:
                return source.read()
        except OSError:
            return None

    def path(self, path: str, member: str, max_bytes: int) -> str | None:
        staged_path = self._files.get((self._key(path), member))
        if staged_path is None:
            return None
        try:
            if os.path.getsize(staged_path) > max_bytes:
                return None
        except OSError:
            return None
        return staged_path

    def close(self) -> None:
        self._temporary.cleanup()


_ACTIVE_STAGING: ContextVar[AttachmentStaging | None] = ContextVar(
    "rag_pdf_attachment_staging", default=None
)


@contextmanager
def attachment_staging(max_temp_bytes: int) -> Iterator[AttachmentStaging]:
    """Make one-pass attachment batches available to the ingest call stack."""
    staging = AttachmentStaging(max_temp_bytes)
    token = _ACTIVE_STAGING.set(staging)
    try:
        yield staging
    finally:
        _ACTIVE_STAGING.reset(token)
        staging.close()


def _resolve(value):
    return value.get_object() if hasattr(value, "get_object") else value


def _safe_name(value: object) -> str:
    # A PDF filename is display metadata, never a destination path.  Keeping
    # only the basename prevents a misleading ../../ path in citations too.
    name = str(value or "embedded-file").replace("\\", "/").split("/")[-1]
    return name.strip().strip(".") or "embedded-file"


def _raw_entries(reader) -> Iterator[tuple[str, bytes]]:
    try:
        for name, versions in reader.attachments.items():
            for data in versions:
                yield _safe_name(name), bytes(data)
    except Exception as exc:
        log.warning("could not read PDF document attachments: %s", exc)

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            annotations = _resolve(page.get("/Annots")) or []
            for reference in annotations:
                annotation = _resolve(reference)
                if str(annotation.get("/Subtype")) != "/FileAttachment":
                    continue
                filespec = _resolve(annotation.get("/FS")) or {}
                name = filespec.get("/UF") or filespec.get("/F")
                embedded = _resolve(filespec.get("/EF")) or {}
                stream = _resolve(embedded.get("/UF") or embedded.get("/F"))
                if stream is None or not hasattr(stream, "get_data"):
                    log.warning("PDF page %d has a file attachment with no data", page_number)
                    continue
                yield _safe_name(name), bytes(stream.get_data())
        except Exception as exc:
            log.warning("could not inspect PDF page %d attachments: %s", page_number, exc)


def _iter_embedded_files(path: str) -> Iterator[EmbeddedFile]:
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf is not installed - PDF attachments are unavailable")
        return
    try:
        reader = PdfReader(path)
    except Exception as exc:
        log.warning("could not open PDF attachments in %s: %s", path, exc)
        return

    fingerprints: set[tuple[str, str]] = set()
    name_counts: dict[str, int] = {}
    for original_name, data in _raw_entries(reader):
        stem, ext = os.path.splitext(original_name)
        ext = ext.lower()
        if ext not in SUPPORTED_EXTS:
            continue
        fingerprint = (original_name.casefold(), hashlib.sha256(data).hexdigest())
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)

        count = name_counts.get(original_name.casefold(), 0) + 1
        name_counts[original_name.casefold()] = count
        name = original_name if count == 1 else f"{stem} ({count}){ext}"
        yield EmbeddedFile(name, data)


def embedded_files(path: str) -> list[EmbeddedFile]:
    """Return supported, de-duplicated spreadsheet attachments from a PDF."""
    return list(_iter_embedded_files(path))


def list_attachments(
    path: str,
    max_container_bytes: int,
    max_member_bytes: int,
    max_members: int,
    max_total_bytes: int | None = None,
) -> list[tuple[str, int]]:
    try:
        if os.path.getsize(path) > max_container_bytes:
            log.warning("skipping attachments in oversized PDF %s", path)
            return []
    except OSError:
        return []

    total_limit = max_total_bytes if max_total_bytes is not None else max_container_bytes
    staging = _ACTIVE_STAGING.get()
    if staging is not None:
        return staging.stage(path, max_member_bytes, max_members, total_limit)

    listed: list[tuple[str, int]] = []
    extracted_bytes = 0
    for item in _iter_embedded_files(path):
        size = len(item.data)
        if size > max_member_bytes:
            log.warning("skipping oversized PDF attachment %s in %s", item.name, path)
            continue
        if len(listed) >= max_members:
            log.warning("PDF %s exceeds the %d attachment limit", path, max_members)
            break
        if extracted_bytes + size > total_limit:
            log.warning(
                "PDF %s exceeds the %d-byte total attachment limit", path, total_limit
            )
            break
        listed.append((item.name, size))
        extracted_bytes += size
    return listed


def read_attachment(path: str, member: str, max_bytes: int) -> bytes | None:
    staging = _ACTIVE_STAGING.get()
    if staging is not None and staging.has_batch(path):
        return staging.read(path, member, max_bytes)
    for item in _iter_embedded_files(path):
        if item.name == member:
            if len(item.data) > max_bytes:
                return None
            return item.data
    return None


def staged_attachment_path(path: str, member: str, max_bytes: int) -> str | None:
    staging = _ACTIVE_STAGING.get()
    if staging is None or not staging.has_batch(path):
        return None
    return staging.path(path, member, max_bytes)
