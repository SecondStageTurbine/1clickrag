# SPDX-License-Identifier: MPL-2.0
"""Corpus-scoped citation previews. Never accepts arbitrary filesystem paths."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .archive import split_member
from .extract_cache import fingerprint_file
from .scanner import is_excluded


class PreviewRequest(BaseModel):
    path: str = Field(min_length=1, max_length=2048)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    version: str | None = Field(default=None, max_length=64)
    page: int = Field(default=1, ge=1, le=100000)


def version(info: dict) -> str:
    return hashlib.sha256(json.dumps(
        [info.get("source_hash"), info.get("mtime"), info.get("source"), info.get("content_version")],
        sort_keys=True,
    ).encode()).hexdigest()


def resolve_source(cfg, path: str) -> tuple[Path, bool]:
    # Reject Windows syntax on every platform, including drive/ADS/device paths.
    if "\\" in path or ":" in path or "\x00" in path or path.startswith("/"):
        raise HTTPException(403, "Invalid document path.")
    member = split_member(path)
    source = member[0] if member else path
    if any(p in ("", ".", "..") for p in source.split("/")) or is_excluded(source, cfg):
        raise HTTPException(403, "Document is outside the permitted corpus.")
    root = Path(cfg.repo_path).resolve()
    target = (root / source).resolve()
    if not target.is_relative_to(root) or target == root:
        raise HTTPException(403, "Document is outside the permitted corpus.")
    return target, bool(member)


def prepare(cfg, store, request: PreviewRequest) -> tuple[dict, Path]:
    target, virtual = resolve_source(cfg, request.path)
    if request.end_line < request.start_line or request.end_line - request.start_line > 10000:
        raise HTTPException(422, "Invalid citation line range.")
    try:
        chunks = store.document_chunks(request.path)
    except ValueError as exc:
        raise HTTPException(413, str(exc)) from exc
    if not chunks:
        raise HTTPException(404, "Document is no longer in the index.")
    current_version = version(chunks[0])
    if request.version and request.version != current_version:
        raise HTTPException(409, "This document was reindexed after the answer. Run the question again for current citations.")
    lines = {}
    for chunk in chunks:
        if version(chunk) != current_version:
            raise HTTPException(409, "Document indexing is in progress. Try again shortly.")
        for number, text in enumerate(chunk.get("text", "").splitlines(), int(chunk["start_line"])):
            if number in lines and lines[number] != text:
                raise HTTPException(409, "Indexed passages disagree. Reindex this document before previewing it.")
            lines[number] = text
            if len(lines) > 100000:
                raise HTTPException(413, "Document has too many lines for the preview.")
    selected = {n: t for n, t in lines.items() if request.start_line <= n <= request.end_line}
    if not selected:
        raise HTTPException(404, "The cited passage is no longer indexed.")
    warning = ""
    fresh = False
    if not target.is_file():
        warning = "The original file is unavailable. Showing the indexed text."
    elif target.stat().st_size > min(cfg.max_file_bytes, 100_000_000):
        warning = "The original file exceeds the preview size limit. Showing the indexed text."
    elif chunks[0].get("source_hash"):
        fresh = fingerprint_file(str(target)) == chunks[0]["source_hash"]
        if not fresh:
            warning = "The document has changed since indexing. Showing indexed text; original-page highlighting is disabled."
    elif chunks[0].get("mtime") is not None:
        fresh = round(target.stat().st_mtime, 3) == chunks[0]["mtime"]
        warning = ("Older index: file timestamp matches, but no content fingerprint is available."
                   if fresh else "The document has changed since indexing. Showing the indexed text.")
    else:
        warning = "The indexed file version is unknown. Showing the indexed text."
    if virtual:
        warning = (warning + " Archive members and embedded attachments currently use indexed-text previews.").strip()
    # Map extracted line numbers to the physical PDF page markers preserved at ingest.
    page, page_text = None, {}
    for number in sorted(lines):
        marker = re.fullmatch(r"\[page (\d+)\]", lines[number].strip())
        if marker:
            page = int(marker.group(1))
        elif number in selected and page is not None:
            page_text.setdefault(page, []).append(lines[number])
    pdf = fresh and not virtual and target.suffix.lower() == ".pdf"
    return {
        "path": request.path, "version": current_version,
        "kind": "pdf" if pdf else "text", "warning": warning,
        "start_line": request.start_line, "end_line": request.end_line,
        "lines": [{"n": n, "text": lines[n]} for n in sorted(lines)],
        "initial_page": min(page_text) if page_text else 1,
        "cited_pages": sorted(page_text),
        "page_text": page_text,
    }, target


def router(cfg, store) -> APIRouter:
    routes = APIRouter()

    @routes.post("/preview")
    def preview(request: PreviewRequest):
        result, _ = prepare(cfg, store, request)
        result.pop("page_text")
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @routes.post("/preview/page")
    def preview_page(request: PreviewRequest):
        result, target = prepare(cfg, store, request)
        if result["kind"] != "pdf":
            raise HTTPException(409, result["warning"] or "Original PDF preview is unavailable.")
        # PDFium is not thread-safe. Isolate each bounded rendering job from
        # the API and concurrent OCR ingestion, and enforce a timeout.
        payload = {"path": str(target), "page": request.page,
                   "text": "\n".join(result["page_text"].get(request.page, []))}
        try:
            process = subprocess.run(
                [sys.executable, "-m", "app.preview_pdf"],
                input=json.dumps(payload), capture_output=True, text=True,
                encoding="utf-8", timeout=30,
                cwd=str(Path(__file__).resolve().parent.parent),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if process.returncode:
                raise ValueError("PDF rendering failed")
            rendered = json.loads(process.stdout)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(422, "This PDF page could not be rendered. Use Indexed text to inspect the citation.") from exc
        # A writer may have replaced the file while the renderer was running.
        again, _ = prepare(cfg, store, request)
        if again["kind"] != "pdf" or again["version"] != result["version"]:
            raise HTTPException(409, "The document changed while opening it. Run the question again.")
        return JSONResponse(rendered, headers={"Cache-Control": "no-store"})

    return routes
