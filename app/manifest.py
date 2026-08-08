# SPDX-License-Identifier: MPL-2.0
"""What built this index, so a changed setting cannot go unnoticed.

Several settings here decide what ends up in the vectors: the embedding model,
the chunk size, which label rides along with the text. Change one and the
chunks already indexed keep whatever the old setting produced, while everything
indexed afterwards uses the new one. The result is an index built two different
ways, and nothing said so - search kept working, returned slightly worse
answers, and blamed nothing.

That is not hypothetical. RAG_EMBED_LABEL was added a day before this, and
flipping it silently leaves exactly that mess.

So the settings are written down next to the index when it is built, and
compared against the running configuration at startup. Drift is reported rather
than repaired: rebuilding an index because a setting moved is minutes to hours
of someone's time, and that is their decision to make, not this file's.

Two kinds of drift, because the consequences differ:

**rebuild** - the vectors themselves were made differently, so old and new
chunks are not comparable and a full reindex is the only fix.

**content** - what got *extracted* changed: OCR turned on, a new text
extension, archives disabled. Old files keep their old text until each is
indexed again, so this corrects itself file by file, and a reindex only hurries
it along.

No manifest at all means an index built before this existed. That is reported
as unknown and never warned about: nagging someone into a two-hour rebuild to
satisfy a bookkeeping file would be a poor trade.
"""

from __future__ import annotations

import json
import logging
import os
import time

log = logging.getLogger("rag.manifest")

FORMAT = 1

# Settings baked into every vector. Different values mean chunks that cannot
# meaningfully be compared with each other.
REBUILD_FIELDS = (
    "embed_model",
    "embed_backend",
    "vector_dim",
    "embed_path",
    "embed_label",
    "chunk_chars",
    "chunk_overlap_lines",
)

# Settings that decide what text was extracted in the first place. Old files
# keep the old text until they are indexed again.
CONTENT_FIELDS = (
    "ocr",
    "ocr_min_chars",
    "ocr_band",
    "ocr_overlap",
    "ocr_min_confidence",
    "archives",
    "extra_text_exts",
    "pdf_keep_boilerplate",
)

# Noun phrases, because they are read inside a sentence: "the embedding model
# ('a' -> 'b') changed since the index was built".
LABELS = {
    "embed_model": "the embedding model",
    "embed_backend": "the embedding backend",
    "vector_dim": "the vector width",
    "embed_path": "how much of the path is embedded",
    "embed_label": "the embedded label",
    "chunk_chars": "the chunk size",
    "chunk_overlap_lines": "the chunk overlap",
    "ocr": "OCR",
    "ocr_min_chars": "the OCR threshold",
    "ocr_band": "the OCR band height",
    "ocr_overlap": "the OCR band overlap",
    "ocr_min_confidence": "the OCR confidence floor",
    "archives": "reading inside archives",
    "extra_text_exts": "the extra text extensions",
    "pdf_keep_boilerplate": "keeping PDF headers and footers",
}


def current(cfg, vector_dim: int | None = None) -> dict:
    """The settings this process would build an index with."""
    return {
        "format": FORMAT,
        "written_at": time.time(),
        "collection": cfg.collection,
        "embed_model": cfg.embed_model,
        "embed_backend": cfg.embed_backend,
        "vector_dim": vector_dim,
        "embed_path": cfg.embed_path,
        "embed_label": cfg.embed_label,
        "chunk_chars": cfg.chunk_chars,
        "chunk_overlap_lines": cfg.chunk_overlap_lines,
        "ocr": bool(cfg.ocr),
        "ocr_min_chars": cfg.ocr_min_chars,
        "ocr_band": cfg.ocr_band,
        "ocr_overlap": cfg.ocr_overlap,
        "ocr_min_confidence": cfg.ocr_min_confidence,
        "archives": bool(cfg.archives),
        # Sorted so a dict's iteration order cannot masquerade as a change.
        "extra_text_exts": sorted(cfg.extra_text_exts.items()),
        "pdf_keep_boilerplate": bool(os.environ.get("RAG_PDF_KEEP_BOILERPLATE")),
    }


def load(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def save(path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Written beside and moved into place: a half-written manifest read on
        # the next start would report drift that never happened.
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    except OSError as exc:
        log.warning("could not record the index manifest: %s", exc)


def drift(recorded: dict | None, now: dict) -> dict:
    """Which settings have moved since the index was built."""
    if not recorded:
        return {"known": False, "rebuild": [], "content": []}

    def changed(fields):
        out = []
        for field in fields:
            was, is_now = recorded.get(field), now.get(field)
            # A field absent from an older manifest is unknown, not different.
            if field not in recorded or was == is_now:
                continue
            # vector_dim is only meaningful once measured; None means the
            # embedder had not loaded when the manifest was written.
            if field == "vector_dim" and (was is None or is_now is None):
                continue
            out.append({"setting": field, "label": LABELS.get(field, field),
                        "was": was, "now": is_now})
        return out

    return {
        "known": True,
        "rebuild": changed(REBUILD_FIELDS),
        "content": changed(CONTENT_FIELDS),
    }


def describe(result: dict) -> str | None:
    """One line for the log and the browser, or None when nothing moved."""
    if not result.get("known"):
        return None
    rebuild, content = result["rebuild"], result["content"]
    if not rebuild and not content:
        return None

    def phrase(items):
        return ", ".join(f"{i['label']} ({i['was']!r} -> {i['now']!r})" for i in items)

    if rebuild:
        message = (
            f"The index was built with different settings: {phrase(rebuild)}. "
            f"Chunks indexed before and after this change are not comparable - "
            f"run `reindex -Full` to rebuild them consistently."
        )
        if content:
            message += (
                f" Also changed, affecting only newly indexed files: "
                f"{phrase(content)}."
            )
        return message
    return (
        f"Changed since the index was built: {phrase(content)}. "
        f"Files already indexed keep their old text until each is indexed "
        f"again; `reindex -Full` applies it to all of them at once."
    )
