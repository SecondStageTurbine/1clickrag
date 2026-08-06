# SPDX-License-Identifier: MPL-2.0
"""Reading text off the page when the PDF has none.

A scanned document, a photographed report, a comic - the words are there on
the page, but as pixels. `pypdf` returns nothing for them, so without this they
are indexed as empty and the index quietly claims the corpus holds no such
text. This module turns those pages into text.

It is off by default (`RAG_OCR=1` to enable) and that is deliberate: OCR is
roughly a thousand times slower than reading a text layer - seconds per page
against microseconds - so a folder of scans is an overnight job, not a pause.
Starting one unasked on somebody's first index would be indefensible. What
happens instead with OCR off is a single log line naming how many pages were
skipped for having no text layer, so the choice is visible rather than silent.

Two decisions do most of the work:

**Never upsample.** These pages carry a bitmap at a fixed resolution and
rendering above it invents pixels. Measured on a 720px-wide page, rendering at
2x made recognition both slower (1.44s against 0.32s) and *worse* (confidence
0.77 against 0.81). Native scale is the fast setting and the accurate one.

**Slice tall pages, do not shrink them.** A webtoon page can be 20,000 pixels
tall. Fitting that into the detector's window squashes it to a couple of
hundred pixels wide and the text dissolves - measured on one such page,
squashing returned "LMNG CANNOTENTERUN-ESS SNENPERWSSICNBY" where slicing
returned "LIVING CANNOT ENTER UNLESS GIVEN PERMISSION BY", for 20% more time.
Bands also come out in reading order, which a whole-page detection pass does
not guarantee.

Engine: RapidOCR on onnxruntime, which this already depends on for embeddings.
Its models ride inside the wheel, so an air-gapped install gains OCR without
fetching anything.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger("rag.ocr")

_engine = None
_lock = threading.Lock()
_unavailable: str | None = None


def available() -> tuple[bool, str]:
    """Whether the OCR libraries import, and what is missing if not."""
    try:
        import numpy  # noqa: F401
        import pypdfium2  # noqa: F401
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
    except ImportError as exc:
        return False, str(exc)
    return True, ""


def _get_engine():
    """One engine for the process, built on first use.

    Loading costs a fraction of a second, but it happens inside whichever
    thread first meets a scanned page - the queue worker or a reindex - and
    both can be running, hence the lock.
    """
    global _engine, _unavailable
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        if _unavailable:
            return None
        try:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
            log.info("OCR engine ready")
        except Exception as exc:
            _unavailable = str(exc)
            log.warning("OCR unavailable: %s", exc)
            return None
    return _engine


def _key(line: str) -> str:
    """Compare readings loosely enough to match across a cut.

    The same bubble read in two bands rarely comes back byte-identical -
    "EXPECTED THIS" against "EXPECTEDTHIS" - because the crop takes a slice of
    the letters and the recogniser spaces words differently. Case and
    non-alphanumerics carry no meaning for this comparison, so they go.
    """
    return "".join(char for char in line.upper() if char.isalnum())


def _dedupe_seam(kept: list[str], incoming: list[str], window: int = 8) -> list[str]:
    """Drop lines the previous band already reported.

    Bands overlap on purpose, so a line sitting across a cut is read twice.
    Both ends are bounded: only the last few lines of the previous band can be
    duplicates, and only the first few of this one. Comparing everything
    against everything would delete a phrase legitimately repeated later on the
    page, and in dialogue those are common.
    """
    if not kept or not incoming:
        return incoming
    tail = {_key(line) for line in kept[-window:] if _key(line)}
    if not tail:
        return incoming
    out = []
    for index, line in enumerate(incoming):
        if index < window and _key(line) in tail:
            continue
        out.append(line)
    return out


def image_text(image, band: int = 2200, overlap: int = 200,
               min_confidence: float = 0.5) -> str:
    """Read one page image, slicing it into bands top to bottom."""
    engine = _get_engine()
    if engine is None:
        return ""

    import numpy as np

    width, height = image.size
    lines: list[str] = []
    top = 0
    while True:
        bottom = min(top + band, height)
        crop = image if (top == 0 and bottom == height) else image.crop(
            (0, top, width, bottom)
        )
        try:
            result, _ = engine(np.array(crop))
        except Exception as exc:
            log.debug("OCR failed on a band: %s", exc)
            result = None
        found = [
            item[1].strip()
            for item in (result or [])
            # The confidence filter earns its place on artwork: a detector
            # given a drawing finds "text" in hatching and panel borders, and
            # those readings come back with low scores and no meaning.
            if float(item[2]) >= min_confidence and item[1].strip()
        ]
        lines += _dedupe_seam(lines, found)
        if bottom >= height:
            break
        top = bottom - overlap

    return "\n".join(lines)


def pdf_page_text(path: str, page_numbers: set[int], cfg) -> dict[int, str]:
    """OCR the given 1-based pages of a PDF.

    Takes a set rather than doing the whole file because a PDF is often mixed -
    a typed report with scanned appendices - and the pages that already gave up
    a text layer need no second reading.
    """
    if not page_numbers:
        return {}
    engine = _get_engine()
    if engine is None:
        return {}

    try:
        import pypdfium2 as pdfium
    except ImportError:
        return {}

    out: dict[int, str] = {}
    try:
        doc = pdfium.PdfDocument(path)
    except Exception as exc:
        log.warning("could not open %s for OCR: %s", path, exc)
        return {}

    try:
        for number in sorted(page_numbers):
            if number < 1 or number > len(doc):
                continue
            page = doc[number - 1]
            width, height = page.get_size()
            # A guard, not a resolution policy: a corrupt or synthetic page can
            # claim an enormous size, and rendering it would exhaust memory
            # before OCR ever ran.
            if width * height > cfg.ocr_max_megapixels * 1_000_000:
                log.warning(
                    "%s page %d is %.0fx%.0f - too large to OCR, skipping",
                    path, number, width, height,
                )
                continue
            try:
                image = page.render(scale=1.0).to_pil().convert("RGB")
            except Exception as exc:
                log.debug("could not render %s page %d: %s", path, number, exc)
                continue
            text = image_text(
                image,
                band=cfg.ocr_band,
                overlap=cfg.ocr_overlap,
                min_confidence=cfg.ocr_min_confidence,
            )
            if text:
                out[number] = text
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out


def images_text(images, cfg) -> list[str]:
    """OCR a sequence of already-decoded page images, in order."""
    if _get_engine() is None:
        return []
    return [
        image_text(
            image,
            band=cfg.ocr_band,
            overlap=cfg.ocr_overlap,
            min_confidence=cfg.ocr_min_confidence,
        )
        for image in images
    ]
