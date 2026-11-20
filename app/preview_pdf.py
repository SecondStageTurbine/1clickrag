# SPDX-License-Identifier: MPL-2.0
"""Isolated local PDF page renderer; stdout is a single JSON response."""

import base64
from contextlib import closing
import io
import json
import math
import sys


def compact(text):
    positions, chars = [], []
    for index, char in enumerate(text):
        if not char.isspace():
            positions.append(index)
            chars.append(char)
    return "".join(chars), positions


def render(path, page_number, quote):
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(path) as document:
        if not 1 <= page_number <= len(document):
            raise ValueError("Page is out of range")
        with closing(document[page_number - 1]) as page:
            width, height = page.get_size()
            if width <= 0 or height <= 0:
                raise ValueError("Invalid page dimensions")
            scale = min(2, 1600 / width, math.sqrt(3_000_000 / (width * height)))
            with closing(page.render(scale=scale)) as bitmap, closing(page.get_textpage()) as textpage:
                image = bitmap.to_pil()
                stream = io.BytesIO()
                image.save(stream, format="PNG")
                boxes = []
                haystack, positions = compact(textpage.get_text_range())
                needle, _ = compact(quote)
                start = haystack.find(needle) if len(needle) >= 8 and all(ord(c) <= 0xffff for c in haystack) else -1
                # Exact whitespace-normalized and unique matching only. No
                # fuzzy match can silently become an authoritative highlight.
                if start >= 0 and haystack.find(needle, start + 1) < 0:
                    first = pdfium.raw.FPDFText_GetCharIndexFromTextIndex(textpage, positions[start])
                    last = pdfium.raw.FPDFText_GetCharIndexFromTextIndex(textpage, positions[start + len(needle) - 1])
                    verified = first >= 0 and last >= first and compact(textpage.get_text_range(first, last - first + 1))[0] == needle
                    count = textpage.count_rects(first, last - first + 1) if verified else 0
                    converter = bitmap.get_posconv(page)
                    for index in range(count):
                        left, bottom, right, top = textpage.get_rect(index)
                        corners = [converter.to_bitmap(x, y) for x, y in
                                   ((left, bottom), (left, top), (right, bottom), (right, top))]
                        xs, ys = zip(*corners)
                        boxes.append({"x": min(xs) / bitmap.width, "y": min(ys) / bitmap.height,
                                      "w": (max(xs) - min(xs)) / bitmap.width,
                                      "h": (max(ys) - min(ys)) / bitmap.height})
                return {"page": page_number, "pages": len(document), "boxes": boxes,
                        "image": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(),
                        "matched": bool(boxes)}


if __name__ == "__main__":
    request = json.load(sys.stdin)
    print(json.dumps(render(request["path"], request["page"], request["text"])))
