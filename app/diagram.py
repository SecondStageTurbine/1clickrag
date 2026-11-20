# SPDX-License-Identifier: MPL-2.0
"""Conservative flowchart extraction from PDF page images.

This is diagram-to-graph extraction, not general image captioning.  It detects
simple flowchart nodes (rectangles, diamonds, circles and off-page connectors),
reads text inside them with the existing local OCR engine, and accepts a
directed edge only when a line terminates at a triangular arrowhead close to
another node.

All coordinates are normalized to the rendered page, making the evidence
stable across render resolutions.  The resulting text is appended to the PDF's
normal extraction and indexed by the same vector, keyword and relationship
paths as every other passage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone

log = logging.getLogger("rag.diagram")

BRANCH_LABELS = frozenset({"yes", "no", "true", "false"})
PAGE_TARGET = r"(?P<target>[A-Za-z0-9][A-Za-z0-9.-]{0,15})"
PAGE_LOCATOR = r"(?P<locator>page|pg\.?|sheet|sht\.?)"
OUTGOING_PAGE = re.compile(
    rf"\b(?:to|see|go\s+to|next|continue(?:d|s)?(?:\s+on)?)\s+"
    rf"{PAGE_LOCATOR}\s*[:#-]?\s*{PAGE_TARGET}\b",
    re.IGNORECASE,
)
INCOMING_PAGE = re.compile(
    rf"\b(?:from|continue(?:d|s)?\s+from)\s+"
    rf"{PAGE_LOCATOR}\s*[:#-]?\s*{PAGE_TARGET}\b",
    re.IGNORECASE,
)
FIGURE_SHEET_HEADING = re.compile(
    r"\bfigure\s+([A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*)[^\r\n]*?"
    r"\(\s*sheet\s+(\d+)(?:\s+of\s+\d+)?\s*\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TextItem:
    text: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class DiagramNode:
    node_id: str
    text: str
    shape: str
    semantic_type: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class DiagramEdge:
    source_id: str
    target_id: str
    relationship: str
    label: str
    confidence: float
    tail: tuple[float, float]
    head: tuple[float, float]


@dataclass(frozen=True)
class PageReference:
    reference_id: str
    page_number: int
    anchor: DiagramNode
    direction: str
    referenced_page: int | None
    target_label: str
    connector_key: str
    raw_text: str
    valid: bool
    reason: str


AnalyzedPage = tuple[tuple[int, int], list[DiagramNode], list[DiagramEdge]]


def available() -> tuple[bool, str]:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import pypdfium2  # noqa: F401
        from PIL import Image  # noqa: F401
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
    except ImportError as exc:
        return False, str(exc)
    return True, ""


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    if not intersection:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / max(union, 1)


def _shapes(image) -> list[tuple[str, tuple[int, int, int, int], float]]:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    page_area = width * height
    minimum = max(500, int(page_area * 0.0007))
    maximum = int(page_area * 0.45)
    candidates: list[tuple[str, tuple[int, int, int, int], float]] = []

    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < minimum or area > maximum:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 45 or h < 25 or w > width * 0.95 or h > height * 0.95:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        shape = ""
        confidence = 0.0
        if len(polygon) == 4:
            points = [(int(p[0][0]), int(p[0][1])) for p in polygon]
            top = min(points, key=lambda p: p[1])
            bottom = max(points, key=lambda p: p[1])
            left = min(points, key=lambda p: p[0])
            right = max(points, key=lambda p: p[0])
            centered = (
                abs(top[0] - (x + w / 2)) < w * 0.25
                and abs(bottom[0] - (x + w / 2)) < w * 0.25
                and abs(left[1] - (y + h / 2)) < h * 0.25
                and abs(right[1] - (y + h / 2)) < h * 0.25
            )
            shape = "diamond" if centered else "rectangle"
            confidence = 0.91 if centered else 0.94
        elif len(polygon) in {5, 6}:
            aspect = w / max(h, 1)
            if aspect >= 2.5:
                # A numbered action box often shares its top border with the
                # little step-number cell, making its outer contour 5/6-sided.
                shape = "rectangle"
                confidence = 0.90
            elif aspect >= 1.6:
                # Maintenance flowcharts commonly use a wide hexagon for a
                # test/decision rather than the conventional diamond.
                shape = "hexagon"
                confidence = 0.90
            else:
                shape = "off_page_connector"
                confidence = 0.82
        elif len(polygon) >= 7:
            circularity = 4 * math.pi * area / max(perimeter * perimeter, 1)
            if circularity >= 0.72 and 0.65 <= w / max(h, 1) <= 1.5:
                shape = "circle"
                confidence = min(0.95, 0.65 + circularity * 0.3)
            elif (
                len(polygon) <= 10
                and w / max(h, 1) >= 1.5
                and (w * h) / page_area <= 0.045
            ):
                # A step-number cell and attached connector can add vertices
                # to an otherwise ordinary decision hexagon.
                shape = "hexagon"
                confidence = 0.86
        if shape:
            candidates.append((shape, (x, y, x + w, y + h), confidence))

    # Borders often yield inner and outer contours. Keep the strongest single
    # observation so one drawn box cannot become two graph nodes.
    # Drop number cells and other small shapes contained by an outer node.
    def area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def covered(inner, outer):
        overlap = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0])) * max(
            0, min(inner[3], outer[3]) - max(inner[1], outer[1])
        )
        return overlap / max(area(inner), 1)

    candidates = [
        item for item in candidates
        if not any(
            area(other[1]) >= area(item[1]) * 1.8
            and covered(item[1], other[1]) >= 0.82
            for other in candidates if other is not item
        )
    ]
    kept: list[tuple[str, tuple[int, int, int, int], float]] = []
    for item in sorted(candidates, key=lambda row: -row[2]):
        if any(_iou(item[1], other[1]) > 0.82 for other in kept):
            continue
        kept.append(item)
    return kept


def _arrowheads(image) -> list[tuple[float, float, float]]:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)[1]
    # A filled arrowhead often touches both its connector and the destination
    # border, producing one large contour. A one-pixel erosion disconnects the
    # thin strokes while leaving the triangular filled area recognizable.
    separated = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    page_area = gray.shape[0] * gray.shape[1]
    heads: list[tuple[float, float, float]] = []
    # Small printed arrowheads disappear under erosion; large ones joined to a
    # connector line need it. Inspect both masks and deduplicate their centers.
    for mask in (binary, separated):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area = abs(cv2.contourArea(contour))
            if contour_area < max(6, page_area * 0.000003) or contour_area > page_area * 0.004:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.08 * perimeter, True)
            if len(polygon) != 3:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if max(w, h) > max(gray.shape) * 0.08 or min(w, h) < 3:
                continue
            points = [tuple(map(float, point[0])) for point in polygon]
            # The vertex farthest from the opposite side is the arrow tip. Its
            # centroid can sit several pixels back and even closer to the box
            # the arrow came from on small print.
            tip = max(points, key=lambda point: max(
                math.hypot(
                    point[0] - (other[0] + third[0]) / 2,
                    point[1] - (other[1] + third[1]) / 2,
                )
                for other in points if other != point
                for third in points if third != point and third != other
            ))
            candidate = (
                tip[0], tip[1],
                float(max(w, h)),
            )
            if any(math.hypot(candidate[0] - old[0], candidate[1] - old[1])
                   <= max(6, min(candidate[2], old[2]) * 0.8) for old in heads):
                continue
            heads.append(candidate)
    return heads


def _segments(image) -> list[tuple[int, int, int, int]]:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(
        edges, 1, math.pi / 180, threshold=15,
        minLineLength=max(15, min(gray.shape) // 50), maxLineGap=15,
    )
    if lines is None:
        return []
    flat = lines.reshape(-1, 4)
    return [tuple(map(int, line)) for line in flat]


def _ocr_items(image, min_confidence: float) -> list[TextItem]:
    from .ocr import _get_engine
    import numpy as np

    engine = _get_engine()
    if engine is None:
        return []
    try:
        result, _ = engine(np.array(image.convert("RGB")))
    except Exception as exc:
        log.debug("diagram OCR failed: %s", exc)
        return []
    items: list[TextItem] = []
    for box, text, confidence in result or []:
        if float(confidence) < min_confidence or not str(text).strip():
            continue
        xs = [int(point[0]) for point in box]
        ys = [int(point[1]) for point in box]
        items.append(TextItem(str(text).strip(), float(confidence),
                              (min(xs), min(ys), max(xs), max(ys))))
    return items


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _contains(box: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _has_page_reference(text: str) -> bool:
    return bool(OUTGOING_PAGE.search(text) or INCOMING_PAGE.search(text))


def _distance_to_box(point: tuple[float, float], box: tuple[int, int, int, int]) -> float:
    x, y = point
    dx = max(box[0] - x, 0, x - box[2])
    dy = max(box[1] - y, 0, y - box[3])
    return math.hypot(dx, dy)


def _distance_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dy * dy
    ratio = 0.0 if not length2 else max(0.0, min(
        1.0,
        ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length2,
    ))
    nearest = (start[0] + ratio * dx, start[1] + ratio * dy)
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def _source_side_segment(
    path: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if len(path) < 2:
        return None
    start, end = path[-1], path[-2]
    return start, (
        start[0] + (end[0] - start[0]) * 0.8,
        start[1] + (end[1] - start[1]) * 0.8,
    )


def _branch_relationship(label: str) -> str:
    return (
        "yes_branch_to"
        if label.strip().casefold() in {"yes", "true"}
        else "no_branch_to"
    )


def _assign_branch_labels(
    edges: list[DiagramEdge],
    paths: list[tuple[list[tuple[float, float]], float]],
    nodes: list[DiagramNode],
    labels: list[TextItem],
) -> list[DiagramEdge]:
    """Assign each OCR branch label to at most one outgoing decision edge."""
    node_by_id = {node.node_id: node for node in nodes}
    candidates: list[tuple[float, int, int]] = []
    for edge_index, (edge, (path, tolerance)) in enumerate(zip(edges, paths)):
        source = node_by_id[edge.source_id]
        segment = _source_side_segment(path)
        if edge.label or source.semantic_type != "decision" or segment is None:
            continue
        # Paths are stored arrowhead-to-source, so the final segment is the
        # branch as it leaves the decision node. Limit matching to its
        # source-side portion so a label on a crossing/adjacent branch cannot
        # be borrowed merely because both connectors are close downstream.
        start, source_side_end = segment
        for label_index, label in enumerate(labels):
            distance = _distance_to_segment(
                _center(label.box), start, source_side_end
            )
            if distance <= max(12.0, tolerance * 0.7):
                candidates.append((distance, edge_index, label_index))

    assigned_edges: set[int] = set()
    assigned_labels: set[int] = set()
    for _distance, edge_index, label_index in sorted(candidates):
        if edge_index in assigned_edges or label_index in assigned_labels:
            continue
        label = labels[label_index].text
        edges[edge_index] = replace(
            edges[edge_index], relationship=_branch_relationship(label), label=label
        )
        assigned_edges.add(edge_index)
        assigned_labels.add(label_index)
    return edges


def _targeted_branch_label(
    image,
    path: list[tuple[float, float]],
    tolerance: float,
    min_confidence: float,
) -> TextItem | None:
    """Re-read a narrow source-side branch corridor for a small YES/NO word."""
    segment = _source_side_segment(path)
    if segment is None:
        return None
    start, end = segment
    margin = max(18.0, tolerance * 0.75)
    left = max(0, int(min(start[0], end[0]) - margin))
    top = max(0, int(min(start[1], end[1]) - margin))
    right = min(image.width, int(max(start[0], end[0]) + margin))
    bottom = min(image.height, int(max(start[1], end[1]) + margin))
    if right <= left or bottom <= top:
        return None

    scale = 3
    crop = image.crop((left, top, right, bottom)).resize(
        ((right - left) * scale, (bottom - top) * scale)
    )
    candidates = []
    for item in _ocr_items(crop, min(min_confidence, 0.4)):
        if item.text.strip().casefold() not in BRANCH_LABELS:
            continue
        box = tuple(
            int(value / scale) + (left if index % 2 == 0 else top)
            for index, value in enumerate(item.box)
        )
        transformed = TextItem(item.text, item.confidence, box)
        distance = _distance_to_segment(_center(box), start, end)
        if distance <= max(10.0, tolerance * 0.55):
            candidates.append((distance, transformed))
    return min(candidates, key=lambda row: row[0])[1] if candidates else None


def analyze_image(
    image,
    min_confidence: float = 0.55,
    text_items: list[TextItem] | None = None,
) -> tuple[list[DiagramNode], list[DiagramEdge]]:
    """Detect simple labeled flowchart nodes and directed edges in one image."""
    shapes = _shapes(image)
    heads = _arrowheads(image)
    if len(shapes) < 2 or not heads:
        return [], []
    texts = text_items if text_items is not None else _ocr_items(image, min_confidence)
    nodes: list[DiagramNode] = []
    assigned_text: set[int] = set()
    width, height = image.size
    diagonal = math.hypot(width, height)
    page_reference_tolerance = max(70, diagonal * 0.045)
    for shape, box, shape_confidence in sorted(shapes, key=lambda row: (row[1][1], row[1][0])):
        inside = [
            (index, item) for index, item in enumerate(texts)
            if index not in assigned_text
            and item.text.strip().casefold() not in BRANCH_LABELS
            and _contains(box, _center(item.box))
        ]
        if shape == "off_page_connector":
            # Some drawings put the page instruction immediately beside an
            # otherwise blank connector. Associate only explicit page text,
            # never arbitrary nearby prose, and consume the closest match.
            inside_indices = {index for index, _ in inside}
            nearby = [
                (index, item) for index, item in enumerate(texts)
                if index not in assigned_text
                and index not in inside_indices
                and _has_page_reference(item.text)
                and _distance_to_box(_center(item.box), box)
                <= page_reference_tolerance
            ]
            if nearby:
                inside.append(min(
                    nearby,
                    key=lambda pair: _distance_to_box(_center(pair[1].box), box),
                ))
        if not inside:
            continue
        inside.sort(key=lambda pair: (pair[1].box[1], pair[1].box[0]))
        assigned_text.update(index for index, _ in inside)
        text = " ".join(item.text for _, item in inside).strip()
        ocr_confidence = sum(item.confidence for _, item in inside) / len(inside)
        semantic = {"diamond": "decision", "hexagon": "decision", "rectangle": "process",
                    "circle": "connector",
                    "off_page_connector": "off_page_connector"}[shape]
        nodes.append(DiagramNode(
            f"N{len(nodes) + 1}", text, shape, semantic,
            round(min(shape_confidence, ocr_confidence), 3), box,
        ))
    # Continuation labels are often printed beside a bare arrow rather than
    # inside a drawn connector. Preserve them as nodes so the arrow and the
    # cross-sheet relationship can both be represented.
    for index, item in enumerate(texts):
        if index in assigned_text or not _has_page_reference(item.text):
            continue
        nodes.append(DiagramNode(
            f"N{len(nodes) + 1}", item.text, "off_page_connector",
            "off_page_connector", round(item.confidence, 3), item.box,
        ))
        assigned_text.add(index)
    if len(nodes) < 2:
        return nodes, []

    # Triangle-like glyphs inside OCR text (A, 4, warning symbols) look like
    # tiny arrowheads in isolation. A real connector head sits outside a node
    # or on its border, never well inside the text area.
    def inside_node_text(point, box):
        margin = max(5, min(box[2] - box[0], box[3] - box[1]) * 0.08)
        return (
            box[0] + margin < point[0] < box[2] - margin
            and box[1] + margin < point[1] < box[3] - margin
        )

    heads = [
        head for head in heads
        if not any(inside_node_text(head[:2], node.box) for node in nodes)
    ]

    segments = []
    for segment in _segments(image):
        x1, y1, x2, y2 = segment
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        length = math.hypot(dx, dy)
        # These manuals route connectors horizontally and vertically. Text
        # strokes create hundreds of short diagonal Hough segments and must
        # not become a path through the words inside a node.
        if length < 24 or min(dx, dy) > max(5, max(dx, dy) * 0.16):
            continue
        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
        if any(inside_node_text(midpoint, node.box) for node in nodes):
            continue
        segments.append(segment)
    heads = [
        head for head in heads
        if any(
            min(math.hypot(x1 - head[0], y1 - head[1]),
                math.hypot(x2 - head[0], y2 - head[1]))
            <= max(24, head[2] * 2.2)
            for x1, y1, x2, y2 in segments
        )
    ]
    heads = sorted(
        heads,
        key=lambda head: min(_distance_to_box(head[:2], node.box) for node in nodes),
    )[:48]
    branch_labels = [
        item for index, item in enumerate(texts)
        if index not in assigned_text
        and item.text.strip().casefold() in BRANCH_LABELS
    ]
    edges: list[DiagramEdge] = []
    edge_paths: list[tuple[list[tuple[float, float]], float]] = []
    seen: dict[tuple[str, str], int] = {}
    for hx, hy, head_size in heads:
        head = (hx, hy)
        tolerance = max(24, diagonal * 0.016)
        target = min(nodes, key=lambda node: _distance_to_box(head, node.box))
        target_distance = _distance_to_box(head, target.box)
        if target_distance > tolerance:
            continue

        # Follow orthogonal Hough segments backwards from the arrowhead. This
        # handles L-shaped connectors without pretending their elbow touches a
        # nearby node. A path must terminate at another node within a tight
        # boundary tolerance.
        join = max(18, diagonal * 0.011)
        queue: list[tuple[tuple[float, float], frozenset[int], list[tuple[float, float]]]] = []
        for index, (x1, y1, x2, y2) in enumerate(segments):
            for near, far in (((x1, y1), (x2, y2)), ((x2, y2), (x1, y1))):
                if math.hypot(near[0] - hx, near[1] - hy) <= max(join, head_size * 2.2):
                    if _distance_to_box(far, target.box) >= _distance_to_box(near, target.box):
                        queue.append((far, frozenset({index}), [head, far]))

        candidates = []
        cursor = 0
        while cursor < len(queue) and cursor < 200:
            point, used, path_points = queue[cursor]
            cursor += 1
            for source in nodes:
                if source.node_id == target.node_id:
                    continue
                source_distance = _distance_to_box(point, source.box)
                if source_distance <= tolerance:
                    candidates.append((len(used), source_distance, source, point, path_points))
            if len(used) >= 5:
                continue
            for index, (x1, y1, x2, y2) in enumerate(segments):
                if index in used:
                    continue
                for near, far in (((x1, y1), (x2, y2)), ((x2, y2), (x1, y1))):
                    if math.hypot(near[0] - point[0], near[1] - point[1]) <= join:
                        queue.append((far, used | {index}, path_points + [far]))
                        break
        if not candidates:
            continue
        _, source_distance, source, tail, path_points = min(
            candidates, key=lambda row: (row[0], row[1])
        )
        # "FROM SHT" is the origin and "TO SHT" is the destination. OCR text
        # contours near their arrows can otherwise reverse that meaning.
        if INCOMING_PAGE.search(target.text) or OUTGOING_PAGE.search(source.text):
            continue
        if re.match(
            r"\s*(?:\d+\s+)?(?:repair|replace|troubleshoot)\b",
            source.text, re.IGNORECASE,
        ):
            # Corrective-action boxes are leaves in these fault-isolation
            # charts. Pointer glyphs near them otherwise create convincing but
            # reversed edges back into a decision.
            continue

        key = (source.node_id, target.node_id)
        proximity = 1.0 - min(1.0, (source_distance + target_distance) / (2 * tolerance))
        confidence = round(min(source.confidence, target.confidence,
                               0.65 + 0.3 * proximity), 3)
        if confidence >= min_confidence:
            edge = DiagramEdge(source.node_id, target.node_id,
                               "flows_to", "", confidence,
                               (float(tail[0]), float(tail[1])), head)
            existing = seen.get(key)
            if existing is None:
                seen[key] = len(edges)
                edges.append(edge)
                edge_paths.append((path_points, tolerance))
    edges = _assign_branch_labels(edges, edge_paths, nodes, branch_labels)
    if text_items is None:
        node_by_id = {node.node_id: node for node in nodes}
        assigned_targeted_labels: list[TextItem] = []
        for edge_index, (edge, (path, tolerance)) in enumerate(
            zip(edges, edge_paths)
        ):
            if edge.label or node_by_id[edge.source_id].semantic_type != "decision":
                continue
            label = _targeted_branch_label(
                image, path, tolerance, min_confidence
            )
            if label and not any(
                _iou(label.box, existing.box) >= 0.5
                for existing in assigned_targeted_labels
            ):
                assigned_targeted_labels.append(label)
                edges[edge_index] = replace(
                    edge,
                    relationship=_branch_relationship(label.text),
                    label=label.text,
                )
    return nodes, edges


def _normalized(box: tuple[int, int, int, int], size: tuple[int, int]) -> str:
    width, height = size
    values = (box[0] / width, box[1] / height, box[2] / width, box[3] / height)
    return ",".join(f"{value:.4f}" for value in values)


def _normalized_point(point: tuple[float, float], size: tuple[int, int]) -> str:
    return f"{point[0] / size[0]:.4f},{point[1] / size[1]:.4f}"


def _relationship_record(
    record_id: str,
    subject: str,
    subject_kind: str,
    predicate: str,
    object_value: str,
    object_kind: str,
    confidence: float,
    method: str,
) -> str:
    return (
        f"[diagram edge {record_id}]\n"
        f"From: {subject}\nFrom Type: {subject_kind}\n"
        f"Relationship: {predicate}\n"
        f"To: {object_value}\nTo Type: {object_kind}\n"
        f"Confidence: {confidence:.3f}\nMethod: {method}"
    )


def _connector_key(text: str, match: re.Match, anchor: DiagramNode) -> str:
    remainder = f"{text[:match.start()]} {text[match.end():]}"
    remainder = re.sub(r"[^A-Za-z0-9]+", " ", remainder).strip().casefold()
    if remainder:
        return remainder
    anchor_text = OUTGOING_PAGE.sub(" ", anchor.text)
    anchor_text = INCOMING_PAGE.sub(" ", anchor_text)
    return re.sub(r"[^A-Za-z0-9]+", " ", anchor_text).strip().casefold()


def _validate_page_target(
    target: str, page_number: int, page_count: int | None,
    resolved_page: int | None = None, require_resolution: bool = False,
) -> tuple[int | None, bool, str]:
    if not target.isdigit():
        return None, page_count is None, "unverifiable_page_label"
    if require_resolution and resolved_page is None:
        return None, False, "sheet_not_found"
    referenced_page = resolved_page if resolved_page is not None else int(target)
    if page_count is not None and not 1 <= referenced_page <= page_count:
        return referenced_page, False, "page_out_of_range"
    if referenced_page == page_number:
        return referenced_page, False, "same_page_reference"
    return referenced_page, True, ""


def _page_references(
    page_number: int,
    nodes: list[DiagramNode],
    edges: list[DiagramEdge],
    page_count: int | None = None,
    locator_pages: dict[tuple[str, str], int] | None = None,
    page_figure: str | None = None,
) -> list[PageReference]:
    references: list[PageReference] = []
    seen: set[tuple[str, str, str]] = set()

    def add_reference(text: str, anchor: DiagramNode, suffix: str) -> None:
        for direction, match in (
            ("outgoing", OUTGOING_PAGE.search(text)),
            ("incoming", INCOMING_PAGE.search(text)),
        ):
            if match is None:
                continue
            target = match.group("target")
            locator = match.group("locator").lower().rstrip(".")
            is_sheet = locator in ("sheet", "sht")
            resolved_page = None
            if is_sheet and locator_pages is not None and page_figure:
                resolved_page = locator_pages.get((page_figure.casefold(), target))
            referenced_page, valid, reason = _validate_page_target(
                target, page_number, page_count,
                resolved_page=resolved_page,
                require_resolution=is_sheet and locator_pages is not None,
            )
            if is_sheet:
                target_label = f"Figure {page_figure} Sheet {target}" if page_figure else f"Sheet {target}"
                if referenced_page is not None:
                    target_label += f" (PDF page {referenced_page})"
            else:
                target_label = f"Page {target}"
            key = (anchor.text, direction, target_label)
            if key not in seen:
                seen.add(key)
                references.append(PageReference(
                    reference_id=f"X{page_number}-{suffix}-{direction[:3].upper()}",
                    page_number=page_number,
                    anchor=anchor,
                    direction=direction,
                    referenced_page=referenced_page,
                    target_label=target_label,
                    connector_key=_connector_key(text, match, anchor),
                    raw_text=text,
                    valid=valid,
                    reason=reason,
                ))

    by_id = {node.node_id: node for node in nodes}
    for index, node in enumerate(nodes, start=1):
        add_reference(node.text, node, f"N{index}")
    for index, edge in enumerate(edges, start=1):
        if edge.label:
            # An explicit "to page 5" printed beside the arrow belongs to the
            # node the arrow leaves, even when the connector itself is blank.
            add_reference(edge.label, by_id[edge.source_id], f"E{index}")
    return references


def _reciprocal_reference_ids(references: list[PageReference]) -> set[str]:
    outgoing: dict[tuple[int, int], list[PageReference]] = {}
    incoming: dict[tuple[int, int], list[PageReference]] = {}
    for reference in references:
        if not reference.valid or reference.referenced_page is None:
            continue
        if reference.direction == "outgoing":
            pair = (reference.page_number, reference.referenced_page)
            outgoing.setdefault(pair, []).append(reference)
        else:
            pair = (reference.referenced_page, reference.page_number)
            incoming.setdefault(pair, []).append(reference)

    reciprocal: set[str] = set()
    for pair, departures in outgoing.items():
        arrivals = incoming.get(pair, [])
        for departure in departures:
            matches = [
                arrival for arrival in arrivals
                if departure.connector_key
                and departure.connector_key == arrival.connector_key
            ]
            same_departures = sum(
                candidate.connector_key == departure.connector_key
                for candidate in departures
            )
            if len(matches) != 1 or same_departures != 1:
                matches = []
            if not matches and len(departures) == 1 and len(arrivals) == 1:
                arrival = arrivals[0]
                if not (
                    departure.connector_key
                    and arrival.connector_key
                    and departure.connector_key != arrival.connector_key
                ):
                    matches = [arrival]
            if matches:
                reciprocal.add(departure.reference_id)
                reciprocal.update(arrival.reference_id for arrival in matches)
    return reciprocal


def _page_reference_records(
    page_number: int,
    nodes: list[DiagramNode],
    references: list[PageReference],
    reciprocal_ids: set[str],
) -> list[str]:
    """Facts that make a non-clickable continuation traversable across pages."""
    records: list[str] = []
    page_name = f"Page {page_number}"
    for index, node in enumerate(nodes, start=1):
        records.append(_relationship_record(
            f"P{page_number}-N{index}", page_name, "page_reference",
            "contains_diagram_node", node.text, node.semantic_type,
            node.confidence, "diagram_layout",
        ))

    for reference in references:
        if not reference.valid:
            continue
        reciprocal = reference.reference_id in reciprocal_ids
        confidence = _reference_confidence(reference, reciprocal)
        method = (
            "reciprocal_page_reference" if reciprocal
            else "explicit_page_reference"
        )
        if reference.direction == "outgoing":
            records.append(_relationship_record(
                reference.reference_id,
                reference.anchor.text,
                reference.anchor.semantic_type,
                "continues_on_page",
                reference.target_label,
                "page_reference",
                confidence,
                method,
            ))
        else:
            records.append(_relationship_record(
                reference.reference_id,
                reference.target_label,
                "page_reference",
                "continues_at",
                reference.anchor.text,
                reference.anchor.semantic_type,
                confidence,
                method,
            ))
    return records


def _reference_confidence(reference: PageReference, reciprocal: bool) -> float:
    if reciprocal:
        return round(min(0.99, max(0.90, reference.anchor.confidence + 0.08)), 3)
    return round(min(0.85, reference.anchor.confidence), 3)


def _reference_review_record(
    reference: PageReference, reciprocal_ids: set[str]
) -> dict:
    reciprocal = reference.reference_id in reciprocal_ids
    return {
        "id": reference.reference_id,
        "direction": reference.direction,
        "raw_text": reference.raw_text,
        "connector": reference.connector_key,
        "referenced_page": reference.referenced_page,
        "target": reference.target_label,
        "valid": reference.valid,
        "reason": reference.reason or None,
        "reciprocal": reciprocal,
        "confidence": (
            _reference_confidence(reference, reciprocal) if reference.valid else 0.0
        ),
    }


def serialize_page(page_number: int, size: tuple[int, int],
                   nodes: list[DiagramNode], edges: list[DiagramEdge],
                   page_count: int | None = None,
                   reciprocal_ids: set[str] | None = None,
                   locator_pages: dict[tuple[str, str], int] | None = None,
                   page_figure: str | None = None) -> str:
    if not edges:
        return ""
    by_id = {node.node_id: node for node in nodes}
    parts = [f"[diagram page {page_number}]"]
    for node in nodes:
        parts.append(
            f"[diagram node {node.node_id}]\n"
            f"Text: {node.text}\nShape: {node.shape}\n"
            f"Semantic Type: {node.semantic_type}\n"
            f"Bounding Box: {_normalized(node.box, size)}\n"
            f"Confidence: {node.confidence:.3f}"
        )
    for index, edge in enumerate(edges, start=1):
        source, target = by_id[edge.source_id], by_id[edge.target_id]
        parts.append(
            f"[diagram edge E{index}]\n"
            f"From: {source.text}\nFrom Type: {source.semantic_type}\n"
            f"Relationship: {edge.relationship}\n"
            f"To: {target.text}\nTo Type: {target.semantic_type}\n"
            f"Arrow Label: {edge.label}\nConfidence: {edge.confidence:.3f}\n"
            f"Arrow Tail: {_normalized_point(edge.tail, size)}\n"
            f"Arrow Head: {_normalized_point(edge.head, size)}\n"
            "Method: diagram_geometry"
        )
    references = _page_references(
        page_number, nodes, edges, page_count, locator_pages, page_figure
    )
    parts.extend(_page_reference_records(
        page_number, nodes, references, reciprocal_ids or set()
    ))
    return "\n\n".join(parts)


def _serialize_document_pages(
    pages: dict[int, AnalyzedPage],
    page_count: int,
    locator_pages: dict[tuple[str, str], int] | None = None,
    page_figures: dict[int, str] | None = None,
) -> tuple[dict[int, str], list[PageReference], set[str]]:
    page_figures = page_figures or {}
    references = [
        reference
        for page_number, (_size, nodes, edges) in pages.items()
        for reference in _page_references(
            page_number, nodes, edges, page_count, locator_pages,
            page_figures.get(page_number),
        )
    ]
    accepted_pages = {
        page_number for page_number, (_size, _nodes, edges) in pages.items() if edges
    }
    reciprocal_ids = _reciprocal_reference_ids([
        reference
        for reference in references
        if reference.page_number in accepted_pages
    ])
    found: dict[int, str] = {}
    for page_number, (size, nodes, edges) in pages.items():
        text = serialize_page(
            page_number,
            size,
            nodes,
            edges,
            page_count=page_count,
            reciprocal_ids=reciprocal_ids,
            locator_pages=locator_pages,
            page_figure=page_figures.get(page_number),
        )
        if text:
            found[page_number] = text
    return found, references, reciprocal_ids


def _write_review_report(
    path: str,
    page_count: int,
    examined_pages: int,
    pages: dict[int, AnalyzedPage],
    page_notes: dict[int, str],
    references: list[PageReference],
    reciprocal_ids: set[str],
    output_dir: str,
) -> None:
    if not output_dir:
        return
    references_by_page: dict[int, list[PageReference]] = {}
    for reference in references:
        references_by_page.setdefault(reference.page_number, []).append(reference)

    page_records = []
    for page_number in sorted(set(pages) | set(page_notes)):
        size, nodes, edges = pages.get(page_number, ((0, 0), [], []))
        page_references = references_by_page.get(page_number, [])
        status = page_notes.get(
            page_number, "accepted" if edges else "needs_review"
        )
        page_records.append({
            "page": page_number,
            "status": status,
            "render_size": list(size),
            "nodes": [
                {
                    "id": node.node_id,
                    "text": node.text,
                    "shape": node.shape,
                    "semantic_type": node.semantic_type,
                    "confidence": node.confidence,
                    "bounding_box": list(node.box),
                }
                for node in nodes
            ],
            "edges": [
                {
                    "from": edge.source_id,
                    "to": edge.target_id,
                    "relationship": edge.relationship,
                    "label": edge.label,
                    "confidence": edge.confidence,
                }
                for edge in edges
            ],
            "page_references": [
                _reference_review_record(reference, reciprocal_ids)
                for reference in page_references
            ],
        })

    report = {
        "source": os.path.abspath(path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "page_count": page_count,
        "summary": {
            "pages_examined": examined_pages,
            "candidate_pages": len(pages),
            "accepted_pages": sum(bool(edges) for _size, _nodes, edges in pages.values()),
            "references": len(references),
            "invalid_references": sum(not reference.valid for reference in references),
            "reciprocal_references": len(reciprocal_ids),
        },
        "pages": page_records,
    }
    safe_stem = re.sub(
        r"[^A-Za-z0-9_.-]+", "-", os.path.splitext(os.path.basename(path))[0]
    ).strip("-.")[:80] or "document"
    digest = hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
    destination = os.path.join(output_dir, f"{safe_stem}-{digest}.json")
    temporary = destination + ".tmp"
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary, destination)
    except OSError as exc:
        log.warning("could not write diagram review report for %s: %s", path, exc)
        try:
            os.unlink(temporary)
        except OSError:
            pass


def pdf_diagram_text(path: str, cfg) -> dict[int, str]:
    """Return structured diagram text keyed by 1-based PDF page number."""
    ok, reason = available()
    if not ok:
        log.warning("diagram analysis unavailable: %s", reason)
        return {}
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(path)
    except Exception as exc:
        log.warning("could not open %s for diagram analysis: %s", path, exc)
        return {}
    analyzed: dict[int, AnalyzedPage] = {}
    page_notes: dict[int, str] = {}
    found: dict[int, str] = {}
    locator_pages: dict[tuple[str, str], int] = {}
    page_figures: dict[int, str] = {}
    try:
        page_count = len(document)
        limit = min(page_count, cfg.diagram_max_pages)
        for page_index in range(limit):
            page = document[page_index]
            try:
                try:
                    page_text = page.get_textpage().get_text_range()
                except Exception:
                    page_text = ""
                heading = FIGURE_SHEET_HEADING.search(page_text)
                if heading:
                    figure, sheet = heading.groups()
                    page_figures[page_index + 1] = figure
                    locator_pages[(figure.casefold(), sheet)] = page_index + 1
                width, height = page.get_size()
                if width * height * cfg.diagram_scale * cfg.diagram_scale > (
                    cfg.ocr_max_megapixels * 1_000_000
                ):
                    page_notes[page_index + 1] = "skipped_render_too_large"
                    continue
                # Cheap first pass prevents OCR on ordinary prose and tables.
                preview = page.render(scale=1.0).to_pil().convert("RGB")
                if len(_shapes(preview)) < 2 or not _arrowheads(preview):
                    continue
                image = page.render(scale=cfg.diagram_scale).to_pil().convert("RGB")
                nodes, edges = analyze_image(
                    image, min_confidence=cfg.diagram_min_confidence
                )
                analyzed[page_index + 1] = (image.size, nodes, edges)
                if not edges:
                    page_notes[page_index + 1] = "needs_review_no_directed_edges"
                else:
                    node_by_id = {node.node_id: node for node in nodes}
                    if any(
                        node_by_id[edge.source_id].semantic_type == "decision"
                        and not edge.label
                        for edge in edges
                    ):
                        page_notes[page_index + 1] = (
                            "needs_review_unlabeled_decision_branch"
                        )
            except Exception as exc:
                page_notes[page_index + 1] = f"analysis_error: {type(exc).__name__}"
                log.debug("diagram analysis failed for %s page %d: %s",
                          path, page_index + 1, exc)
        found, references, reciprocal_ids = _serialize_document_pages(
            analyzed, page_count, locator_pages, page_figures
        )
        if cfg.diagram_review:
            _write_review_report(
                path,
                page_count,
                limit,
                analyzed,
                page_notes,
                references,
                reciprocal_ids,
                cfg.diagram_review_dir,
            )
    finally:
        try:
            document.close()
        except Exception:
            pass
    return found
