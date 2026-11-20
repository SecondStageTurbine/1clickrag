# SPDX-License-Identifier: MPL-2.0
"""Conservative typed relationships extracted from structured records.

Spreadsheet extraction emits one blank-line-delimited record per row as
``Header: value`` pairs.  Only values explicitly present in the same record are
connected here; this module never guesses a relationship from prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .entities import normalise_key
from .sysml import MODEL_PREDICATES

FIELD = re.compile(r"^([^:\n]{1,100}):\s*(.+)$")
ROW = re.compile(r"^\[row\s+\d+\]$", re.IGNORECASE)
EDGE_RECORD = re.compile(r"^\[(?:diagram|model) edge\s+[^]]+\]$", re.IGNORECASE)


@dataclass(frozen=True)
class Relationship:
    subject: str
    subject_kind: str
    predicate: str
    object: str
    object_kind: str
    confidence: float = 1.0
    inference: str = "explicit"


ALIASES = {
    "fault": {
        "fault", "fault condition", "failure", "failure condition",
        "failure mode", "symptom", "observed symptom",
    },
    "parameter": {
        "parameter", "parameter name", "signal", "signal name",
        "data parameter", "data item",
    },
    "arinc_label": {
        "arinc label", "arinc 429 label", "label", "label number",
    },
    "artifact": {
        "artifact", "artifact name", "document", "document name",
        "report", "report name", "test report", "source artifact",
    },
}

HEADER_KIND = {
    normalise_key(alias): kind for kind, aliases in ALIASES.items() for alias in aliases
}

RELATIONSHIPS = (
    ("fault", "maps_to_parameter", "parameter"),
    ("fault", "maps_to_arinc_label", "arinc_label"),
    ("parameter", "uses_arinc_label", "arinc_label"),
    ("artifact", "contains_parameter", "parameter"),
    ("artifact", "documents_fault", "fault"),
    ("artifact", "contains_arinc_label", "arinc_label"),
)

DIAGRAM_PREDICATES = frozenset({
    "flows_to", "yes_branch_to", "no_branch_to",
    "contains_diagram_node", "continues_on_page", "continues_at",
})

# Facts read straight out of a Cameo/XMI or BPMN export. Same record shape as
# a diagram edge - stated by the file rather than derived from geometry.
EDGE_PREDICATES = DIAGRAM_PREDICATES | MODEL_PREDICATES


def _records(text: str) -> list[dict[str, list[str]]]:
    records: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        if ROW.match(line) or line.lower().startswith("[sheet "):
            continue
        match = FIELD.match(line)
        if not match:
            continue
        kind = HEADER_KIND.get(normalise_key(match.group(1)))
        value = match.group(2).strip()
        if kind and value:
            current.setdefault(kind, []).append(value)
    if current:
        records.append(current)
    return records


def extract_relationships(text: str) -> list[Relationship]:
    """Return only schema-approved facts explicitly stated in one record."""
    found: dict[tuple[str, str, str], Relationship] = {}
    for record in _records(text):
        for subject_kind, predicate, object_kind in RELATIONSHIPS:
            for subject in record.get(subject_kind, []):
                for object_value in record.get(object_kind, []):
                    if normalise_key(subject) == normalise_key(object_value):
                        continue
                    relation = Relationship(
                        subject, subject_kind, predicate, object_value, object_kind
                    )
                    found[(normalise_key(subject), predicate,
                           normalise_key(object_value))] = relation

    # Edge records are not statements made in prose: a diagram edge is derived
    # from geometry, a model edge is a resolved xmi:idref. They are accepted
    # only from the structured records emitted by app.diagram and app.sysml,
    # and each keeps its own method and confidence, distinct from the explicit
    # spreadsheet facts above.
    current: dict[str, str] | None = None
    diagram_records: list[dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if EDGE_RECORD.match(line):
            if current:
                diagram_records.append(current)
            current = {}
            continue
        if line.startswith("["):
            if current:
                diagram_records.append(current)
            current = None
            continue
        if current is None:
            continue
        match = FIELD.match(line)
        if match:
            current[normalise_key(match.group(1))] = match.group(2).strip()
    if current:
        diagram_records.append(current)

    for record in diagram_records:
        subject = record.get("from", "")
        object_value = record.get("to", "")
        predicate = record.get("relationship", "")
        if not subject or not object_value or predicate not in EDGE_PREDICATES:
            continue
        try:
            confidence = max(0.0, min(1.0, float(record.get("confidence", "0"))))
        except ValueError:
            continue
        relation = Relationship(
            subject=subject,
            subject_kind=record.get("from type", "diagram_node"),
            predicate=predicate,
            object=object_value,
            object_kind=record.get("to type", "diagram_node"),
            confidence=confidence,
            inference=record.get("method", "diagram_geometry"),
        )
        found[(normalise_key(subject), predicate,
               normalise_key(object_value))] = relation
    return list(found.values())
