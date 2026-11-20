# SPDX-License-Identifier: MPL-2.0
"""Cameo / MagicDraw model exports (XMI, MDXML) and BPMN, read as records.

A model export is not a document: its meaning lives in which element owns
which, which block satisfies which requirement, which flow leads where.  Read
as ordinary XML that meaning disappears - ``itertext()`` returns a soup of
names with no edges, and every ``xmi:idref`` stays an opaque hex string.

So this resolves the id graph first and then writes the same blank-line
delimited ``Key: value`` records that spreadsheets and flowcharts already
produce, which app.relationships already turns into typed, evidence-backed
facts.  Nothing is inferred: every edge here is a reference the file states.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

log = logging.getLogger("rag.sysml")

WHITESPACE = re.compile(r"\s+")

# SysML puts requirement traceability on stereotyped dependencies, so the
# stereotype - not the UML metaclass - is what names the fact.
STEREOTYPE_PREDICATES = {
    "satisfy": "satisfies",
    "verify": "verifies",
    "derive": "derives_from",
    "derivereqt": "derives_from",
    "refine": "refines",
    "trace": "traces_to",
    "tracereqt": "traces_to",
    "allocate": "allocated_to",
    "copy": "copies",
}

METACLASS_PREDICATES = {
    "generalization": "specializes",
    "dependency": "depends_on",
    "usage": "depends_on",
    "abstraction": "depends_on",
    "realization": "realizes",
    "interfacerealization": "realizes",
    "association": "associated_with",
    "informationflow": "flows_to",
    "itemflow": "flows_to",
    "sequenceflow": "flows_to",
    "messageflow": "sends_message_to",
}

# Every predicate this module can emit. app.relationships accepts only these.
MODEL_PREDICATES = frozenset(
    set(STEREOTYPE_PREDICATES.values())
    | set(METACLASS_PREDICATES.values())
    | {"contains", "has_part", "typed_by"}
)

# Cross-element references, as attributes or as child elements.
REF_NAMES = (
    "client", "supplier", "general", "specific", "memberEnd", "type",
    "informationSource", "informationTarget", "realizingConnector",
    "sourceRef", "targetRef", "flowNodeRef", "processRef",
)

# Element kinds that earn a record of their own even when unnamed.
ALWAYS_RECORD = frozenset({"requirement", "testcase", "constraint"})

# SysML writes value properties as composite too, so composition alone does not
# mean "is made of": a block composing a Real has a parameter, not a part.
VALUE_TYPES = frozenset({
    "datatype", "primitivetype", "enumeration", "valuetype", "unit",
    "quantitykind", "instancespecification",
})


def _local(tag: str) -> str:
    """Local name of a namespaced tag or attribute."""
    return tag.rsplit("}", 1)[-1]


def _attr(elem: ET.Element, name: str) -> str | None:
    """Attribute by local name, so XMI 2.1 and 2.5 namespaces both resolve."""
    value = elem.get(name)
    if value is not None:
        return value
    for key, val in elem.attrib.items():
        if _local(key) == name:
            return val
    return None


def _meta(elem: ET.Element) -> str:
    """The metaclass: xmi:type / xsi:type if present, else the tag itself.

    Only a *namespaced* type attribute counts. A bare ``type="_id17"`` on
    ``<ownedAttribute>`` is a reference to the property's type, not its
    metaclass, and reading it as one turns every typed property into garbage.
    """
    for key, value in elem.attrib.items():
        if "}" in key and _local(key) == "type":
            return value.rsplit(":", 1)[-1]
    return _local(elem.tag)


def _refs(elem: ET.Element, name: str) -> list[str]:
    """Ids referenced by attribute ``name`` or by child elements of that name."""
    found: list[str] = []
    attribute = elem.get(name)
    if attribute:
        found.extend(attribute.split())
    for child in elem:
        if _local(child.tag) != name:
            continue
        idref = _attr(child, "idref")
        href = child.get("href")
        if idref:
            found.append(idref)
        elif href:
            found.append(href.rsplit("#", 1)[-1])
    return found


def _flat(text: str | None) -> str:
    """One line, so a value can never break out of its record."""
    return WHITESPACE.sub(" ", text).strip() if text else ""


def _literal(elem: ET.Element, name: str) -> str:
    """A UML ValueSpecification child read as text (default, lower, upper)."""
    for child in elem:
        if _local(child.tag) != name:
            continue
        value = child.get("value")
        if value:
            return _flat(value)
        for grandchild in child:
            if _local(grandchild.tag) == "body" and grandchild.text:
                return _flat(grandchild.text)
        if _meta(child).lower() == "literalnull":
            return ""
        return _flat(child.get("symbol") or "")
    return ""


def _documentation(elem: ET.Element) -> str:
    for child in elem:
        if _local(child.tag) != "ownedComment":
            continue
        body = child.get("body")
        if body:
            return _flat(body)
        for grandchild in child:
            if _local(grandchild.tag) == "body" and grandchild.text:
                return _flat(grandchild.text)
    return ""


@dataclass
class Node:
    node_id: str
    name: str
    meta: str
    qname: str
    owner_id: str
    documentation: str = ""
    aggregation: str = ""
    multiplicity: str = ""
    default: str = ""
    stereotypes: list[str] = field(default_factory=list)
    stereotype_attrs: dict[str, str] = field(default_factory=dict)
    refs: dict[str, list[str]] = field(default_factory=dict)


def _stereotypes(root: ET.Element) -> tuple[dict[str, list[str]], dict[str, dict[str, str]]]:
    """Stereotype applications, keyed by the id of the element they extend.

    A stereotype application is a sibling element carrying ``base_Class`` (or
    ``base_Dependency``, ``base_Property``, ...) rather than a child of its
    target, and its namespace prefix differs per profile and per Cameo
    version. Matching on the ``base_`` attribute instead of the tag keeps this
    working across profiles nobody here has seen.
    """
    applied: dict[str, list[str]] = {}
    attrs: dict[str, dict[str, str]] = {}
    for elem in root.iter():
        base = next(
            (v for k, v in elem.attrib.items() if _local(k).startswith("base_")), None
        )
        if not base:
            continue
        applied.setdefault(base, []).append(_local(elem.tag).rsplit(":", 1)[-1])
        carried = {
            _local(k): _flat(v)
            for k, v in elem.attrib.items()
            if not _local(k).startswith("base_")
            and _local(k) not in ("id", "type", "uuid", "idref", "version")
        }
        if carried:
            attrs.setdefault(base, {}).update(carried)
    return applied, attrs


def _collect(root: ET.Element) -> list[Node]:
    applied, stereotype_attrs = _stereotypes(root)
    nodes: list[Node] = []

    def visit(elem: ET.Element, parent_qname: str, parent_id: str) -> None:
        node_id = _attr(elem, "id") or ""
        name = _flat(elem.get("name"))
        meta = _meta(elem)
        qname = f"{parent_qname}::{name}" if parent_qname and name else (name or parent_qname)
        stereotypes = applied.get(node_id, [])
        interesting = bool(node_id) and (
            bool(name) or bool(stereotypes) or meta.lower() in ALWAYS_RECORD
            or meta.lower() in METACLASS_PREDICATES
        )
        if interesting:
            refs = {n: ids for n in REF_NAMES if (ids := _refs(elem, n))}
            nodes.append(Node(
                node_id=node_id,
                name=name,
                meta=meta,
                qname=qname or node_id,
                owner_id=parent_id,
                documentation=_documentation(elem),
                aggregation=_flat(elem.get("aggregation")),
                multiplicity=_multiplicity(elem),
                default=_literal(elem, "defaultValue"),
                stereotypes=stereotypes,
                stereotype_attrs=stereotype_attrs.get(node_id, {}),
                refs=refs,
            ))
        for child in elem:
            # A stereotype application is not part of the model tree; it was
            # already folded into the element it extends.
            if any(_local(k).startswith("base_") for k in child.attrib):
                continue
            visit(child, qname, node_id if interesting else parent_id)

    visit(root, "", "")
    return nodes


def _multiplicity(elem: ET.Element) -> str:
    lower, upper = _literal(elem, "lowerValue"), _literal(elem, "upperValue")
    if not lower and not upper:
        return ""
    return f"{lower or '0'}..{upper or '1'}"


def _kind(node: Node) -> str:
    """The kind a reader would recognise: the stereotype if there is one."""
    return node.stereotypes[0] if node.stereotypes else node.meta


def _predicate(node: Node) -> str | None:
    for stereotype in node.stereotypes:
        predicate = STEREOTYPE_PREDICATES.get(stereotype.casefold())
        if predicate:
            return predicate
    return METACLASS_PREDICATES.get(node.meta.casefold())


def _edge(index: int, subject: Node, predicate: str, obj: Node, method: str) -> str:
    return (
        f"[model edge M{index}]\n"
        f"From: {subject.qname}\nFrom Type: {_kind(subject)}\n"
        f"Relationship: {predicate}\n"
        f"To: {obj.qname}\nTo Type: {_kind(obj)}\n"
        f"Confidence: 1.000\nMethod: {method}"
    )


def _element_record(node: Node) -> str:
    lines = [f"[model element {node.node_id}]", f"Name: {node.name or node.node_id}"]
    lines.append(f"Type: {_kind(node)}")
    if node.stereotypes:
        lines.append(f"Stereotypes: {', '.join(node.stereotypes)}")
    if node.name and node.qname != node.name:
        lines.append(f"Qualified Name: {node.qname}")
    if node.meta.casefold() == "property":
        # A SysML value property is the model's word for a parameter, and the
        # header matches the one spreadsheet exports of the same data use.
        lines.append(f"Parameter Name: {node.name or node.node_id}")
    if node.multiplicity:
        lines.append(f"Multiplicity: {node.multiplicity}")
    if node.default:
        lines.append(f"Default: {node.default}")
    if node.aggregation:
        lines.append(f"Aggregation: {node.aggregation}")
    for key, value in node.stereotype_attrs.items():
        if value:
            lines.append(f"{key}: {value}")
    if node.documentation:
        lines.append(f"Documentation: {node.documentation}")
    return "\n".join(lines)


def _edges(nodes: list[Node], by_id: dict[str, Node]) -> list[str]:
    records: list[str] = []
    counter = 0

    def add(subject: Node | None, predicate: str, obj_id: str, method: str) -> None:
        nonlocal counter
        obj = by_id.get(obj_id)
        if subject is None or obj is None or subject.node_id == obj.node_id:
            return
        counter += 1
        records.append(_edge(counter, subject, predicate, obj, method))

    for node in nodes:
        owner = by_id.get(node.owner_id)
        predicate = _predicate(node)

        if owner is not None and node.name:
            counter += 1
            records.append(_edge(counter, owner, "contains", node, "xmi_containment"))

        if node.meta.casefold() == "property":
            for type_id in node.refs.get("type", []):
                add(node, "typed_by", type_id, "xmi_reference")
                target = by_id.get(type_id)
                if (
                    node.aggregation.casefold() == "composite"
                    and target is not None
                    and _kind(target).casefold() not in VALUE_TYPES
                ):
                    add(owner, "has_part", type_id, "xmi_reference")

        if predicate is None:
            continue

        if node.meta.casefold() == "generalization":
            # Owned by the specific classifier and pointing at the general one.
            for general in node.refs.get("general", []):
                add(owner, predicate, general, "xmi_reference")
            continue

        if node.meta.casefold() == "association":
            ends = [
                type_id
                for end in node.refs.get("memberEnd", [])
                if (member := by_id.get(end))
                for type_id in member.refs.get("type", [])
            ]
            for source, target in zip(ends, ends[1:]):
                add(by_id.get(source), predicate, target, "xmi_reference")
            continue

        sources = (
            node.refs.get("client")
            or node.refs.get("informationSource")
            or node.refs.get("sourceRef")
            or ([owner.node_id] if owner else [])
        )
        targets = (
            node.refs.get("supplier")
            or node.refs.get("informationTarget")
            or node.refs.get("targetRef")
            or []
        )
        method = "bpmn_reference" if "sourceRef" in node.refs else "xmi_reference"
        for source in sources:
            for target in targets:
                add(by_id.get(source), predicate, target, method)

    for node in nodes:
        # BPMN lanes and pools own their contents by reference, not by nesting.
        for name, predicate in (("flowNodeRef", "contains"), ("processRef", "contains")):
            for target in node.refs.get(name, []):
                add(node, predicate, target, "bpmn_reference")

    return records


def is_model(path: str) -> bool:
    """True for an XMI/MDXML or BPMN file, cheaply, without a full parse."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(8192).decode("utf-8", "replace")
    except OSError:
        return False
    return any(
        marker in head
        for marker in ("XMI/2.", "spec/XMI", "xmlns:uml", "xmlns:sysml", "spec/BPMN")
    )


def from_model(path: str) -> str | None:
    """Render a Cameo/XMI or BPMN export as relationship-bearing records."""
    try:
        root = ET.parse(path).getroot()  # ponytail: whole tree in memory;
        # RAG_MAX_FILE_BYTES caps the input, switch to iterparse if that rises.
    except (OSError, ET.ParseError) as exc:
        log.warning("could not parse %s: %s", path, exc)
        return None

    nodes = _collect(root)
    if not nodes:
        return None
    by_id = {node.node_id: node for node in nodes}

    dialect = "BPMN" if "BPMN" in root.tag else "XMI"
    model = next((n for n in nodes if n.meta.casefold() in ("model", "definitions")), None)
    parts = [f"[model {model.name if model and model.name else path}]\nSource Format: {dialect}"]
    # An unnamed Satisfy or Generalization is pure wiring: it says everything
    # it has to say as an edge, and a record for it would only be an id.
    parts.extend(
        _element_record(node) for node in nodes
        if node.name or node.meta.casefold() in ALWAYS_RECORD
    )
    parts.extend(_edges(nodes, by_id))
    return "\n\n".join(parts)
