from __future__ import annotations

import json
import os
import tempfile
import unittest

from app.chunker import Chunk
from app.diagram import (
    DiagramEdge,
    DiagramNode,
    FIGURE_SHEET_HEADING,
    TextItem,
    _assign_branch_labels,
    _page_references,
    _serialize_document_pages,
    _write_review_report,
    analyze_image,
    serialize_page,
)
from app.graph import Graph
from app.relationships import extract_relationships


class DiagramGeometryTests(unittest.TestCase):
    def test_branch_labels_are_exclusive_and_follow_decision_topology(self) -> None:
        nodes = [
            DiagramNode("D", "Is voltage valid?", "hexagon", "decision",
                        0.9, (50, 50, 150, 150)),
            DiagramNode("R", "Repair", "rectangle", "process",
                        0.9, (280, 50, 380, 150)),
            DiagramNode("C", "Continue", "rectangle", "process",
                        0.9, (50, 280, 150, 380)),
        ]
        edges = [
            DiagramEdge("D", "R", "flows_to", "", 0.9,
                        (150, 100), (280, 100)),
            DiagramEdge("D", "C", "flows_to", "", 0.9,
                        (100, 150), (100, 280)),
        ]
        paths = [
            ([(280.0, 100.0), (100.0, 100.0)], 50.0),
            ([(100.0, 280.0), (100.0, 100.0)], 50.0),
        ]
        labels = [
            TextItem("NO", 0.99, (175, 85, 205, 105)),
            TextItem("YES", 0.99, (85, 175, 115, 195)),
        ]

        assigned = _assign_branch_labels(edges, paths, nodes, labels)

        self.assertEqual(assigned[0].relationship, "no_branch_to")
        self.assertEqual(assigned[1].relationship, "yes_branch_to")
        self.assertEqual({edge.label for edge in assigned}, {"YES", "NO"})

    def test_figure_sheet_heading_accepts_total_sheet_count(self) -> None:
        match = FIGURE_SHEET_HEADING.search(
            "Figure 2-11. STARTING SYSTEM FAULT ISOLATION (Sheet 1 of 82)"
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.groups(), ("2-11", "1"))

    def test_page_text_beside_off_page_connector_is_attached(self) -> None:
        try:
            from PIL import Image, ImageDraw
            import cv2  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))

        image = Image.new("RGB", (900, 400), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((50, 130, 250, 270), outline="black", width=5)
        draw.polygon(
            [(540, 125), (700, 125), (700, 230), (620, 285), (540, 230)],
            outline="black",
        )
        draw.line((250, 200, 510, 200), fill="black", width=5)
        draw.polygon([(514, 188), (514, 212), (540, 200)], fill="black")

        nodes, edges = analyze_image(image, text_items=[
            TextItem("Check valve", 0.99, (90, 180, 210, 210)),
            TextItem("A", 0.99, (615, 175, 630, 200)),
            TextItem("TO PAGE 5", 0.99, (710, 175, 810, 205)),
        ])

        connector = next(
            node for node in nodes if node.semantic_type == "off_page_connector"
        )
        self.assertEqual(connector.text, "A TO PAGE 5")
        self.assertTrue(edges)
        relationships = extract_relationships(
            serialize_page(3, image.size, nodes, edges)
        )
        self.assertTrue(any(
            item.subject == "A TO PAGE 5"
            and item.predicate == "continues_on_page"
            and item.object == "Page 5"
            for item in relationships
        ))

    def test_rectangle_arrow_diamond_becomes_directional_relationship(self) -> None:
        try:
            from PIL import Image, ImageDraw
            import cv2  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))

        image = Image.new("RGB", (900, 400), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((50, 130, 250, 270), outline="black", width=5)
        draw.polygon(
            [(500, 200), (680, 80), (860, 200), (680, 320)],
            outline="black",
        )
        draw.line((250, 200, 470, 200), fill="black", width=5)
        draw.polygon([(476, 188), (476, 212), (502, 200)], fill="black")

        text = [
            TextItem("Pressure below limit", 0.99, (70, 180, 230, 210)),
            TextItem("Set fault", 0.99, (620, 180, 740, 210)),
            TextItem("Yes", 0.99, (350, 150, 390, 170)),
        ]
        nodes, edges = analyze_image(image, text_items=text)

        by_text = {node.text: node for node in nodes}
        self.assertEqual(by_text["Pressure below limit"].shape, "rectangle")
        self.assertEqual(by_text["Pressure below limit"].semantic_type, "process")
        self.assertEqual(by_text["Set fault"].shape, "diamond")
        self.assertEqual(by_text["Set fault"].semantic_type, "decision")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].relationship, "flows_to")

        structured = serialize_page(7, image.size, nodes, edges)
        relationships = extract_relationships(structured)
        relation = next(item for item in relationships if item.predicate == "flows_to")
        self.assertEqual(relation.subject, "Pressure below limit")
        self.assertEqual(relation.predicate, "flows_to")
        self.assertEqual(relation.object, "Set fault")
        self.assertEqual(relation.inference, "diagram_geometry")
        self.assertGreaterEqual(relation.confidence, 0.55)

        with tempfile.TemporaryDirectory() as directory:
            graph = Graph(os.path.join(directory, "graph.db"), max_df=1.0)
            try:
                graph.replace_path(
                    "phm-flow.pdf",
                    [Chunk("phm-flow.pdf", "pdf", 1, 20, structured,
                           section="page 7", source="phm-flow.pdf")],
                )
                hits, facts = graph.relationship_search(
                    "Pressure below limit", top_k=5
                )
                self.assertTrue(hits)
                self.assertTrue(any(
                    fact["predicate"] == "flows_to"
                    and fact["object"] == "Set fault"
                    and fact["citation"].startswith("phm-flow.pdf:")
                    for fact in facts
                ))
            finally:
                graph.close()

    def test_explicit_page_connectors_retrieve_the_target_page(self) -> None:
        page3_nodes = [
            DiagramNode("N1", "Check valve", "rectangle", "process", 0.94,
                        (20, 120, 220, 220)),
            DiagramNode("N2", "A TO PAGE 5", "off_page_connector",
                        "off_page_connector", 0.88, (400, 130, 520, 210)),
        ]
        page3_edges = [
            DiagramEdge("N1", "N2", "flows_to", "", 0.86,
                        (220, 170), (400, 170)),
        ]
        page5_nodes = [
            DiagramNode("N1", "A FROM PAGE 3", "off_page_connector",
                        "off_page_connector", 0.87, (20, 130, 140, 210)),
            DiagramNode("N2", "Command shutdown", "rectangle", "process", 0.93,
                        (320, 120, 560, 220)),
        ]
        page5_edges = [
            DiagramEdge("N1", "N2", "flows_to", "", 0.85,
                        (140, 170), (320, 170)),
        ]

        serialized, references, reciprocal_ids = _serialize_document_pages(
            {
                3: ((600, 400), page3_nodes, page3_edges),
                5: ((600, 400), page5_nodes, page5_edges),
            },
            page_count=6,
        )
        page3, page5 = serialized[3], serialized[5]
        relationships = extract_relationships(page3 + "\n\n" + page5)
        triples = {(item.subject, item.predicate, item.object) for item in relationships}
        self.assertIn(("A TO PAGE 5", "continues_on_page", "Page 5"), triples)
        self.assertIn(("Page 5", "contains_diagram_node", "Command shutdown"), triples)
        self.assertIn(("Page 3", "continues_at", "A FROM PAGE 3"), triples)
        cross_page = [
            item for item in relationships
            if item.predicate in {"continues_on_page", "continues_at"}
        ]
        self.assertTrue(reciprocal_ids)
        self.assertEqual(len(reciprocal_ids), 2)
        self.assertTrue(all(
            item.inference == "reciprocal_page_reference"
            and item.confidence >= 0.90
            for item in cross_page
        ))

        with tempfile.TemporaryDirectory() as directory:
            graph = Graph(os.path.join(directory, "graph.db"), max_df=1.0)
            try:
                graph.replace_path(
                    "multi-page.pdf",
                    [
                        Chunk("multi-page.pdf", "pdf", 1, 20, page3,
                              section="page 3", source="multi-page.pdf"),
                        Chunk("multi-page.pdf", "pdf", 21, 40, page5,
                              section="page 5", source="multi-page.pdf"),
                    ],
                )
                graph.replace_path(
                    "unrelated.pdf",
                    [Chunk(
                        "unrelated.pdf", "pdf", 1, 10,
                        serialize_page(
                            5, (600, 400),
                            [
                                DiagramNode("N1", "Unrelated input", "rectangle",
                                            "process", 0.92, (20, 120, 220, 220)),
                                DiagramNode("N2", "Unrelated output", "rectangle",
                                            "process", 0.92, (340, 120, 560, 220)),
                            ],
                            [DiagramEdge("N1", "N2", "flows_to", "", 0.85,
                                         (220, 170), (340, 170))],
                        ),
                        section="page 5", source="unrelated.pdf",
                    )],
                )
                hits, facts = graph.relationship_search("A TO PAGE 5", top_k=10)
                self.assertTrue(any(
                    fact["predicate"] == "continues_on_page"
                    and fact["object"] == "Page 5"
                    and fact["inference"] == "reciprocal_page_reference"
                    for fact in facts
                ))
                self.assertTrue(any(hit.get("section") == "page 5" for hit in hits))
                self.assertFalse(any(hit.get("path") == "unrelated.pdf" for hit in hits))
            finally:
                graph.close()

    def test_invalid_page_reference_is_rejected_and_reported(self) -> None:
        nodes = [
            DiagramNode("N1", "Check valve", "rectangle", "process", 0.94,
                        (20, 120, 220, 220)),
            DiagramNode("N2", "A TO PAGE 9", "off_page_connector",
                        "off_page_connector", 0.88, (400, 130, 520, 210)),
        ]
        edges = [
            DiagramEdge("N1", "N2", "flows_to", "", 0.86,
                        (220, 170), (400, 170)),
        ]
        references = _page_references(3, nodes, edges, page_count=5)
        self.assertEqual(len(references), 1)
        self.assertFalse(references[0].valid)
        self.assertEqual(references[0].reason, "page_out_of_range")

        structured = serialize_page(
            3, (600, 400), nodes, edges, page_count=5
        )
        self.assertFalse(any(
            item.predicate == "continues_on_page"
            for item in extract_relationships(structured)
        ))

        with tempfile.TemporaryDirectory() as directory:
            _write_review_report(
                "supplier.pdf",
                page_count=5,
                examined_pages=5,
                pages={3: ((600, 400), nodes, edges)},
                page_notes={},
                references=references,
                reciprocal_ids=set(),
                output_dir=directory,
            )
            reports = os.listdir(directory)
            self.assertEqual(len(reports), 1)
            with open(os.path.join(directory, reports[0]), encoding="utf-8") as source:
                report = json.load(source)

        self.assertEqual(report["summary"]["invalid_references"], 1)
        evidence = report["pages"][0]["page_references"][0]
        self.assertFalse(evidence["valid"])
        self.assertEqual(evidence["reason"], "page_out_of_range")

    def test_figure_sheet_reference_resolves_to_physical_pdf_page(self) -> None:
        page55_nodes = [
            DiagramNode("N1", "TO SHT 57", "off_page_connector",
                        "off_page_connector", 0.90, (20, 120, 220, 220)),
            DiagramNode("N2", "Inspect starter", "rectangle", "process",
                        0.92, (320, 120, 560, 220)),
        ]
        page57_nodes = [
            DiagramNode("N1", "FROM SHT 55", "off_page_connector",
                        "off_page_connector", 0.90, (20, 120, 220, 220)),
            DiagramNode("N2", "Continue test", "rectangle", "process",
                        0.92, (320, 120, 560, 220)),
        ]
        edge = [DiagramEdge("N2", "N1", "flows_to", "", 0.86,
                            (320, 170), (220, 170))]
        serialized, references, reciprocal = _serialize_document_pages(
            {
                206: ((600, 400), page55_nodes, edge),
                208: ((600, 400), page57_nodes, edge),
            },
            page_count=256,
            locator_pages={("2-11", "55"): 206, ("2-11", "57"): 208},
            page_figures={206: "2-11", 208: "2-11"},
        )
        self.assertEqual({item.referenced_page for item in references}, {206, 208})
        self.assertEqual(len(reciprocal), 2)
        self.assertIn("Figure 2-11 Sheet 57 (PDF page 208)", serialized[206])

    def test_mismatched_connector_labels_do_not_receive_reciprocal_confidence(self) -> None:
        departure_nodes = [
            DiagramNode("N1", "Start", "rectangle", "process", 0.94,
                        (20, 120, 220, 220)),
            DiagramNode("N2", "A TO PAGE 5", "off_page_connector",
                        "off_page_connector", 0.88, (400, 130, 520, 210)),
        ]
        arrival_nodes = [
            DiagramNode("N1", "B FROM PAGE 3", "off_page_connector",
                        "off_page_connector", 0.87, (20, 130, 140, 210)),
            DiagramNode("N2", "Finish", "rectangle", "process", 0.93,
                        (320, 120, 560, 220)),
        ]
        serialized, _references, reciprocal_ids = _serialize_document_pages(
            {
                3: ((600, 400), departure_nodes, [
                    DiagramEdge("N1", "N2", "flows_to", "", 0.86,
                                (220, 170), (400, 170)),
                ]),
                5: ((600, 400), arrival_nodes, [
                    DiagramEdge("N1", "N2", "flows_to", "", 0.85,
                                (140, 170), (320, 170)),
                ]),
            },
            page_count=6,
        )
        relationships = extract_relationships("\n\n".join(serialized.values()))
        page_links = [
            item for item in relationships
            if item.predicate in {"continues_on_page", "continues_at"}
        ]
        self.assertFalse(reciprocal_ids)
        self.assertTrue(all(
            item.inference == "explicit_page_reference"
            and item.confidence <= 0.85
            for item in page_links
        ))


if __name__ == "__main__":
    unittest.main()
