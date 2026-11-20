from __future__ import annotations

import os
import tempfile
import unittest

from app.chunker import Chunk, chunk_file
from app.extract import from_xlsx
from app.graph import Graph
from app.relationships import extract_relationships


class SpreadsheetExtractionTests(unittest.TestCase):
    def test_xlsx_rows_keep_headers_sheet_and_row_provenance(self) -> None:
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl is optional")

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "self-test.xlsx")
            book = Workbook()
            sheet = book.active
            sheet.title = "Fault Mapping"
            sheet.append(["Fault Condition", "Parameter Name", "ARINC Label", "Artifact"])
            sheet.append(["Low pressure", "Hydraulic Pressure", 213, "BIT Report"])
            book.save(path)

            text = from_xlsx(path)

        self.assertIsNotNone(text)
        assert text is not None
        self.assertIn("[sheet Fault Mapping]", text)
        self.assertIn("[row 2]", text)
        self.assertIn("Fault Condition: Low pressure", text)
        self.assertIn("Parameter Name: Hydraulic Pressure", text)
        self.assertIn("ARINC Label: 213", text)
        self.assertIn("Artifact: BIT Report", text)

        chunks = chunk_file("self-test.xlsx", "text", text, chunk_chars=1600)
        self.assertTrue(any("ARINC Label: 213" in chunk.text for chunk in chunks))

    def test_explicit_schema_extracts_only_supported_row_relationships(self) -> None:
        text = (
            "[sheet Fault Mapping]\n[row 2]\n"
            "Fault Condition: Low pressure\n"
            "Parameter Name: Hydraulic Pressure\n"
            "ARINC Label: 213\n"
            "Artifact: BIT Report"
        )
        relations = extract_relationships(text)
        triples = {(r.subject, r.predicate, r.object) for r in relations}
        self.assertIn(
            ("Low pressure", "maps_to_parameter", "Hydraulic Pressure"), triples
        )
        self.assertIn(
            ("Hydraulic Pressure", "uses_arinc_label", "213"), triples
        )
        self.assertTrue(all(r.confidence == 1.0 for r in relations))
        self.assertTrue(all(r.inference == "explicit" for r in relations))


class HybridRelationshipRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.graph = Graph(os.path.join(self.directory.name, "graph.db"), max_df=1.0)
        row = Chunk(
            path="self-test.xlsx",
            source="self-test.xlsx",
            language="text",
            start_line=1,
            end_line=6,
            text=(
                "[sheet Fault Mapping]\n[row 2]\n"
                "Fault Condition: Low pressure\n"
                "Parameter Name: Hydraulic Pressure\n"
                "ARINC Label: 213\n"
                "Artifact: BIT Report"
            ),
        )
        supporting = Chunk(
            path="parameter-guide.md",
            source="parameter-guide.md",
            language="markdown",
            start_line=10,
            end_line=12,
            text="Hydraulic Pressure is sampled by the flight data interface.",
        )
        self.graph.replace_path("self-test.xlsx", [row])
        self.graph.replace_path("parameter-guide.md", [supporting])

    def tearDown(self) -> None:
        self.graph.close()
        self.directory.cleanup()

    def test_parameter_label_fault_and_artifact_queries_regress(self) -> None:
        for query in ("Hydraulic Pressure", "213", "Low pressure", "BIT Report"):
            with self.subTest(query=query):
                self.assertTrue(self.graph.keyword_search(query, top_k=5))

    def test_fault_query_retrieves_linked_parameter_across_documents(self) -> None:
        hits, facts = self.graph.relationship_search("Low pressure", top_k=10)
        self.assertTrue(any(f["predicate"] == "maps_to_parameter" for f in facts))
        self.assertTrue(any(hit["path"] == "parameter-guide.md" for hit in hits))

    def test_real_xlsx_flows_from_extraction_through_relationship_retrieval(self) -> None:
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl is optional")

        path = os.path.join(self.directory.name, "integration.xlsx")
        book = Workbook()
        sheet = book.active
        sheet.title = "Self Test"
        sheet.append(["Fault Condition", "Parameter Name", "ARINC Label", "Artifact"])
        sheet.append(["ADC disagreement", "Computed Airspeed", "204", "Flight Test Report"])
        book.save(path)

        extracted = from_xlsx(path)
        self.assertIsNotNone(extracted)
        chunks = chunk_file("integration.xlsx", "text", extracted or "", chunk_chars=1600)
        for chunk in chunks:
            chunk.source = "integration.xlsx"
        self.graph.replace_path("integration.xlsx", chunks)

        for query, predicate, linked_value in (
            ("ADC disagreement", "maps_to_parameter", "Computed Airspeed"),
            ("Computed Airspeed", "uses_arinc_label", "204"),
            ("Flight Test Report", "contains_parameter", "Computed Airspeed"),
            ("204", "uses_arinc_label", "Computed Airspeed"),
        ):
            with self.subTest(query=query):
                hits, facts = self.graph.relationship_search(query, top_k=10)
                self.assertTrue(hits)
                self.assertTrue(
                    any(
                        fact["predicate"] == predicate
                        and linked_value in (fact["subject"], fact["object"])
                        and fact["confidence"] == 1.0
                        and fact["inference"] == "explicit"
                        and fact["citation"].startswith("integration.xlsx:")
                        for fact in facts
                    )
                )

    def test_supplier_style_identifiers_and_colon_labels_resolve(self) -> None:
        row = Chunk(
            path="supplier.pdf!faults.xlsx",
            source="supplier.pdf",
            language="text",
            start_line=1,
            end_line=4,
            text=(
                "[row 2]\n"
                "Parameter Name: APU_ECU_NONCRIT_INTERNAL_FAULT\n"
                "ARINC Label: 350:11"
            ),
        )
        self.graph.replace_path("supplier.pdf!faults.xlsx", [row])
        for query in ("APU_ECU_NONCRIT_INTERNAL_FAULT", "350:11"):
            with self.subTest(query=query):
                hits, facts = self.graph.relationship_search(query, top_k=5)
                self.assertTrue(hits)
                self.assertTrue(
                    any(fact["predicate"] == "uses_arinc_label" for fact in facts)
                )


if __name__ == "__main__":
    unittest.main()
