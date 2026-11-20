from __future__ import annotations

import io
import os
import tempfile
import unittest
from unittest import mock

from app.archive import SEPARATOR, read_member
from app.config import Config
from app.pdf_attachments import (
    EmbeddedFile,
    attachment_staging,
    list_attachments,
    read_attachment,
)
from app.scanner import iter_source_files, load_text


def _xlsx_bytes() -> bytes:
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.title = "Faults"
    sheet.append(["Parameter Name", "ARINC Label", "Description"])
    sheet.append(["APU_TEST_FAULT", "350:11", "Controller internal fault"])
    output = io.BytesIO()
    book.save(output)
    return output.getvalue()


def _pdf_with_file_annotation(path: str, filename: str, data: bytes) -> None:
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        FloatObject,
        NameObject,
        TextStringObject,
    )

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    stream = DecodedStreamObject()
    stream.set_data(data)
    stream[NameObject("/Type")] = NameObject("/EmbeddedFile")
    stream_ref = writer._add_object(stream)
    filespec = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Filespec"),
            NameObject("/F"): TextStringObject(filename),
            NameObject("/UF"): TextStringObject(filename),
            NameObject("/EF"): DictionaryObject({NameObject("/F"): stream_ref}),
        }
    )
    annotation = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/FileAttachment"),
            NameObject("/Rect"): ArrayObject(
                [FloatObject(10), FloatObject(10), FloatObject(30), FloatObject(30)]
            ),
            NameObject("/FS"): writer._add_object(filespec),
        }
    )
    page[NameObject("/Annots")] = ArrayObject([writer._add_object(annotation)])
    with open(path, "wb") as output:
        writer.write(output)


class PdfAttachmentTests(unittest.TestCase):
    def test_unchanged_members_are_listed_without_using_temporary_storage(self) -> None:
        def items(_path):
            yield EmbeddedFile("unchanged.xlsx", b"old")
            yield EmbeddedFile("changed.xlsx", b"new-data")

        with tempfile.TemporaryDirectory() as directory:
            pdf = os.path.join(directory, "supplier.pdf")
            with open(pdf, "wb") as output:
                output.write(b"placeholder")
            with mock.patch("app.pdf_attachments._iter_embedded_files", side_effect=items):
                with attachment_staging(100) as staging:
                    staging.set_member_filter(
                        lambda _path, name: name == "changed.xlsx"
                    )
                    members = list_attachments(pdf, 1_000, 100, 10, 100)
                    unchanged = read_attachment(pdf, "unchanged.xlsx", 100)
                    changed = read_attachment(pdf, "changed.xlsx", 100)
                    staged_bytes = staging.temp_bytes

        self.assertEqual(
            members,
            [("unchanged.xlsx", 3), ("changed.xlsx", 8)],
        )
        self.assertIsNone(unchanged)
        self.assertEqual(changed, b"new-data")
        self.assertEqual(staged_bytes, 8)

    def test_ingest_staging_reads_a_pdf_batch_once_and_cleans_up(self) -> None:
        calls = 0

        def items(_path):
            nonlocal calls
            calls += 1
            yield EmbeddedFile("one.csv", b"Name,Value\nAlpha,1\n")
            yield EmbeddedFile("two.csv", b"Name,Value\nBeta,2\n")

        with tempfile.TemporaryDirectory() as directory:
            pdf = os.path.join(directory, "supplier.pdf")
            with open(pdf, "wb") as output:
                output.write(b"placeholder")
            with mock.patch("app.pdf_attachments._iter_embedded_files", side_effect=items):
                with attachment_staging(100) as staging:
                    members = list_attachments(pdf, 1_000, 100, 10, 100)
                    staged_directory = staging.temp_dir
                    self.assertIn("Alpha | 1", read_member(pdf, "one.csv", 100) or "")
                    self.assertIn("Beta | 2", read_member(pdf, "two.csv", 100) or "")
                    self.assertIn(b"Alpha", read_attachment(pdf, "one.csv", 100) or b"")
                    self.assertEqual(calls, 1)
                    self.assertTrue(os.path.isdir(staged_directory))
            self.assertFalse(os.path.exists(staged_directory))
            self.assertEqual(
                members,
                [("one.csv", 19), ("two.csv", 18)],
            )

    def test_total_and_temporary_byte_limits_bound_staging(self) -> None:
        def items(_path):
            yield EmbeddedFile("one.xlsx", b"1234")
            yield EmbeddedFile("two.xlsx", b"5678")

        with tempfile.TemporaryDirectory() as directory:
            pdf = os.path.join(directory, "supplier.pdf")
            with open(pdf, "wb") as output:
                output.write(b"placeholder")

            with mock.patch("app.pdf_attachments._iter_embedded_files", side_effect=items):
                with attachment_staging(100):
                    total_limited = list_attachments(pdf, 1_000, 100, 10, 7)
            with mock.patch("app.pdf_attachments._iter_embedded_files", side_effect=items):
                with attachment_staging(7):
                    temp_limited = list_attachments(pdf, 1_000, 100, 10, 100)
                    first = read_attachment(pdf, "one.xlsx", 100)
                    second = read_attachment(pdf, "two.xlsx", 100)

        self.assertEqual(total_limited, [("one.xlsx", 4)])
        self.assertEqual(
            temp_limited,
            [("one.xlsx", 4), ("two.xlsx", 4)],
        )
        self.assertEqual(first, b"1234")
        self.assertIsNone(second)

    def test_attachment_limit_stops_streaming_early(self) -> None:
        consumed: list[int] = []

        def items():
            for index in range(100):
                consumed.append(index)
                yield EmbeddedFile(f"sheet-{index}.xlsx", b"data")

        with tempfile.TemporaryDirectory() as directory:
            pdf = os.path.join(directory, "supplier.pdf")
            with open(pdf, "wb") as output:
                output.write(b"placeholder")
            with mock.patch(
                "app.pdf_attachments._iter_embedded_files", return_value=items()
            ):
                members = list_attachments(pdf, 1_000, 100, 2)

        self.assertEqual(
            members,
            [("sheet-0.xlsx", 4), ("sheet-1.xlsx", 4)],
        )
        self.assertEqual(consumed, [0, 1, 2])

    def test_file_annotation_xlsx_is_discovered_and_header_parsed(self) -> None:
        try:
            import openpyxl  # noqa: F401
            import pypdf  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as directory:
            pdf = os.path.join(directory, "supplier.pdf")
            name = "Appendix Faults.xlsx"
            _pdf_with_file_annotation(pdf, name, _xlsx_bytes())

            members = list_attachments(pdf, 10_000_000, 1_000_000, 10)
            self.assertEqual([item[0] for item in members], [name])
            cfg = Config()
            cfg.repo_path = directory
            discovered = list(iter_source_files(cfg))
            self.assertIn(("supplier.pdf", "pdf"), discovered)
            self.assertIn((f"supplier.pdf{SEPARATOR}{name}", "xlsx"), discovered)
            text = load_text(f"{pdf}{SEPARATOR}{name}", 1_000_000)

        self.assertIsNotNone(text)
        assert text is not None
        self.assertIn("[sheet Faults]", text)
        self.assertIn("Parameter Name: APU_TEST_FAULT", text)
        self.assertIn("ARINC Label: 350:11", text)


if __name__ == "__main__":
    unittest.main()
