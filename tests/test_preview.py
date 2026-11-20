import os
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from app.chunker import chunk_file
from app.config import Config
from app.extract_cache import fingerprint_file
from app.preview import router, version
from app.preview_pdf import render
from app.store import Store


def make_pdf(path, rotate=0):
    writer = PdfWriter()
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                             NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    for text in ('Introduction to this example.', 'The pressure limit is 42 units.'):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 18 Tf 60 600 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
        if rotate:
            page.rotate(rotate)
    with open(path, 'wb') as handle:
        writer.write(handle)


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = Config(repo_path=str(self.root), max_file_bytes=100_000_000)
        self.store = Store('', 'preview-test', str(self.root / 'index'))
        self.addCleanup(self.store.close)
        self.store.ensure_collection(2)
        app = FastAPI()
        app.include_router(router(self.cfg, self.store))
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def index(self, name, text, legacy=False):
        target = self.root / name
        if target.suffix != '.pdf':
            target.write_text(text, encoding='utf-8')
        chunks = chunk_file(name, 'text', text, chunk_chars=1600, overlap_lines=0)
        self.store.upsert(chunks, [[1.0, 0.0] for _ in chunks],
                          mtime=round(target.stat().st_mtime, 3),
                          source_hash=None if legacy else fingerprint_file(str(target)))
        return version(self.store.document_info(name))

    def test_text_preview_and_exact_line_highlight_range(self):
        v = self.index('sample.txt', 'First line\nThe cited evidence.\nLast line')
        response = self.client.post('/preview', json={'path': 'sample.txt', 'start_line': 2, 'end_line': 2, 'version': v})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['kind'], 'text')
        self.assertEqual(data['lines'][1], {'n': 2, 'text': 'The cited evidence.'})
        self.assertEqual(data['warning'], '')
        self.assertEqual(response.headers['cache-control'], 'no-store')

    def test_pdf_citation_page_and_original_highlight(self):
        make_pdf(self.root / 'sample.pdf')
        v = self.index('sample.pdf', '[page 1]\nIntroduction to this example.\n\n[page 2]\nThe pressure limit is 42 units.')
        query = {'path': 'sample.pdf', 'start_line': 5, 'end_line': 5, 'version': v}
        response = self.client.post('/preview', json=query)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['initial_page'], 2)
        response = self.client.post('/preview/page', json={**query, 'page': 2})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertTrue(data['matched'])
        self.assertEqual(data['pages'], 2)
        self.assertTrue(data['image'].startswith('data:image/png;base64,'))
        self.assertTrue(all(0 <= box['x'] < 1 and 0 <= box['y'] < 1 for box in data['boxes']))

    def test_changed_file_with_preserved_timestamp_disables_pdf(self):
        make_pdf(self.root / 'sample.pdf')
        v = self.index('sample.pdf', '[page 1]\nIntroduction to this example.')
        target = self.root / 'sample.pdf'
        stamp = target.stat()
        target.write_bytes(b'changed file')
        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        query = {'path': 'sample.pdf', 'start_line': 2, 'end_line': 2, 'version': v}
        data = self.client.post('/preview', json=query).json()
        self.assertEqual(data['kind'], 'text')
        self.assertIn('changed since indexing', data['warning'])
        self.assertEqual(self.client.post('/preview/page', json=query).status_code, 409)

    def test_old_answer_after_reindex_is_rejected(self):
        old = self.index('sample.txt', 'Original answer.')
        self.index('sample.txt', 'Changed answer.')
        response = self.client.post('/preview', json={'path': 'sample.txt', 'start_line': 1, 'end_line': 1, 'version': old})
        self.assertEqual(response.status_code, 409)

    def test_new_extraction_of_same_pdf_invalidates_old_line_references(self):
        make_pdf(self.root / 'sample.pdf')
        old = self.index('sample.pdf', '[page 1]\nOriginal extraction.')
        self.index('sample.pdf', '[page 1]\nNew extraction settings changed the text.')
        response = self.client.post('/preview', json={'path': 'sample.pdf', 'start_line': 2, 'end_line': 2, 'version': old})
        self.assertEqual(response.status_code, 409)

    def test_unindexed_and_unsafe_paths_are_rejected(self):
        (self.root / 'unindexed.txt').write_text('private', encoding='utf-8')
        for path in ('../outside.txt', '/absolute.txt', 'C:/secret.txt', 'a\\b.txt'):
            with self.subTest(path=path):
                response = self.client.post('/preview', json={'path': path, 'start_line': 1, 'end_line': 1})
                self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.post('/preview', json={'path': 'unindexed.txt', 'start_line': 1, 'end_line': 1}).status_code, 404)
        self.assertIn(self.client.post('/preview', json={'path': '.env', 'start_line': 1, 'end_line': 1}).status_code, (403, 404))

    def test_missing_file_and_legacy_index_are_explicit(self):
        self.index('sample.txt', 'An older indexed document.', legacy=True)
        query = {'path': 'sample.txt', 'start_line': 1, 'end_line': 1}
        self.assertIn('Older index', self.client.post('/preview', json=query).json()['warning'])
        (self.root / 'sample.txt').unlink()
        self.assertIn('unavailable', self.client.post('/preview', json=query).json()['warning'])

    def test_pdf_unmatched_and_rotated_pages(self):
        target = self.root / 'rotated.pdf'
        make_pdf(target, rotate=90)
        self.assertFalse(render(str(target), 2, 'This passage does not exist.')['matched'])
        result = render(str(target), 2, 'The pressure limit is 42 units.')
        self.assertTrue(result['matched'])
        for box in result['boxes']:
            self.assertGreaterEqual(box['x'], 0)
            self.assertLessEqual(box['x'] + box['w'], 1)
            self.assertGreaterEqual(box['y'], 0)
            self.assertLessEqual(box['y'] + box['h'], 1)


if __name__ == '__main__':
    unittest.main()
