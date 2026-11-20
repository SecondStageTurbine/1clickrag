from __future__ import annotations

from contextlib import closing
from dataclasses import replace
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.config import Config
from app.embedder import EmbeddingError, make_embedder
from app.embedding_cache import CachedEmbedder
from app.embedding_contract import fingerprint
from app import manifest


class FakeEmbedder:
    dim = 2
    backend = "fake"

    def __init__(self):
        self.calls = []

    def embed_documents(self, texts, batch_size=16):
        self.calls.append(list(texts))
        return [[float(len(t)), float(sum(t.encode()))] for t in texts]

    def embed_query(self, text):
        return [42.0, 1.0]


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = os.path.join(self.temp.name, "vectors.db")
        self.backend = FakeEmbedder()
        self.cache = CachedEmbedder(self.backend, self.path, "model-v1", 10000)

    def test_reuses_across_edits_reorders_duplicates_and_restart(self):
        first = self.cache.embed_documents(["one", "two", "one"])
        self.assertEqual(self.backend.calls, [["one", "two"]])
        restarted_backend = FakeEmbedder()
        restarted = CachedEmbedder(restarted_backend, self.path, "model-v1", 10000)
        result = restarted.embed_documents(["two", "changed", "one", "one"])
        self.assertEqual(restarted_backend.calls, [["changed"]])
        self.assertEqual(result[0], first[1])
        self.assertEqual(result[2:], [first[0], first[0]])
        self.assertIsNot(result[2], result[3])
        self.assertEqual(restarted.cache_stats(), {"enabled": True, "hits": 3, "misses": 1})

    def test_model_and_dimension_have_separate_namespaces(self):
        self.cache.embed_documents(["one"])
        other = FakeEmbedder()
        CachedEmbedder(other, self.path, "model-v2", 10000).embed_documents(["one"])
        self.assertEqual(other.calls, [["one"]])
        other.dim = 3
        with patch.object(other, "embed_documents", return_value=[[1.0, 2.0, 3.0]]) as encode:
            actual = CachedEmbedder(other, self.path, "model-v1", 10000).embed_documents(["one"])
        encode.assert_called_once()
        self.assertEqual(actual, [[1.0, 2.0, 3.0]])

    def test_corrupted_vector_is_recomputed(self):
        expected = self.cache.embed_documents(["one"])
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE embeddings SET vector=?", (b"[99,99]",))
        self.assertEqual(self.cache.embed_documents(["one"]), expected)
        self.assertEqual(len(self.backend.calls), 2)

    def test_invalid_vectors_never_enter_cache(self):
        for vectors in ([], [[1.0]], [[float("nan"), 2.0]], [[True, 2.0]]):
            with self.subTest(vectors=vectors):
                with patch.object(self.backend, "embed_documents", return_value=vectors):
                    with self.assertRaises(EmbeddingError):
                        self.cache.embed_documents(["one"])
        self.cache.embed_documents(["one"])
        self.assertEqual(self.backend.calls, [["one"]])

    def test_unavailable_cache_falls_back_and_queries_bypass_cache(self):
        cache = CachedEmbedder(self.backend, self.temp.name, "model-v1", 10000)
        with self.assertLogs("rag.embedding_cache", level="WARNING"):
            self.assertEqual(cache.embed_documents(["one"]), [[3.0, 322.0]])
        self.assertEqual(cache.embed_query("one"), [42.0, 1.0])
        self.assertEqual(self.backend.calls, [["one"]])

    def test_eviction_respects_recent_use_and_byte_budget(self):
        cache = CachedEmbedder(self.backend, self.path, "v1", 30)
        cache.embed_documents(["one", "two"])
        cache.embed_documents(["one"])
        cache.embed_documents(["three"])
        cache.embed_documents(["one"])
        self.assertEqual(self.backend.calls, [["one", "two"], ["three"]])
        with closing(sqlite3.connect(self.path)) as db:
            self.assertLessEqual(db.execute("SELECT SUM(length(vector)) FROM embeddings").fetchone()[0], 30)
        cache.embed_documents(["two"])
        self.assertEqual(self.backend.calls[-1], ["two"])

    def test_factory_can_disable_cache(self):
        for cfg in (Config(embed_cache=False), Config(embed_cache_max_mb=0)):
            with patch("app.embedder._make_backend", return_value=self.backend):
                self.assertIs(make_embedder(cfg), self.backend)
        with patch("app.embedder._make_backend", return_value=self.backend):
            enabled = make_embedder(Config(embed_cache=True, embed_cache_path=self.path))
            self.assertIsInstance(enabled, CachedEmbedder)
            self.assertEqual(enabled.backend, "fake")

    def test_ingest_reuses_unchanged_section_but_not_changed_labels_or_paths(self):
        from app.ingest import IngestStats, index_file

        cfg = Config(repo_path=self.temp.name, embed_path="relative", embed_label="section")
        first = "## First\nKeep this text.\n\n## Second\nOriginal text.\n"
        second = first.replace("Original text.", "Edited text.")
        store = Mock()
        with patch("app.ingest.open_graph", return_value=None):
            with patch("app.ingest.load_text", return_value=first):
                index_file("notes.md", "markdown", cfg, self.cache, store, IngestStats())
            with patch("app.ingest.load_text", return_value=second):
                index_file("notes.md", "markdown", cfg, self.cache, store, IngestStats())
            self.assertEqual(len(self.backend.calls[0]), 2)
            self.assertEqual(len(self.backend.calls[1]), 1)
            self.assertIn("Edited text.", self.backend.calls[1][0])
            with patch("app.ingest.load_text", return_value=second):
                index_file("renamed.md", "markdown", cfg, self.cache, store, IngestStats())
            self.assertEqual(len(self.backend.calls[2]), 2)
            with patch("app.ingest.load_text", return_value=second):
                index_file("renamed.md", "markdown", replace(cfg, embed_label="off"),
                           self.cache, store, IngestStats())
            self.assertEqual(len(self.backend.calls[3]), 2)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(embed_backend="http", embed_url="http://localhost:9999/v1",
                          embed_doc_prefix="doc: ", embed_query_prefix="query: ")

    def test_semantic_settings_change_fingerprint_and_manifest(self):
        before = manifest.current(self.cfg, 2)
        for field, value in (("embed_doc_prefix", "passage: "),
                             ("embed_query_prefix", "question: "),
                             ("embed_revision", "new-weights"),
                             ("embed_url", "http://localhost:9998/v1"),
                             ("embed_model", "different-model")):
            with self.subTest(field=field):
                cfg = replace(self.cfg, **{field: value})
                self.assertNotEqual(fingerprint(cfg), fingerprint(self.cfg))
                fields = {d["setting"] for d in manifest.drift(before, manifest.current(cfg, 2))["rebuild"]}
                self.assertIn("embed_fingerprint", fields)

    def test_nonsemantic_settings_and_credentials_do_not_invalidate(self):
        cfg = replace(self.cfg, embed_api_key="secret", embed_batch=1,
                      embed_cache=False, embed_http_timeout=1)
        self.assertEqual(fingerprint(cfg), fingerprint(self.cfg))
        self.assertNotIn("localhost", fingerprint(cfg))

    def test_fastembed_runtime_change_and_inactive_prefixes(self):
        cfg = replace(self.cfg, embed_backend="fastembed")
        self.assertEqual(fingerprint(cfg), fingerprint(replace(cfg, embed_doc_prefix="ignored")))
        with patch("app.embedding_contract._version", return_value="old"):
            old = fingerprint(cfg)
        with patch("app.embedding_contract._version", return_value="new"):
            self.assertNotEqual(old, fingerprint(cfg))

    def test_legacy_manifest_retains_existing_checks(self):
        old = manifest.current(self.cfg, 2)
        del old["embed_fingerprint"]
        self.assertEqual(manifest.drift(old, manifest.current(self.cfg, 2))["rebuild"], [])
        changed = manifest.current(replace(self.cfg, embed_model="new"), 2)
        self.assertIn("embed_model", {x["setting"] for x in manifest.drift(old, changed)["rebuild"]})


if __name__ == "__main__":
    unittest.main()
