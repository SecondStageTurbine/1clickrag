# SPDX-License-Identifier: MPL-2.0
"""Disposable, bounded SQLite cache of document embeddings (no source text)."""

from contextlib import closing
import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time

log = logging.getLogger("rag.embedding_cache")


def _valid(vector, dimension: int) -> bool:
    return (isinstance(vector, list) and len(vector) == dimension
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in vector))


class CachedEmbedder:
    """Delegate model operations; reuse only document vectors, preserving order.

    Connections live for one operation, so standalone ingestion and server
    shutdown need no extra cleanup. SQLite serializes writes across processes.
    Cache failures fall back to embedding; model failures still propagate.
    """

    def __init__(self, embedder, path: str, fingerprint: str, max_bytes: int):
        self._embedder = embedder
        self._path = path
        self._fingerprint = fingerprint
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._hits = self._misses = 0

    def __getattr__(self, name):
        return getattr(self._embedder, name)

    def cache_stats(self) -> dict:
        with self._lock:
            return {"enabled": True, "hits": self._hits, "misses": self._misses}

    def _connect(self):
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        db = sqlite3.connect(self._path, timeout=5)
        try:
            db.execute("""CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY, vector BLOB NOT NULL,
                checksum TEXT NOT NULL, accessed REAL NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS embedding_age ON embeddings(accessed)")
            # Maintain the byte total incrementally, rather than scanning the
            # entire cache after each file in a large corpus.
            db.execute("CREATE TABLE IF NOT EXISTS embedding_usage (id INTEGER PRIMARY KEY CHECK(id=1), bytes INTEGER NOT NULL)")
            if db.execute("SELECT 1 FROM embedding_usage WHERE id=1").fetchone() is None:
                db.execute("INSERT OR IGNORE INTO embedding_usage SELECT 1, COALESCE(SUM(length(vector)), 0) FROM embeddings")
            db.execute("""CREATE TRIGGER IF NOT EXISTS embedding_insert AFTER INSERT ON embeddings
                BEGIN UPDATE embedding_usage SET bytes=bytes+length(NEW.vector) WHERE id=1; END""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS embedding_delete AFTER DELETE ON embeddings
                BEGIN UPDATE embedding_usage SET bytes=bytes-length(OLD.vector) WHERE id=1; END""")
            db.execute("""CREATE TRIGGER IF NOT EXISTS embedding_update AFTER UPDATE OF vector ON embeddings
                BEGIN UPDATE embedding_usage SET bytes=bytes+length(NEW.vector)-length(OLD.vector) WHERE id=1; END""")
            db.commit()
        except Exception:
            db.close()
            raise
        return db

    def _get(self, keys: list[str], dimension: int) -> dict:
        found = {}
        with closing(self._connect()) as db, db:
            for start in range(0, len(keys), 500):
                batch = keys[start:start + 500]
                marks = ",".join("?" for _ in batch)
                rows = db.execute(
                    f"SELECT key, vector, checksum FROM embeddings WHERE key IN ({marks})", batch)
                for key, blob, checksum in rows:
                    try:
                        if hashlib.sha256(blob).hexdigest() != checksum:
                            continue
                        vector = json.loads(blob)
                        if _valid(vector, dimension):
                            found[key] = vector
                    except (ValueError, TypeError, OverflowError):
                        continue
            db.executemany("UPDATE embeddings SET accessed=? WHERE key=?",
                           [(time.time(), key) for key in found])
        return found

    def _put(self, values: dict) -> None:
        records = []
        for key, vector in values.items():
            blob = json.dumps(vector, separators=(",", ":"), allow_nan=False).encode()
            if len(blob) <= self._max_bytes:
                records.append((key, blob, hashlib.sha256(blob).hexdigest(), time.time()))
        with closing(self._connect()) as db, db:
            db.executemany("""INSERT INTO embeddings VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET vector=excluded.vector,
                checksum=excluded.checksum, accessed=excluded.accessed""", records)
            excess = db.execute("SELECT bytes FROM embedding_usage WHERE id=1").fetchone()[0] - self._max_bytes
            # Bound stored vector bytes. Freed pages are reused by SQLite;
            # the file also has index/page overhead and is not vacuumed here.
            while excess > 0:
                oldest = db.execute("SELECT key, length(vector) FROM embeddings ORDER BY accessed LIMIT 128").fetchall()
                remove = []
                for key, size in oldest:
                    remove.append((key,))
                    excess -= size
                    if excess <= 0:
                        break
                db.executemany("DELETE FROM embeddings WHERE key=?", remove)

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        if not texts:
            return []
        dimension = self._embedder.dim
        keys = [hashlib.sha256(
            f"{self._fingerprint}:{dimension}:document:".encode() + text.encode("utf-8")
        ).hexdigest() for text in texts]
        unique = dict(zip(keys, texts))
        try:
            found = self._get(list(unique), dimension)
        except (OSError, sqlite3.Error) as exc:
            log.warning("embedding cache read failed; computing vectors: %s", exc)
            found = {}
        with self._lock:
            self._hits += sum(key in found for key in keys)
            self._misses += sum(key not in found for key in keys)
        missing = [key for key in unique if key not in found]
        if missing:
            vectors = self._embedder.embed_documents([unique[key] for key in missing], batch_size=batch_size)
            if len(vectors) != len(missing) or any(not _valid(v, dimension) for v in vectors):
                from .embedder import EmbeddingError
                raise EmbeddingError("embedding backend returned an invalid vector count, dimension, or value")
            fresh = dict(zip(missing, vectors))
            found.update(fresh)
            try:
                self._put(fresh)
            except (OSError, sqlite3.Error) as exc:
                log.warning("embedding cache write failed; using computed vectors: %s", exc)
        return [list(found[key]) for key in keys]
