# SPDX-License-Identifier: MPL-2.0
"""Keeping the expensive half of an ingest so it need not happen twice.

Indexing a file is two jobs of wildly different cost. Turning a PDF into text
can take seconds - minutes, with OCR - while chunking and embedding that text
takes milliseconds. Every rebuild redid both, so `reindex -Full` on a scanned
corpus cost 1h38m here, essentially all of it re-reading pages whose bytes had
not changed since the last run. The README's advice was to avoid the command.

That is the wrong shape. A full reindex exists for when something about
*chunking or embedding* changes - a new chunk size, a different model, another
label in the vector - and none of those alter what the words on page 12 are. So
the extracted text is kept, keyed by the content it came from, and a rebuild
re-chunks and re-embeds without re-reading.

Two things decide when a cached extract is still valid, and both are in the key
rather than checked afterwards, so a stale entry cannot be returned by mistake:

**The content**, as a SHA-256 of the bytes. Modification time would be cheaper
and is what the incremental scan already trusts, but this cache has to survive
`reindex -Full`, and distrusting timestamps is precisely why someone runs that.
Hashing costs one sequential read against an extraction measured in seconds.

**The extraction profile** - every setting that changes the output rather than
merely the input. OCR on or off, its band and confidence, whether PDF headers
are stripped. Change one and the key changes with it, so the old text is not
reused and the new text does not overwrite it either; switch back and yesterday's
entry is still there.

It holds verbatim document text, which puts it in the same category as the
keyword index: never committed, never packaged, and dropped by `down -Wipe`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
import zlib

log = logging.getLogger("rag.cache")

SCHEMA_VERSION = 1
# Bump when an extractor's output changes for input it already handled, so
# every entry made by the old code is bypassed. Config-driven differences do
# not need this - they are in the profile - but a bug fix in a reader does.
EXTRACTOR_VERSION = 1


def _profile(cfg) -> str:
    """A short digest of every setting that changes what extraction returns."""
    parts = [
        f"v{EXTRACTOR_VERSION}",
        f"ocr={int(bool(cfg.ocr))}",
        f"band={cfg.ocr_band}",
        f"overlap={cfg.ocr_overlap}",
        f"conf={cfg.ocr_min_confidence}",
        f"minchars={cfg.ocr_min_chars}",
        f"mpx={cfg.ocr_max_megapixels}",
        f"pdfboiler={int(bool(os.environ.get('RAG_PDF_KEEP_BOILERPLATE')))}",
        f"members={cfg.archive_max_members}",
        f"extra={sorted(cfg.extra_text_exts.items())}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def fingerprint_file(path: str) -> str | None:
    """SHA-256 of a file's bytes, or None if it cannot be read."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


class ExtractCache:
    def __init__(self, path: str, profile: str, max_bytes: int) -> None:
        self.path = path
        self.profile = profile
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        # Losing this to a crash costs time, never correctness - the extract is
        # reproducible from the file. NORMAL rather than FULL accordingly.
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS extracts ("
            "  key TEXT PRIMARY KEY,"
            "  body BLOB NOT NULL,"
            "  bytes INTEGER NOT NULL,"
            "  used_at REAL NOT NULL)"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS extracts_used ON extracts(used_at)")
        self.db.commit()
        self.hits = 0
        self.misses = 0

    def _key(self, fingerprint: str) -> str:
        return f"{self.profile}:{fingerprint}"

    def get(self, fingerprint: str) -> str | None:
        key = self._key(fingerprint)
        with self._lock:
            row = self.db.execute(
                "SELECT body FROM extracts WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.misses += 1
                return None
            # Touched so pruning drops what is genuinely cold rather than what
            # happens to be oldest.
            self.db.execute(
                "UPDATE extracts SET used_at = ? WHERE key = ?", (time.time(), key)
            )
            self.db.commit()
            self.hits += 1
        try:
            return zlib.decompress(row[0]).decode("utf-8")
        except (zlib.error, UnicodeDecodeError):
            return None  # corrupt entry: treat as absent, it will be rewritten

    def put(self, fingerprint: str, text: str) -> None:
        # Level 6 rather than max: prose compresses about four to one either
        # way, and the difference is measured in milliseconds per file against
        # seconds saved per rebuild.
        body = zlib.compress(text.encode("utf-8"), 6)
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO extracts (key, body, bytes, used_at)"
                " VALUES (?, ?, ?, ?)",
                (self._key(fingerprint), body, len(body), time.time()),
            )
            self.db.commit()
        self._prune()

    def _prune(self) -> None:
        with self._lock:
            row = self.db.execute("SELECT SUM(bytes) AS n FROM extracts").fetchone()
            total = int(row["n"] or 0) if hasattr(row, "keys") else int(row[0] or 0)
            if total <= self.max_bytes:
                return
            # Coldest first, until back under budget. A cache that grows without
            # limit on a corpus of scans is a disk-space incident.
            target = int(self.max_bytes * 0.9)
            for key, size in self.db.execute(
                "SELECT key, bytes FROM extracts ORDER BY used_at ASC"
            ).fetchall():
                self.db.execute("DELETE FROM extracts WHERE key = ?", (key,))
                total -= size
                if total <= target:
                    break
            self.db.commit()

    def stats(self) -> dict:
        with self._lock:
            row = self.db.execute(
                "SELECT COUNT(*) , COALESCE(SUM(bytes), 0) FROM extracts"
            ).fetchone()
        return {
            "entries": int(row[0]),
            "bytes": int(row[1]),
            "hits": self.hits,
            "misses": self.misses,
        }


_cache: ExtractCache | None = None
_opened = False
_open_lock = threading.Lock()


def open_cache(cfg):
    """The process's cache, or None when disabled or unopenable."""
    global _cache, _opened
    if _opened:
        return _cache
    with _open_lock:
        if _opened:
            return _cache
        _opened = True
        if not getattr(cfg, "extract_cache", True):
            return None
        try:
            _cache = ExtractCache(
                cfg.extract_cache_path,
                _profile(cfg),
                cfg.extract_cache_max_mb * 1024 * 1024,
            )
        except sqlite3.Error as exc:
            # A cache that cannot open is a slower ingest, never a failed one.
            log.warning("extraction cache unavailable: %s", exc)
            _cache = None
        return _cache


def reset_for_tests() -> None:
    global _cache, _opened
    _cache, _opened = None, False
