# SPDX-License-Identifier: MPL-2.0
"""A durable work queue for indexing, in SQLite.

The watcher used to re-index inline: notice a change, embed it there and then.
That loses work in two ways. A file whose read fails - locked by the
application that is still writing it, a share that blinked - got a log line and
was forgotten, so the index stayed wrong until someone reindexed by hand. And
anything noticed but not yet embedded died with the process, which matters
precisely because this is meant to run unattended for weeks.

So changes are recorded first and processed second. The record is a table in
its own SQLite file, which means it survives a crash, a reboot and a kill -9,
and a failure becomes a retry with backoff rather than a silent hole. After
enough attempts an item moves to a dead-letter table, where it stops consuming
attempts and starts being evidence: `GET /queue` lists what is stuck and why.

Deliberately its own file rather than a table in graph.db: the graph is an
optional feature, and losing queued work because someone set RAG_GRAPH=0 would
be an absurd way to couple the two.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time

log = logging.getLogger("rag.queue")

SCHEMA_VERSION = 1

# Backoff between attempts. Short enough that a file locked by Word is picked
# up while you are still looking at it, long enough that a share which is down
# is not hammered for an hour.
BASE_BACKOFF_SECONDS = 5.0
MAX_BACKOFF_SECONDS = 600.0


class WorkQueue:
    """Pending index/delete work, persisted across restarts."""

    def __init__(self, path: str, max_attempts: int = 5) -> None:
        self.path = path
        self.max_attempts = max_attempts
        self._lock = threading.RLock()

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")  # durability is the point
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                -- path is the primary key, so a file saved ten times while the
                -- queue is busy is one unit of work, not ten.
                CREATE TABLE IF NOT EXISTS work (
                    path            TEXT    PRIMARY KEY,
                    op              TEXT    NOT NULL,
                    enqueued_at     REAL    NOT NULL,
                    attempts        INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL    NOT NULL DEFAULT 0,
                    last_error      TEXT
                );
                CREATE INDEX IF NOT EXISTS work_due ON work(next_attempt_at);
                CREATE TABLE IF NOT EXISTS dead (
                    path      TEXT    PRIMARY KEY,
                    op        TEXT    NOT NULL,
                    attempts  INTEGER NOT NULL,
                    failed_at REAL    NOT NULL,
                    error     TEXT
                );
                """
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.db.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self.db.close()
            except Exception:
                pass

    # -- writing ----------------------------------------------------------

    def enqueue(self, paths: list[str], op: str = "index") -> int:
        """Record work to be done. Re-queuing a known path resets its backoff.

        A path arriving again is news: whatever failed last time may well
        succeed now, and the user who just saved the file should not wait out
        the previous item's exponential backoff.
        """
        if not paths:
            return 0
        now = time.time()
        with self._lock:
            for path in paths:
                self.db.execute(
                    """
                    INSERT INTO work(path, op, enqueued_at, attempts, next_attempt_at)
                         VALUES (?,?,?,0,?)
                    ON CONFLICT(path) DO UPDATE SET
                         op = excluded.op,
                         attempts = 0,
                         next_attempt_at = excluded.next_attempt_at,
                         last_error = NULL
                    """,
                    (path, op, now, now),
                )
                # A file that comes back is no longer dead.
                self.db.execute("DELETE FROM dead WHERE path = ?", (path,))
            self.db.commit()
        return len(paths)

    def complete(self, paths: list[str]) -> None:
        if not paths:
            return
        with self._lock:
            self.db.executemany("DELETE FROM work WHERE path = ?", [(p,) for p in paths])
            self.db.commit()

    def fail(self, path: str, error: str) -> bool:
        """Record an attempt that failed. Returns True if it was given up on."""
        with self._lock:
            row = self.db.execute(
                "SELECT op, attempts FROM work WHERE path = ?", (path,)
            ).fetchone()
            if row is None:
                return False

            attempts = int(row["attempts"]) + 1
            if attempts >= self.max_attempts:
                self.db.execute(
                    "INSERT OR REPLACE INTO dead(path, op, attempts, failed_at, error)"
                    " VALUES (?,?,?,?,?)",
                    (path, row["op"], attempts, time.time(), error[:2000]),
                )
                self.db.execute("DELETE FROM work WHERE path = ?", (path,))
                self.db.commit()
                log.warning(
                    "giving up on %s after %d attempts - moved to the dead letter "
                    "queue, see GET /queue: %s",
                    path,
                    attempts,
                    error,
                )
                return True

            delay = min(BASE_BACKOFF_SECONDS * (2 ** (attempts - 1)), MAX_BACKOFF_SECONDS)
            self.db.execute(
                "UPDATE work SET attempts = ?, next_attempt_at = ?, last_error = ?"
                " WHERE path = ?",
                (attempts, time.time() + delay, error[:2000], path),
            )
            self.db.commit()
            log.info("retrying %s in %.0fs (attempt %d): %s", path, delay, attempts, error)
            return False

    def retry_dead(self, path: str | None = None) -> int:
        """Move dead letters back into the queue, for after a fix."""
        with self._lock:
            if path:
                rows = self.db.execute(
                    "SELECT path, op FROM dead WHERE path = ?", (path,)
                ).fetchall()
            else:
                rows = self.db.execute("SELECT path, op FROM dead").fetchall()
            if not rows:
                return 0
            now = time.time()
            for row in rows:
                self.db.execute(
                    "INSERT OR REPLACE INTO work(path, op, enqueued_at, attempts,"
                    " next_attempt_at, last_error) VALUES (?,?,?,0,?,NULL)",
                    (row["path"], row["op"], now, now),
                )
                self.db.execute("DELETE FROM dead WHERE path = ?", (row["path"],))
            self.db.commit()
            return len(rows)

    # -- reading ----------------------------------------------------------

    def claim(self, limit: int = 25) -> list[dict]:
        """The next items whose backoff has elapsed."""
        with self._lock:
            rows = self.db.execute(
                "SELECT path, op, attempts FROM work WHERE next_attempt_at <= ?"
                " ORDER BY enqueued_at LIMIT ?",
                (time.time(), limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> dict:
        with self._lock:
            pending = self.db.execute("SELECT COUNT(*) AS n FROM work").fetchone()["n"]
            waiting = self.db.execute(
                "SELECT COUNT(*) AS n FROM work WHERE next_attempt_at > ?", (time.time(),)
            ).fetchone()["n"]
            dead = self.db.execute("SELECT COUNT(*) AS n FROM dead").fetchone()["n"]
        return {"pending": int(pending), "retrying": int(waiting), "dead": int(dead)}

    def snapshot(self, limit: int = 50) -> dict:
        with self._lock:
            work = self.db.execute(
                "SELECT path, op, attempts, enqueued_at, next_attempt_at, last_error"
                " FROM work ORDER BY enqueued_at LIMIT ?",
                (limit,),
            ).fetchall()
            dead = self.db.execute(
                "SELECT path, op, attempts, failed_at, error FROM dead"
                " ORDER BY failed_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {
            "counts": self.counts(),
            "pending": [dict(row) for row in work],
            "dead": [dict(row) for row in dead],
        }
