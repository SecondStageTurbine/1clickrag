# SPDX-License-Identifier: MPL-2.0
"""SQLite sidecar: keyword search and an entity co-occurrence graph.

Two things the vector index cannot do, in one file that costs no dependency and
no daemon:

* **Exact terms.** Embeddings are worst at precisely the strings people search
  for most confidently - ticket IDs, part numbers, standards, error codes.
  FTS5 is a real BM25 index over the same chunks, and the server fuses the two
  rankings rather than choosing between them.
* **Multi-hop.** "Which supplier feeds the part that failed" is two lookups
  chained through a shared name. Storing which entities appear in which chunk
  turns that into a join, and the connection between two entities is derived
  from the chunks they share rather than asserted by an extractor.

There are deliberately no triples here. A stored "A supplies B" would be a claim
this stack cannot check without a language model; co-occurrence is only a claim
that two names appear together, which is exactly what it is used for - widening
the candidate set before the reranker narrows it again.

`sqlite3` is in the standard library and the bundled interpreter has FTS5, so
this adds nothing to an offline install. The file lives beside the vector
store in .data/ and is rebuilt by a full reindex like everything else there.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from collections import defaultdict

from .chunker import Chunk
from .entities import extract_entities, normalise_key

log = logging.getLogger("rag.graph")

SCHEMA_VERSION = 2

# Bare words for the MATCH expression. Everything else is punctuation as far as
# the unicode61 tokenizer is concerned, so a query keeps only what can match.
TERM = re.compile(r"[^\W_]+", re.UNICODE)
MAX_TERMS = 24


class Graph:
    """The sidecar. One connection, one lock, WAL journal.

    Every method is safe to call from the ingest thread and the request threads
    at once. A single connection under one lock rather than a pool: writes are
    per-file and take milliseconds, and this keeps the "one process, one file"
    property that makes the whole tool portable.
    """

    def __init__(self, path: str, max_df: float = 0.2) -> None:
        self.path = path
        self.max_df = max_df
        self._lock = threading.RLock()
        self._total_docs: int | None = None

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    # -- lifecycle --------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._lock:
            version = 0
            try:
                row = self.db.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()
                version = int(row["value"]) if row else 0
            except sqlite3.Error:
                version = 0

            if version and version != SCHEMA_VERSION:
                # An old sidecar is rebuilt rather than migrated: it is derived
                # data, and a reindex regenerates it from the corpus anyway.
                log.info("graph schema %d != %d - rebuilding", version, SCHEMA_VERSION)
                self._drop_all()
                version = 0

            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id         INTEGER PRIMARY KEY,
                    path       TEXT    NOT NULL,
                    source     TEXT    NOT NULL,
                    language   TEXT    NOT NULL DEFAULT '',
                    symbol     TEXT    NOT NULL DEFAULT '',
                    section    TEXT    NOT NULL DEFAULT '',
                    start_line INTEGER NOT NULL DEFAULT 0,
                    end_line   INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS chunks_path   ON chunks(path);
                CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    text,
                    tokenize = 'unicode61 remove_diacritics 2'
                );

                CREATE TABLE IF NOT EXISTS entities (
                    id        INTEGER PRIMARY KEY,
                    name      TEXT    NOT NULL,
                    key       TEXT    NOT NULL UNIQUE,
                    kind      TEXT    NOT NULL DEFAULT 'name',
                    doc_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS entities_kind ON entities(kind);

                CREATE TABLE IF NOT EXISTS mentions (
                    entity_id INTEGER NOT NULL,
                    chunk_id  INTEGER NOT NULL,
                    count     INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (entity_id, chunk_id)
                ) WITHOUT ROWID;
                CREATE INDEX IF NOT EXISTS mentions_chunk ON mentions(chunk_id);
                """
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.db.commit()

    def _drop_all(self) -> None:
        self.db.executescript(
            """
            DROP TABLE IF EXISTS mentions;
            DROP TABLE IF EXISTS entities;
            DROP TABLE IF EXISTS chunk_fts;
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS meta;
            """
        )
        self.db.commit()
        self._total_docs = None

    def clear(self) -> None:
        """Empty the sidecar, for a --full reindex."""
        with self._lock:
            self.db.executescript(
                """
                DELETE FROM mentions;
                DELETE FROM entities;
                DELETE FROM chunk_fts;
                DELETE FROM chunks;
                """
            )
            self.db.commit()
            self._total_docs = None

    def close(self) -> None:
        with self._lock:
            try:
                self.db.close()
            except Exception:
                pass

    # -- writes -----------------------------------------------------------

    def delete_path(self, path: str) -> None:
        """Drop one real file's chunks, mentions and orphaned entities.

        Matches `source` OR `path` for the same reason Store.delete_path does:
        an archive's members carry the archive as their source, so one call
        clears every document inside it.
        """
        with self._lock:
            touched = self._delete_path_locked(path)
            # Same bookkeeping as replace_path: without it the entities that
            # only lived in this file survive with a document count nothing
            # supports, and keep turning up as graph neighbours.
            self._refresh_doc_counts(touched)
            self.db.commit()

    def _delete_path_locked(self, path: str) -> set[int]:
        rows = self.db.execute(
            "SELECT id FROM chunks WHERE path = ? OR source = ?", (path, path)
        ).fetchall()
        if not rows:
            return set()

        ids = [row["id"] for row in rows]
        touched = {
            row["entity_id"]
            for row in self.db.execute(
                "SELECT DISTINCT entity_id FROM mentions WHERE chunk_id IN (%s)"
                % ",".join("?" * len(ids)),
                ids,
            )
        }
        placeholders = ",".join("?" * len(ids))
        self.db.execute(f"DELETE FROM mentions WHERE chunk_id IN ({placeholders})", ids)
        self.db.execute(f"DELETE FROM chunk_fts WHERE rowid IN ({placeholders})", ids)
        self.db.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
        self._total_docs = None
        return touched

    def replace_path(self, path: str, chunks: list[Chunk]) -> int:
        """Re-index one file: delete what was there, insert what is there now.

        Called from the same place as Store.upsert, so the sidecar cannot drift
        from the vector index across incremental re-ingests - the failure that
        makes hand-rolled secondary indexes untrustworthy.
        """
        with self._lock:
            touched = self._delete_path_locked(path)

            for chunk in chunks:
                cursor = self.db.execute(
                    "INSERT INTO chunks(path, source, language, symbol, section,"
                    " start_line, end_line) VALUES (?,?,?,?,?,?,?)",
                    (
                        chunk.path,
                        chunk.source or chunk.path,
                        chunk.language,
                        chunk.symbol,
                        chunk.section,
                        chunk.start_line,
                        chunk.end_line,
                    ),
                )
                chunk_id = cursor.lastrowid
                self.db.execute(
                    "INSERT INTO chunk_fts(rowid, text) VALUES (?,?)",
                    (chunk_id, chunk.text),
                )

                for mention in extract_entities(chunk.text):
                    row = self.db.execute(
                        "INSERT INTO entities(name, key, kind, doc_count) VALUES (?,?,?,0)"
                        " ON CONFLICT(key) DO UPDATE SET name = entities.name"
                        " RETURNING id",
                        (mention.name, mention.key, mention.kind),
                    ).fetchone()
                    entity_id = row["id"]
                    touched.add(entity_id)
                    self.db.execute(
                        "INSERT INTO mentions(entity_id, chunk_id, count) VALUES (?,?,?)"
                        " ON CONFLICT(entity_id, chunk_id)"
                        " DO UPDATE SET count = count + excluded.count",
                        (entity_id, chunk_id, mention.count),
                    )

            self._refresh_doc_counts(touched)
            self._total_docs = None
            self.db.commit()
            return len(chunks)

    def _refresh_doc_counts(self, entity_ids: set[int]) -> None:
        """Recompute document frequency for the entities this file touched.

        Bounded by the file, not the corpus: df is what the query-time noise
        filter reads, and letting it drift would quietly change which entities
        are considered boilerplate.
        """
        if not entity_ids:
            return
        ids = list(entity_ids)
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            placeholders = ",".join("?" * len(batch))
            self.db.execute(
                f"""
                UPDATE entities SET doc_count = (
                    SELECT COUNT(DISTINCT c.source)
                      FROM mentions m JOIN chunks c ON c.id = m.chunk_id
                     WHERE m.entity_id = entities.id
                ) WHERE id IN ({placeholders})
                """,
                batch,
            )
            self.db.execute(
                f"DELETE FROM entities WHERE doc_count = 0 AND id IN ({placeholders})",
                batch,
            )

    # -- reads ------------------------------------------------------------

    def has_data(self) -> bool:
        """Cheap "is there anything here" for the per-query guard.

        A COUNT(*) on every search would scan; this stops at the first row.
        """
        with self._lock:
            return self.db.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is not None

    def count_chunks(self) -> int:
        with self._lock:
            row = self.db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
            return int(row["n"]) if row else 0

    def total_docs(self) -> int:
        with self._lock:
            if self._total_docs is None:
                row = self.db.execute(
                    "SELECT COUNT(DISTINCT source) AS n FROM chunks"
                ).fetchone()
                self._total_docs = int(row["n"]) if row else 0
            return self._total_docs

    def df_ceiling(self) -> int:
        """How many documents an entity may appear in before it is boilerplate.

        A name in most of the corpus - a header, a footer, the company's own
        name on every page - connects everything to everything, which is worse
        than useless in a graph. The floor of 3 keeps small corpora usable.
        """
        return max(3, int(self.total_docs() * self.max_df))

    def stats(self) -> dict:
        with self._lock:
            def scalar(sql: str) -> int:
                row = self.db.execute(sql).fetchone()
                return int(row["n"]) if row else 0

            # The write-ahead log holds everything not yet checkpointed, so the
            # main file alone reports a sidecar far smaller than it is on disk.
            size = 0
            for suffix in ("", "-wal", "-shm"):
                try:
                    size += os.path.getsize(self.path + suffix)
                except OSError:
                    pass
            return {
                "chunks": scalar("SELECT COUNT(*) AS n FROM chunks"),
                "documents": self.total_docs(),
                "entities": scalar("SELECT COUNT(*) AS n FROM entities"),
                "mentions": scalar("SELECT COUNT(*) AS n FROM mentions"),
                "df_ceiling": self.df_ceiling(),
                "bytes": size,
            }

    def _match_expression(self, query: str) -> str:
        terms = []
        for term in TERM.findall(query):
            if len(term) < 2:
                continue
            terms.append('"' + term.replace('"', '""') + '"')
            if len(terms) >= MAX_TERMS:
                break
        # OR rather than the implicit AND: a natural-language question shares
        # few exact words with the passage that answers it, and BM25 already
        # rewards the documents that match more of them.
        return " OR ".join(terms)

    def keyword_search(
        self,
        query: str,
        top_k: int,
        language_filter: str | None = None,
        path_prefix: str | None = None,
    ) -> list[dict]:
        """BM25 over the same chunks the vector index holds."""
        expression = self._match_expression(query)
        if not expression:
            return []

        sql = [
            "SELECT c.id, c.path, c.language, c.symbol, c.section, c.start_line, c.end_line,",
            "       chunk_fts.text AS text, bm25(chunk_fts) AS rank",
            "  FROM chunk_fts JOIN chunks c ON c.id = chunk_fts.rowid",
            " WHERE chunk_fts MATCH ?",
        ]
        params: list = [expression]
        if language_filter:
            sql.append("   AND c.language = ?")
            params.append(language_filter)
        if path_prefix:
            # substr() rather than LIKE: no wildcard escaping to get wrong.
            sql.append("   AND substr(c.path, 1, ?) = ?")
            params.extend([len(path_prefix), path_prefix])
        sql.append(" ORDER BY rank LIMIT ?")
        # Over-fetch for the same reason the vector search does: overlapping
        # chunks from one file would otherwise fill the results.
        params.append(max(top_k * 5, 20))

        with self._lock:
            rows = self.db.execute("\n".join(sql), params).fetchall()
        return _dedupe_overlaps(
            [
                {
                    "path": row["path"],
                    "language": row["language"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "symbol": row["symbol"],
                    "section": row["section"],
                    # bm25() is negative with better matches more negative;
                    # flipped so "higher is better" holds everywhere.
                    "score": round(-float(row["rank"]), 6),
                    "text": row["text"],
                }
                for row in rows
            ],
            top_k,
        )

    # -- entities ---------------------------------------------------------

    def resolve_entities(self, text: str, limit: int = 8) -> list[dict]:
        """Entities in the index that this text mentions.

        Matching is on the normalised key, not on capitalisation: a question is
        typed in lower case ("who supplies northwind?") and would otherwise
        resolve to nothing at all. Word n-grams up to four long are looked up
        directly, longest first, and a match swallows the shorter matches
        inside it so "Northwind Industries" does not also return "Northwind".
        """
        words = [word for word in TERM.findall(text) if word]
        candidates: dict[str, str] = {}
        for size in range(4, 0, -1):
            for start in range(0, len(words) - size + 1):
                phrase = " ".join(words[start : start + size])
                key = normalise_key(phrase)
                if key and key not in candidates:
                    candidates[key] = phrase

        # Identifiers keep punctuation the tokenizer would have thrown away.
        for mention in extract_entities(text, max_entities=16):
            candidates.setdefault(mention.key, mention.name)

        if not candidates:
            return []

        keys = list(candidates)
        found: list[dict] = []
        with self._lock:
            for start in range(0, len(keys), 400):
                batch = keys[start : start + 400]
                placeholders = ",".join("?" * len(batch))
                rows = self.db.execute(
                    f"""
                    SELECT e.id, e.name, e.key, e.kind, e.doc_count,
                           (SELECT COUNT(*) FROM mentions m WHERE m.entity_id = e.id) AS chunks
                      FROM entities e WHERE e.key IN ({placeholders})
                    """,
                    batch,
                ).fetchall()
                found.extend(dict(row) for row in rows)

        # Longest surface form wins; drop anything covered by a longer match.
        found.sort(key=lambda row: -len(row["key"]))
        kept: list[dict] = []
        for row in found:
            if any(row["key"] in other["key"] for other in kept):
                continue
            # A lone lower-case word in the question that happens to match an
            # all-capitals "acronym" is a heading the extractor picked up, not
            # a name being asked about: "what does it cost" must not seed COST.
            surface = candidates.get(row["key"], "")
            if (
                row["kind"] == "acronym"
                and len(surface.split()) == 1
                and not surface.isupper()
            ):
                continue
            kept.append(row)
            if len(kept) >= limit:
                break
        return kept

    def top_entities(
        self,
        limit: int = 50,
        kind: str | None = None,
        contains: str | None = None,
        path_prefix: str | None = None,
        include_common: bool = False,
    ) -> list[dict]:
        sql = [
            "SELECT e.id, e.name, e.kind, e.doc_count,",
            "       COUNT(DISTINCT m.chunk_id) AS chunks, SUM(m.count) AS mentions",
            "  FROM entities e JOIN mentions m ON m.entity_id = e.id",
        ]
        params: list = []
        where = []
        if path_prefix:
            sql.append("  JOIN chunks c ON c.id = m.chunk_id")
            where.append("substr(c.path, 1, ?) = ?")
            params.extend([len(path_prefix), path_prefix])
        if kind:
            where.append("e.kind = ?")
            params.append(kind)
        if contains:
            where.append("e.key LIKE ?")
            params.append(f"%{normalise_key(contains)}%")
        if not include_common:
            where.append("e.doc_count <= ?")
            params.append(self.df_ceiling())
        if where:
            sql.append(" WHERE " + " AND ".join(where))
        sql.append(" GROUP BY e.id ORDER BY mentions DESC LIMIT ?")
        params.append(limit)

        with self._lock:
            rows = self.db.execute("\n".join(sql), params).fetchall()
        return [dict(row) for row in rows]

    def neighbours(
        self, entity_ids: list[int], limit: int = 12, exclude: set[int] | None = None
    ) -> list[dict]:
        """Entities that share chunks with these ones.

        This is the edge set, computed rather than stored. Deriving it from
        `mentions` means an incremental re-ingest can never leave a stale edge
        behind pointing at a document that changed.
        """
        if not entity_ids:
            return []
        skip = set(entity_ids) | (exclude or set())
        placeholders = ",".join("?" * len(entity_ids))
        skip_placeholders = ",".join("?" * len(skip))
        # Ranked by lift, not by raw co-occurrence. Counting shared chunks alone
        # promotes whatever is ubiquitous - a word that appears near everything
        # shares chunks with everything - and buries the specific name that
        # only ever appears next to this one. Dividing by the neighbour's own
        # frequency asks the more useful question: of the places this entity
        # turns up, how many are here?
        sql = f"""
            SELECT e.id, e.name, e.kind, e.doc_count,
                   COUNT(DISTINCT m2.chunk_id) AS shared,
                   (SELECT COUNT(*) FROM mentions mm WHERE mm.entity_id = e.id) AS total
              FROM mentions m1
              JOIN mentions m2 ON m2.chunk_id = m1.chunk_id
              JOIN entities e  ON e.id = m2.entity_id
             WHERE m1.entity_id IN ({placeholders})
               AND m2.entity_id NOT IN ({skip_placeholders})
               AND e.doc_count <= ?
             GROUP BY e.id
             ORDER BY (shared * shared) / (total * 1.0) DESC, shared DESC
             LIMIT ?
        """
        params = list(entity_ids) + list(skip) + [self.df_ceiling(), limit]
        with self._lock:
            rows = self.db.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def expand(
        self, entity_ids: list[int], hops: int, fanout: int = 8
    ) -> list[dict]:
        """Breadth-first walk from the seeds, `hops` steps out.

        Returns every entity reached with the hop it was reached at, seeds
        included at hop 0, so a caller can weight nearer ones more heavily.
        """
        reached: dict[int, dict] = {}
        with self._lock:
            rows = self.db.execute(
                "SELECT id, name, kind, doc_count FROM entities WHERE id IN (%s)"
                % ",".join("?" * len(entity_ids)),
                entity_ids,
            ).fetchall() if entity_ids else []
        for row in rows:
            reached[row["id"]] = dict(row, hop=0, shared=None)

        frontier = list(reached)
        for hop in range(1, max(hops, 0) + 1):
            if not frontier:
                break
            found = self.neighbours(frontier, limit=fanout * max(len(frontier), 1),
                                    exclude=set(reached))
            frontier = []
            for row in found[: fanout * 2]:
                if row["id"] in reached:
                    continue
                reached[row["id"]] = dict(row, hop=hop)
                frontier.append(row["id"])
        return sorted(reached.values(), key=lambda row: (row["hop"], -(row.get("shared") or 0)))

    def chunks_for_entities(
        self,
        entity_ids: list[int],
        top_k: int,
        language_filter: str | None = None,
        path_prefix: str | None = None,
    ) -> list[dict]:
        """Chunks mentioning these entities, most-entities-matched first."""
        if not entity_ids:
            return []
        placeholders = ",".join("?" * len(entity_ids))
        sql = [
            "SELECT c.id, c.path, c.language, c.symbol, c.section, c.start_line, c.end_line,",
            "       f.text AS text,",
            "       COUNT(DISTINCT m.entity_id) AS matched, SUM(m.count) AS mentions",
            "  FROM mentions m",
            "  JOIN chunks c    ON c.id = m.chunk_id",
            "  JOIN chunk_fts f ON f.rowid = c.id",
            f" WHERE m.entity_id IN ({placeholders})",
        ]
        params: list = list(entity_ids)
        if language_filter:
            sql.append("   AND c.language = ?")
            params.append(language_filter)
        if path_prefix:
            sql.append("   AND substr(c.path, 1, ?) = ?")
            params.extend([len(path_prefix), path_prefix])
        sql.append(" GROUP BY c.id ORDER BY matched DESC, mentions DESC LIMIT ?")
        params.append(max(top_k * 5, 20))

        with self._lock:
            rows = self.db.execute("\n".join(sql), params).fetchall()
        return _dedupe_overlaps(
            [
                {
                    "path": row["path"],
                    "language": row["language"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "symbol": row["symbol"],
                    "section": row["section"],
                    "score": float(row["matched"]),
                    "text": row["text"],
                    "entities_matched": int(row["matched"]),
                }
                for row in rows
            ],
            top_k,
        )

    def link(self, a_id: int, b_id: int, limit: int = 3) -> list[dict]:
        """The chunks that put two entities in the same place.

        This is what makes a hop inspectable: the edge is not an assertion, it
        is these passages, and a reader can go and check them.
        """
        sql = """
            SELECT c.path, c.start_line, c.end_line
              FROM mentions m1
              JOIN mentions m2 ON m2.chunk_id = m1.chunk_id
              JOIN chunks c    ON c.id = m1.chunk_id
             WHERE m1.entity_id = ? AND m2.entity_id = ?
             ORDER BY (m1.count + m2.count) DESC
             LIMIT ?
        """
        with self._lock:
            rows = self.db.execute(sql, (a_id, b_id, limit)).fetchall()
        return [
            {
                "path": row["path"],
                "citation": f"{row['path']}:{row['start_line']}-{row['end_line']}",
            }
            for row in rows
        ]

    def path_between(self, start: str, end: str, max_hops: int = 3, fanout: int = 12) -> dict:
        """Shortest chain of shared documents from one name to another."""
        seeds = self.resolve_entities(start, limit=1)
        targets = self.resolve_entities(end, limit=1)
        if not seeds or not targets:
            return {
                "found": False,
                "from": seeds[0]["name"] if seeds else None,
                "to": targets[0]["name"] if targets else None,
                "detail": "one of the two names is not in the index",
                "steps": [],
            }

        source, target = seeds[0], targets[0]
        if source["id"] == target["id"]:
            return {"found": True, "from": source["name"], "to": target["name"], "steps": []}

        # Breadth-first, expanding one node at a time rather than the whole
        # frontier at once: the chain has to be walked back afterwards, so each
        # neighbour must remember which node actually reached it.
        parents: dict[int, tuple[int, dict]] = {}
        seen = {source["id"]}
        frontier = [source["id"]]
        reached_target = False

        for _ in range(max(max_hops, 1)):
            next_frontier: list[int] = []
            for node in frontier:
                for row in self.neighbours([node], limit=fanout, exclude=seen):
                    if row["id"] in parents or row["id"] == source["id"]:
                        continue
                    parents[row["id"]] = (node, row)
                    next_frontier.append(row["id"])
                    if row["id"] == target["id"]:
                        reached_target = True
                        break
                if reached_target:
                    break
            seen.update(next_frontier)
            if reached_target or not next_frontier:
                break
            frontier = next_frontier

        if target["id"] not in parents:
            return {
                "found": False,
                "from": source["name"],
                "to": target["name"],
                "detail": f"no chain within {max_hops} hop(s)",
                "steps": [],
            }

        chain: list[dict] = []
        node = target["id"]
        while node != source["id"]:
            previous, row = parents[node]
            chain.append(
                {
                    "from_id": previous,
                    "to_id": node,
                    "to": row["name"],
                    "shared_chunks": row["shared"],
                    "evidence": self.link(previous, node),
                }
            )
            node = previous
        chain.reverse()

        with self._lock:
            names = {
                row["id"]: row["name"]
                for row in self.db.execute(
                    "SELECT id, name FROM entities WHERE id IN (%s)"
                    % ",".join("?" * len({step["from_id"] for step in chain})),
                    list({step["from_id"] for step in chain}),
                ).fetchall()
            }
        for step in chain:
            step["from"] = names.get(step["from_id"], "")
            step.pop("from_id", None)
            step.pop("to_id", None)

        return {
            "found": True,
            "from": source["name"],
            "to": target["name"],
            "hops": len(chain),
            "steps": chain,
        }


def _dedupe_overlaps(hits: list[dict], top_k: int) -> list[dict]:
    """Drop a chunk whose lines overlap one already kept from the same file.

    Mirrors Store.search: chunking overlaps deliberately, so neighbouring
    chunks share most of their text and would otherwise return three views of
    one passage instead of three answers.
    """
    results: list[dict] = []
    kept_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for hit in hits:
        spans = kept_spans[hit["path"]]
        start, end = int(hit["start_line"] or 0), int(hit["end_line"] or 0)
        if any(start <= kept_end and end >= kept_start for kept_start, kept_end in spans):
            continue
        spans.append((start, end))
        results.append(hit)
        if len(results) >= top_k:
            break
    return results


_GRAPHS: dict[str, Graph] = {}
_GRAPHS_LOCK = threading.Lock()


def open_graph(cfg) -> Graph | None:
    """The process-wide sidecar for this config, or None when disabled.

    Cached by path so the ingest thread, the watcher and the API share one
    connection - and so a caller never has to thread it through a signature
    the way the vector store has to be.
    """
    if not cfg.graph:
        return None
    with _GRAPHS_LOCK:
        graph = _GRAPHS.get(cfg.graph_path)
        if graph is None:
            try:
                graph = Graph(cfg.graph_path, max_df=cfg.graph_max_df)
            except Exception as exc:
                # A broken sidecar must not take the vector index down with it:
                # everything it powers degrades to "vector search only".
                log.warning("graph disabled - could not open %s: %s", cfg.graph_path, exc)
                return None
            _GRAPHS[cfg.graph_path] = graph
        return graph
