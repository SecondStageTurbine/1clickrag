# SPDX-License-Identifier: MPL-2.0
"""Qdrant-backed vector store."""

from __future__ import annotations

import logging
import os
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .chunker import Chunk

log = logging.getLogger("rag.store")

_NAMESPACE = uuid.UUID("6f1a6a2e-1f3d-4a2e-9c4b-0f9c1c2f8f11")


def _point_id(path: str, index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{path}#{index}"))


class Store:
    """Vector storage, either embedded (file-backed) or against a server.

    An empty ``url`` selects qdrant-client's embedded mode: same query engine,
    same API, no daemon and no container — it just keeps its segments in
    ``path``. That mode is single-process (it takes an exclusive lock on the
    directory), which is exactly right for a local index served by one process.
    """

    def __init__(self, url: str, collection: str, path: str | None = None) -> None:
        self.url = url
        self.collection = collection
        self.path = path
        self.embedded = not url
        if self.embedded:
            if not path:
                raise ValueError("embedded mode needs a storage path")
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(url=url, timeout=60.0)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    # -- lifecycle --------------------------------------------------------

    def alive(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # qdrant-client raises a family of transport errors
            return False

    def wait_until_alive(self, timeout: float = 180.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.alive():
                return True
            time.sleep(2.0)
        return False

    def exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection)
        except Exception:
            return False

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        if recreate and self.exists():
            log.info("dropping collection %s", self.collection)
            self.client.delete_collection(self.collection)
        if self.exists():
            return
        log.info("creating collection %s (dim=%d)", self.collection, dim)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        # Keyword indexes make the language filter a cheap server-side
        # prefilter. Embedded mode scans linearly and ignores them, so skip the
        # call there rather than emit a warning per field.
        if not self.embedded:
            for field in ("language", "path", "source"):
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection,
                        field_name=field,
                        field_schema=qm.PayloadSchemaType.KEYWORD,
                    )
                except Exception as exc:  # index already present is fine
                    log.debug("payload index %s: %s", field, exc)

    def count(self) -> int:
        try:
            return int(self.client.count(self.collection, exact=True).count)
        except Exception:
            return 0

    def indexed_state(self) -> dict[str, dict]:
        """Every indexed path with the source file and mtime it came from.

        One scroll of the payloads, no vectors and no text, so this is cheap
        enough to run on every startup - which is the point: it is the record
        of what the index believes, to be compared against what is on disk.
        """
        state: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=2048,
                offset=offset,
                with_payload=["path", "source", "mtime"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                path = payload.get("path")
                if not path:
                    continue
                # Chunks of one file all carry the same values; first wins.
                state.setdefault(
                    path,
                    {
                        "source": payload.get("source") or path,
                        # Absent for anything indexed before mtime was recorded.
                        # Treated as "unknown", which reindexes it once.
                        "mtime": payload.get("mtime"),
                    },
                )
            if offset is None:
                break
        return state

    def indexed_paths(self) -> set[str]:
        paths: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=1024,
                offset=offset,
                with_payload=["path"],
                with_vectors=False,
            )
            for point in points:
                path = (point.payload or {}).get("path")
                if path:
                    paths.add(path)
            if offset is None:
                break
        return paths

    # -- writes -----------------------------------------------------------

    def delete_path(self, path: str) -> None:
        """Remove everything indexed from one real file.

        Matches `source` OR `path`: an archive's members carry the archive as
        their source, so one call clears them all, and points written before
        `source` existed are still matched by `path`.
        """
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    should=[
                        qm.FieldCondition(key="source", match=qm.MatchValue(value=path)),
                        qm.FieldCondition(key="path", match=qm.MatchValue(value=path)),
                    ]
                )
            ),
        )

    def upsert(
        self, chunks: list[Chunk], vectors: list[list[float]], mtime: float | None = None
    ) -> None:
        points = [
            qm.PointStruct(
                id=_point_id(chunk.path, i),
                vector=vector,
                payload={
                    "path": chunk.path,
                    "source": chunk.source or chunk.path,
                    "language": chunk.language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "symbol": chunk.symbol,
                    # When the file on disk was last written. This is what lets
                    # a later run tell "already indexed" from "indexed, then
                    # edited while nothing was watching".
                    "mtime": mtime,
                    "text": chunk.text,
                },
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    # -- reads ------------------------------------------------------------

    def search(
        self,
        vector: list[float],
        top_k: int,
        language_filter: str | None = None,
        path_prefix: str | None = None,
    ) -> list[dict]:
        query_filter = None
        if language_filter:
            query_filter = qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="language", match=qm.MatchValue(value=language_filter)
                    )
                ]
            )

        # Over-fetch, then filter and de-duplicate client-side. Two reasons:
        # Qdrant has no native prefix match on a keyword field, and chunking
        # overlaps deliberately - an oversized section is windowed with a few
        # lines of overlap, so neighbouring chunks share most of their text and
        # score almost identically. Returned raw, that fills the results with
        # three views of one passage instead of ten distinct answers.
        limit = top_k * 20 if path_prefix else top_k * 5

        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=max(limit, 1),
            query_filter=query_filter,
            with_payload=True,
        ).points

        results: list[dict] = []
        kept_spans: dict[str, list[tuple[int, int]]] = {}
        for hit in hits:
            payload = hit.payload or {}
            if path_prefix and not str(payload.get("path", "")).startswith(path_prefix):
                continue

            # Drop a chunk whose lines overlap one already kept from the same
            # file. Hits arrive best-first, so the survivor is the better match.
            path = str(payload.get("path", ""))
            start = int(payload.get("start_line", 0) or 0)
            end = int(payload.get("end_line", 0) or 0)
            spans = kept_spans.setdefault(path, [])
            if any(start <= kept_end and end >= kept_start for kept_start, kept_end in spans):
                continue
            spans.append((start, end))

            results.append(
                {
                    "path": payload.get("path", ""),
                    "language": payload.get("language", ""),
                    "start_line": payload.get("start_line", 0),
                    "end_line": payload.get("end_line", 0),
                    "symbol": payload.get("symbol", ""),
                    "score": round(float(hit.score), 6),
                    "text": payload.get("text", ""),
                }
            )
            if len(results) >= top_k:
                break
        return results
