# SPDX-License-Identifier: MPL-2.0
"""HTTP API + browser UI for the RAG index.

Endpoints (the JSON contract matches what this project's coders already call):

    GET  /                -> browser search UI
    GET  /health          -> {"status":"healthy","ollama":true,"qdrant":true,...}
    GET  /stats           -> index size, model, repo, last ingest
    POST /search          -> truncated snippets
    POST /search/full     -> full chunk text
    POST /context         -> hits assembled into a citable block for a generator
    POST /reindex         -> {"full": bool} kick off a background re-ingest
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import CONFIG, corpus_warning
from .embedder import make_embedder
from .ingest import run_ingest
from .reranker import Reranker
from .store import Store
from .watcher import start_watcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("rag.server")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Lets a launcher tell our server apart from anything else listening on the port.
SERVICE_ID = "rag-local"

# One embedder and one store for the whole process. In embedded-Qdrant mode
# the storage directory is exclusively locked by whoever opens it, so the
# bootstrap, the watcher, /reindex and /search must all share these.
embedder = make_embedder(CONFIG)
store = Store(CONFIG.qdrant_url, CONFIG.collection, CONFIG.qdrant_path)
reranker = (
    Reranker(CONFIG.rerank_model, CONFIG.model_cache, CONFIG.embed_threads)
    if CONFIG.rerank
    else None
)

STATE: dict = {
    "ingest_running": False,
    "ingest_error": None,
    "last_ingest": None,
    "started_at": time.time(),
    "warning": None,
}


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    language_filter: str | None = None
    path_prefix: str | None = None


def _bootstrap() -> None:
    """First-boot sequence: wait for deps, pull the model, ingest if empty."""
    STATE["ingest_running"] = True
    STATE["ingest_error"] = None
    warning = corpus_warning()
    if warning:
        log.warning("%s", warning)
        STATE["warning"] = warning
    try:
        if not store.wait_until_alive():
            raise RuntimeError(f"qdrant unreachable at {CONFIG.qdrant_url}")
        if not embedder.wait_until_alive():
            raise RuntimeError(f"embedding backend unreachable at {CONFIG.ollama_url}")

        needs_ingest = CONFIG.ingest_on_start and (
            not store.exists() or store.count() == 0
        )
        if needs_ingest:
            log.info("index is empty - running first ingest")
            run_ingest(CONFIG, full=False, store=store, embedder=embedder)
            STATE["last_ingest"] = time.time()
        else:
            if CONFIG.auto_pull_model:
                embedder.prepare()
            store.ensure_collection(embedder.dim)
            log.info("index already holds %d chunks - skipping ingest", store.count())

        if reranker:
            # Load it now rather than on the first query, which would otherwise
            # pay the download and the model load while a user waits.
            reranker.prepare()

        if CONFIG.watch:
            start_watcher(CONFIG, store=store, embedder=embedder)
    except Exception as exc:
        STATE["ingest_error"] = str(exc)
        log.error("bootstrap failed: %s", exc)
    finally:
        STATE["ingest_running"] = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Bootstrap off the event loop: the API must answer /health immediately so
    # the start scripts and the UI can show progress while the first ingest
    # (model pull + full index) is still running.
    threading.Thread(target=_bootstrap, name="rag-bootstrap", daemon=True).start()
    yield


app = FastAPI(
    title="Local RAG", version="1.0.0", docs_url="/api-docs", lifespan=lifespan
)


@app.get("/")
def index_page():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    embed_ok = embedder.alive()
    qdrant_ok = store.alive()
    chunks = store.count() if qdrant_ok else 0

    # Precedence matters: a dependency being down outranks everything (an
    # ingest that is "running" against a dead Qdrant is waiting, not working),
    # and an in-progress ingest outranks a partially-filled index — so
    # "healthy" means the index is complete, which is what the start scripts
    # and the UI wait on.
    if not (embed_ok and qdrant_ok):
        status = "degraded"
    elif STATE["ingest_running"]:
        status = "indexing"
    elif chunks > 0:
        status = "healthy"
    else:
        status = "empty"

    payload = {
        # Identity marker. Port 8404 has a prior occupant on some machines (the
        # hand-rolled RAG this replaces), and it also answers /health with
        # status=healthy — so a launcher that only checks "does something reply"
        # will report success while our own server has quietly failed to bind.
        "service": SERVICE_ID,
        "status": status,
        "mode": CONFIG.mode,
        "embeddings": embed_ok,
        "embed_backend": embedder.backend,
        "rerank": bool(reranker),
        "rerank_model": CONFIG.rerank_model if reranker else None,
        "qdrant": qdrant_ok,
        "vector_store": "embedded" if store.embedded else "server",
        "chunks": chunks,
        "model": CONFIG.embed_model,
        "collection": CONFIG.collection,
        "indexing": STATE["ingest_running"],
        "error": STATE["ingest_error"],
        "warning": STATE["warning"],
    }
    # Kept for continuity with the pre-existing Ollama-backed deployment, where
    # callers looked at `.ollama`. Omitted in native mode rather than reported
    # as a meaningless false: `.status` is the field to check.
    if embedder.backend == "ollama":
        payload["ollama"] = embed_ok
    return payload


@app.get("/stats")
def stats():
    return {
        "service": SERVICE_ID,
        "repo": CONFIG.repo_label,
        "repo_path": CONFIG.repo_path,
        "mode": CONFIG.mode,
        "collection": CONFIG.collection,
        "chunks": store.count(),
        "model": CONFIG.embed_model,
        "embed_backend": embedder.backend,
        "vector_store": "embedded" if store.embedded else "server",
        "watching": CONFIG.watch,
        "indexing": STATE["ingest_running"],
        "last_ingest": STATE["last_ingest"],
        "uptime_seconds": round(time.time() - STATE["started_at"], 1),
        "error": STATE["ingest_error"],
    }


def _do_search(request: SearchRequest, truncate: bool) -> JSONResponse:
    if not store.exists() or store.count() == 0:
        state = health()
        detail = {
            "degraded": "the embedding backend or the vector store is not ready — check the logs",
            "indexing": "the first ingest is still running",
            "empty": "nothing indexed yet — POST /reindex to build the index",
        }.get(state["status"], "the index is not ready")
        return JSONResponse(
            status_code=503,
            content={
                "error": "index is not ready",
                "status": state["status"],
                "detail": STATE["ingest_error"] or detail,
                "indexing": STATE["ingest_running"],
            },
        )

    vector = embedder.embed_query(request.query)
    # With reranking on, the vector search is a wide cheap filter and the
    # cross-encoder picks the final order from its candidates.
    wanted = max(request.top_k, CONFIG.rerank_candidates) if reranker else request.top_k
    hits = store.search(
        vector,
        top_k=wanted,
        language_filter=request.language_filter,
        path_prefix=request.path_prefix,
    )
    if reranker:
        hits = reranker.rerank(request.query, hits, request.top_k)

    results = []
    for hit in hits:
        text = hit["text"]
        if truncate and len(text) > CONFIG.snippet_chars:
            text = text[: CONFIG.snippet_chars].rstrip() + "\n..."
        entry = dict(hit)
        entry["text"] = text
        entry["location"] = f"{hit['path']}:{hit['start_line']}"
        results.append(entry)

    return JSONResponse(
        content={"query": request.query, "count": len(results), "results": results}
    )


@app.post("/search")
def search(request: SearchRequest):
    return _do_search(request, truncate=True)


@app.post("/search/full")
def search_full(request: SearchRequest):
    return _do_search(request, truncate=False)


class ContextRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    language_filter: str | None = None
    path_prefix: str | None = None
    # Character budget for the assembled block. Whole chunks are dropped from
    # the bottom to fit; a chunk is never cut in half, because a generator
    # handed a sentence that stops mid-clause will finish it from imagination.
    max_chars: int = Field(default=8000, ge=500, le=100_000)


@app.post("/context")
def context(request: ContextRequest):
    """Retrieval formatted for a generator's prompt.

    This endpoint runs no language model and needs none installed - it is the
    same search as /search/full, with the hits assembled into a numbered,
    citable block. Whatever writes the actual answer lives outside this
    service.
    """
    found = _do_search(
        SearchRequest(
            query=request.query,
            top_k=request.top_k,
            language_filter=request.language_filter,
            path_prefix=request.path_prefix,
        ),
        truncate=False,
    )
    if found.status_code != 200:
        return found

    hits = json.loads(bytes(found.body)).get("results", [])

    blocks: list[str] = []
    sources: list[dict] = []
    used = 0
    dropped = 0
    for index, hit in enumerate(hits, start=1):
        citation = f"{hit['path']}:{hit['start_line']}-{hit['end_line']}"
        block = f"[{index}] {citation}\n{hit['text']}"
        if used + len(block) > request.max_chars and blocks:
            dropped = len(hits) - len(blocks)
            break
        blocks.append(block)
        used += len(block) + 2
        sources.append(
            {
                "n": index,
                "path": hit["path"],
                "start_line": hit["start_line"],
                "end_line": hit["end_line"],
                "score": hit["score"],
                "citation": citation,
            }
        )

    return {
        "query": request.query,
        "context": "\n\n".join(blocks),
        "sources": sources,
        "chunks_used": len(blocks),
        # Reported rather than silent: a caller that always wanted 8 chunks
        # should be able to see that it got 5 and why.
        "chunks_dropped_for_budget": dropped,
        "chars": used,
    }


class ReindexRequest(BaseModel):
    full: bool = False


@app.post("/reindex")
def reindex(request: ReindexRequest):
    if STATE["ingest_running"]:
        return JSONResponse(
            status_code=409, content={"error": "an ingest is already running"}
        )

    def worker() -> None:
        STATE["ingest_running"] = True
        STATE["ingest_error"] = None
        try:
            run_ingest(CONFIG, full=request.full, store=store, embedder=embedder)
            STATE["last_ingest"] = time.time()
        except Exception as exc:
            STATE["ingest_error"] = str(exc)
            log.error("reindex failed: %s", exc)
        finally:
            STATE["ingest_running"] = False

    threading.Thread(target=worker, name="rag-reindex", daemon=True).start()
    return {"started": True, "full": request.full}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=CONFIG.host, port=CONFIG.port, log_level="info")


if __name__ == "__main__":
    main()
