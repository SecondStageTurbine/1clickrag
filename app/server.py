# SPDX-License-Identifier: MPL-2.0
"""HTTP API + browser UI for the RAG index.

Endpoints (the JSON contract matches what this project's coders already call):

    GET  /                -> browser search UI
    GET  /health          -> {"status":"healthy","ollama":true,"qdrant":true,...}
    GET  /stats           -> index size, model, repo, last ingest
    POST /search          -> truncated snippets
    POST /search/full     -> full chunk text
    POST /context         -> hits assembled into a citable block for a generator
    GET  /chat/config     -> which generator, if any, is configured
    POST /chat            -> a cited answer, streamed (needs RAG_CHAT_PROVIDER)
    POST /reindex         -> {"full": bool} kick off a background re-ingest
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import classify as classify_module
from . import corpus as corpus_module
from . import sheet as sheet_module
from . import expand as expand_module
from . import manifest as manifest_module
from .config import CONFIG, corpus_warning
from .embedder import make_embedder
from .graph import open_graph
from .ingest import remove_paths, run_ingest
from .llm import LlmError, discover, make_provider
from .queue import WorkQueue
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
# Keyword index + entity graph. None when disabled or unopenable, and every
# use is guarded: this is an addition to vector search, never a dependency of it.
graph = open_graph(CONFIG)
# Durable record of work the watcher has seen. Opened eagerly so a restart
# reports what it inherited rather than discovering it later.
queue = WorkQueue(CONFIG.queue_path, max_attempts=CONFIG.queue_max_attempts)
# The chat pane's generator, or None when none is configured - which is the
# default. Search does not depend on it in any way.
llm = make_provider(CONFIG)

# Every generator this machine can reach, refreshed on a timer rather than per
# request: discovery costs an HTTP call to Ollama and two PATH lookups, which
# is nothing once and rude on every keystroke. Short enough that pulling a
# model shows up without a restart.
_BACKEND_TTL = 30.0
_backends: dict = {"at": 0.0, "list": []}
_backend_lock = threading.Lock()


def backends(refresh: bool = False) -> list:
    with _backend_lock:
        stale = time.time() - _backends["at"] > _BACKEND_TTL
        if refresh or stale or not _backends["list"]:
            try:
                _backends["list"] = discover(CONFIG)
                _backends["at"] = time.time()
            except Exception as exc:  # discovery must never break the pane
                log.warning("backend discovery failed: %s", exc)
        return _backends["list"]


def pick_backend(wanted: str | None):
    """The backend the browser asked for, or the sensible default."""
    available = backends()
    if wanted:
        for provider in available:
            if provider.id == wanted:
                return provider
        # Asked for something that has gone away - a model unloaded, a CLI
        # uninstalled. Rediscover once before deciding it is really absent.
        for provider in backends(refresh=True):
            if provider.id == wanted:
                return provider
        return None
    # No explicit choice: the configured provider if there is one, else
    # whatever was discovered first. Taken from the discovered list rather
    # than the module-level provider so it carries an id the browser can send
    # back - the bare one from make_provider has never been given one.
    for provider in available:
        if provider.id == "configured":
            return provider
    return available[0] if available else None

STATE: dict = {
    "ingest_running": False,
    "ingest_error": None,
    "last_ingest": None,
    "last_reconcile": None,
    "started_at": time.time(),
    "warning": None,
    "drift": None,
    "drift_message": None,
}


def check_manifest() -> None:
    """Compare the running settings against what built the index."""
    try:
        recorded = manifest_module.load(CONFIG.manifest_path)
        dim = embedder.dim if getattr(embedder, "dim", None) else None
        result = manifest_module.drift(recorded, manifest_module.current(CONFIG, dim))
        STATE["drift"] = result
        STATE["drift_message"] = manifest_module.describe(result)
        if STATE["drift_message"]:
            log.warning("%s", STATE["drift_message"])
    except Exception as exc:  # bookkeeping must never block a start
        log.debug("manifest check failed: %s", exc)


def start_queue_worker(cfg) -> threading.Thread:
    """Drain the durable queue: index what is due, retry what fails.

    Runs one batch at a time and reports per-path outcomes back to the queue,
    so a single unreadable file backs off on its own without holding up the
    twenty good ones queued behind it.
    """

    def loop() -> None:
        while True:
            time.sleep(cfg.queue_poll_seconds)
            try:
                batch = queue.claim(cfg.queue_batch)
                if not batch:
                    continue
                if STATE["ingest_running"]:
                    continue  # a reindex or rescan holds the lock; try later

                deletes = [item["path"] for item in batch if item["op"] == "delete"]
                indexes = [item["path"] for item in batch if item["op"] != "delete"]

                STATE["ingest_running"] = True
                try:
                    if deletes:
                        try:
                            remove_paths(cfg, deletes, store=store)
                            queue.complete(deletes)
                        except Exception as exc:
                            for path in deletes:
                                queue.fail(path, str(exc))

                    if indexes:
                        try:
                            stats = run_ingest(
                                cfg, full=False, paths=indexes,
                                store=store, embedder=embedder,
                            )
                            failed = stats.failures
                            queue.complete([p for p in indexes if p not in failed])
                            for path, error in failed.items():
                                queue.fail(path, error)
                            if stats.files_indexed:
                                STATE["last_ingest"] = time.time()
                        except Exception as exc:
                            # The whole batch fell over - the store is down, say.
                            # Blame every path in it; they retry individually.
                            for path in indexes:
                                queue.fail(path, str(exc))
                finally:
                    STATE["ingest_running"] = False
            except Exception as exc:
                # The worker must outlive anything it processes.
                log.warning("queue worker error: %s", exc)

    thread = threading.Thread(target=loop, name="rag-queue", daemon=True)
    thread.start()
    counts = queue.counts()
    if counts["pending"]:
        log.info("queue worker started - %d item(s) carried over", counts["pending"])
    else:
        log.info("queue worker started")
    return thread


def start_rescan(cfg) -> threading.Thread:
    """Re-walk the corpus every `rescan_minutes`, forever.

    Belt to the watcher's braces, and the only thing that works at all on a
    network share: SMB does not deliver change notifications dependably, so
    RAG_WATCH defaults off there and this is what keeps the index current.
    Cheap by design - unchanged files are skipped on modification time, so a
    quiet corpus costs a directory walk and no embedding.
    """

    def loop() -> None:
        interval = cfg.rescan_minutes * 60
        while True:
            time.sleep(interval)
            if STATE["ingest_running"]:
                continue  # a reindex or the watcher already has the lock
            STATE["ingest_running"] = True
            try:
                stats = run_ingest(cfg, full=False, store=store, embedder=embedder)
                STATE["last_ingest"] = time.time()
                STATE["last_reconcile"] = stats.as_dict()
                if stats.files_indexed or stats.files_removed:
                    log.info(
                        "rescan: %d changed, %d removed",
                        stats.files_indexed,
                        stats.files_removed,
                    )
            except Exception as exc:
                # Never let a failed sweep kill the loop: a share that is down
                # now is usually back in fifteen minutes.
                log.warning("rescan failed: %s", exc)
            finally:
                STATE["ingest_running"] = False

    thread = threading.Thread(target=loop, name="rag-rescan", daemon=True)
    thread.start()
    log.info("rescanning %s every %d minute(s)", cfg.repo_path, cfg.rescan_minutes)
    return thread


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    language_filter: str | None = None
    path_prefix: str | None = None
    # None means "whatever the server is configured for". Both are no-ops when
    # the sidecar is disabled or empty, so an existing caller's body still
    # means exactly what it meant before.
    hybrid: bool | None = None
    hops: int | None = Field(default=None, ge=0, le=3)
    # Search again with the document's own vocabulary when the question's
    # wording finds little. None uses the server default.
    expand_query: bool | None = None


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
        elif CONFIG.reconcile_on_start:
            # Catch up on whatever happened while this was not running. The
            # watcher cannot see any of it, and unchanged files are skipped by
            # modification time, so the usual cost of this is one directory
            # walk and no embedding at all.
            if CONFIG.auto_pull_model:
                embedder.prepare()
            store.ensure_collection(embedder.dim)
            log.info("index holds %d chunks - reconciling against the corpus", store.count())
            stats = run_ingest(CONFIG, full=False, store=store, embedder=embedder)
            STATE["last_ingest"] = time.time()
            STATE["last_reconcile"] = stats.as_dict()
        else:
            if CONFIG.auto_pull_model:
                embedder.prepare()
            store.ensure_collection(embedder.dim)
            log.info("index already holds %d chunks - skipping ingest", store.count())

        if reranker:
            # Load it now rather than on the first query, which would otherwise
            # pay the download and the model load while a user waits.
            reranker.prepare()

        # After ingest, so a first build has already written its manifest and a
        # brand new index never reports drift against itself.
        check_manifest()

        start_queue_worker(CONFIG)
        if CONFIG.watch:
            start_watcher(CONFIG, store=store, embedder=embedder, queue=queue)
        if CONFIG.rescan_minutes > 0:
            start_rescan(CONFIG)
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
        # What onnxruntime actually chose, not what was asked for. Asking for
        # CUDA and silently getting CPU is the normal failure - a missing
        # driver library logs a warning and falls back - and without this
        # reported, a benchmark shows a 1.1x "speedup" that is CPU both times.
        "embed_provider": getattr(embedder, "provider", None),
        "rerank": bool(reranker),
        "rerank_model": CONFIG.rerank_model if reranker else None,
        "qdrant": qdrant_ok,
        "vector_store": "embedded" if store.embedded else "server",
        "graph": bool(graph and graph.has_data()),
        "hybrid": bool(graph and graph.has_data() and CONFIG.hybrid),
        "chunks": chunks,
        "model": CONFIG.embed_model,
        "collection": CONFIG.collection,
        # Which folder these answers come from. /stats has carried this all
        # along, but nothing polls /stats - so the browser could show a healthy
        # index for minutes without ever saying what was in it, and one machine
        # running two collections looked identical to one running the right one.
        "repo": CONFIG.repo_label,
        "repo_path": CONFIG.repo_path,
        "indexing": STATE["ingest_running"],
        "error": STATE["ingest_error"],
        "warning": STATE["warning"],
        # A setting that shapes the vectors has moved since the index was
        # built. Not an error - search still works - but the results are being
        # drawn from chunks made two different ways.
        "stale": STATE["drift_message"],
    }
    # Kept for continuity with the pre-existing Ollama-backed deployment, where
    # callers looked at `.ollama`. Omitted in native mode rather than reported
    # as a meaningless false: `.status` is the field to check.
    if embedder.backend == "ollama":
        payload["ollama"] = embed_ok
    return payload


@app.get("/stats")
def stats():
    graph_stats = None
    if graph is not None:
        try:
            graph_stats = graph.stats()
        except Exception as exc:  # a sidecar problem must not break /stats
            graph_stats = {"error": str(exc)}
    return {
        "service": SERVICE_ID,
        "graph": graph_stats,
        "repo": CONFIG.repo_label,
        "repo_path": CONFIG.repo_path,
        "mode": CONFIG.mode,
        "collection": CONFIG.collection,
        "chunks": store.count(),
        "model": CONFIG.embed_model,
        "embed_backend": embedder.backend,
        "vector_store": "embedded" if store.embedded else "server",
        "watching": CONFIG.watch,
        "rescan_minutes": CONFIG.rescan_minutes,
        "queue": queue.counts(),
        # Whether a rebuild will have to re-read the documents or not. Worth
        # seeing before starting one on a corpus of scans.
        "extract_cache": _cache_stats(),
        # The full detail behind /health's `stale`: which settings moved, and
        # from what to what.
        "manifest": manifest_module.load(CONFIG.manifest_path),
        "drift": STATE["drift"],
        "indexing": STATE["ingest_running"],
        "last_ingest": STATE["last_ingest"],
        "last_reconcile": STATE["last_reconcile"],
        "uptime_seconds": round(time.time() - STATE["started_at"], 1),
        "error": STATE["ingest_error"],
    }


def _cache_stats() -> dict | None:
    """Extraction cache size and hit rate, or None when it is off."""
    from .extract_cache import open_cache

    try:
        cache = open_cache(CONFIG)
        return cache.stats() if cache else None
    except Exception:  # a reporting problem must not break /stats
        return None


def _citation_key(hit: dict) -> str:
    return f"{hit['path']}:{hit['start_line']}-{hit['end_line']}"


def _fuse(rankings: list[tuple[str, list[dict]]], k: int) -> list[dict]:
    """Reciprocal rank fusion of several rankings of the same chunks.

    Ranks are combined, not scores. A cosine similarity, a BM25 value and a
    count of matched entities share no scale and no distribution; normalising
    them against each other would invent a comparison that does not exist,
    whereas "this arm put it third" is meaningful in every arm. A chunk found
    by two arms outranks a chunk any one of them liked slightly more, which is
    the property that makes hybrid retrieval work.
    """
    scores: dict[str, float] = {}
    entries: dict[str, dict] = {}
    origins: dict[str, list[str]] = {}

    for name, hits in rankings:
        for rank, hit in enumerate(hits, start=1):
            key = _citation_key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            origins.setdefault(key, []).append(name)
            if key not in entries:
                entries[key] = hit

    ordered = []
    for key in sorted(scores, key=lambda item: -scores[item]):
        entry = dict(entries[key])
        entry["score"] = round(scores[key], 6)
        entry["origins"] = origins[key]
        ordered.append(entry)

    # Each arm drops its own overlapping chunks, but chunking overlaps on
    # purpose, so two arms can return lines 62-68 and 63-68 of one file and the
    # fusion sees two distinct keys. Left alone that spends the answer's budget
    # showing the same paragraph twice.
    fused: list[dict] = []
    kept_spans: dict[str, list[tuple[int, int]]] = {}
    for entry in ordered:
        spans = kept_spans.setdefault(entry["path"], [])
        start = int(entry.get("start_line", 0) or 0)
        end = int(entry.get("end_line", 0) or 0)
        if any(start <= kept_end and end >= kept_start for kept_start, kept_end in spans):
            continue
        spans.append((start, end))
        fused.append(entry)
    return fused


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

    rankings: list[tuple[str, list[dict]]] = [("vector", hits)]
    graph_used: dict | None = None
    expanded_with: list[str] = []
    use_hybrid = CONFIG.hybrid if request.hybrid is None else request.hybrid
    hops = CONFIG.graph_hops if request.hops is None else request.hops

    if graph is not None and graph.has_data():
        if use_hybrid:
            rankings.append(
                (
                    "keyword",
                    graph.keyword_search(
                        request.query,
                        top_k=wanted,
                        language_filter=request.language_filter,
                        path_prefix=request.path_prefix,
                    ),
                )
            )

            # Two more keyword rankings, for the case where the question and
            # the passage that answers it share no wording. See app/expand.py:
            # the first strips the question's scaffolding, the second takes
            # the vocabulary the first pass turned up and searches with that.
            expand = CONFIG.expand if request.expand_query is None else request.expand_query
            if expand:
                terms = expand_module.content_terms(request.query)
                if terms and len(terms) < len(request.query.split()):
                    rankings.append(
                        (
                            "terms",
                            graph.keyword_search(
                                " ".join(terms),
                                top_k=wanted,
                                language_filter=request.language_filter,
                                path_prefix=request.path_prefix,
                            ),
                        )
                    )

                # Fed from what the first arms actually found - the document's
                # own words rather than the asker's. Cheap: one more BM25 pass
                # over SQLite, no model and no embedding.
                seen_text = [
                    hit["text"]
                    for _, arm in rankings
                    for hit in arm[: CONFIG.expand_feedback_docs]
                ]
                follow_up = expand_module.feedback_terms(
                    seen_text,
                    exclude=terms,
                    limit=CONFIG.expand_terms,
                    frequency=graph.chunk_frequency,
                    corpus_chunks=graph.count_chunks(),
                    max_share=CONFIG.expand_max_share,
                )
                if follow_up:
                    expanded_with = follow_up
                    rankings.append(
                        (
                            "feedback",
                            graph.keyword_search(
                                " ".join(follow_up),
                                top_k=wanted,
                                language_filter=request.language_filter,
                                path_prefix=request.path_prefix,
                            ),
                        )
                    )
        if hops > 0:
            # Multi-hop: resolve the names in the question to entities, walk
            # out `hops` steps, and add every chunk mentioning anything reached.
            # This arm is recall, deliberately wide - the fusion and the
            # reranker are what keep the width from becoming noise.
            seeds = graph.resolve_entities(request.query)
            reached = graph.expand(
                [seed["id"] for seed in seeds], hops, CONFIG.graph_fanout
            )
            rankings.append(
                (
                    "graph",
                    graph.chunks_for_entities(
                        [entity["id"] for entity in reached],
                        top_k=wanted,
                        language_filter=request.language_filter,
                        path_prefix=request.path_prefix,
                    ),
                )
            )
            graph_used = {
                "hops": hops,
                "seeds": [
                    {"name": seed["name"], "kind": seed["kind"]} for seed in seeds
                ],
                "reached": [
                    {"name": entity["name"], "hop": entity["hop"]}
                    for entity in reached
                    if entity["hop"] > 0
                ][:20],
            }

    # One arm and no reranker is the original path, byte for byte: the score
    # stays the cosine similarity a caller may already be reading.
    if len(rankings) > 1:
        hits = _fuse(rankings, CONFIG.rrf_k)

    if reranker:
        hits = reranker.rerank(request.query, hits, request.top_k)
    else:
        hits = hits[: request.top_k]

    results = []
    for hit in hits:
        text = hit["text"]
        if truncate and len(text) > CONFIG.snippet_chars:
            text = text[: CONFIG.snippet_chars].rstrip() + "\n..."
        entry = dict(hit)
        entry["text"] = text
        entry["location"] = f"{hit['path']}:{hit['start_line']}"
        results.append(entry)

    payload = {
        "query": request.query,
        "count": len(results),
        # What produced the ordering, so `score` is interpretable: a cosine
        # similarity, a fused rank, and a cross-encoder logit are different
        # numbers and only the caller can know which one it is looking at.
        "ranking": [name for name, _ in rankings] + (["rerank"] if reranker else []),
        "results": results,
    }
    if expanded_with:
        # Shown rather than kept internal: when an answer comes from a passage
        # sharing no words with the question, "how did it get there" is the
        # first thing worth being able to check.
        payload["expanded_with"] = expanded_with
    if graph_used:
        payload["graph"] = graph_used
    return JSONResponse(content=payload)


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
    hybrid: bool | None = None
    hops: int | None = Field(default=None, ge=0, le=3)
    # Widen each hit to the rest of its section before assembling. None uses
    # the server default.
    expand: bool | None = None
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
            hybrid=request.hybrid,
            hops=request.hops,
        ),
        truncate=False,
    )
    if found.status_code != 200:
        return found

    body = json.loads(bytes(found.body))
    hits = body.get("results", [])

    expand = CONFIG.context_expand if request.expand is None else request.expand

    # Expansion first, numbering second. Two hits from one section widen to the
    # same passage, and numbering as we go would hand the generator the same
    # text twice under two citations - paying for it twice out of the budget,
    # and inviting an answer that cites [2] and [3] as if they corroborated
    # each other.
    prepared: list[dict] = []
    seen_citations: set[str] = set()
    for hit in hits:
        text = hit["text"]
        start_line = hit["start_line"]
        end_line = hit["end_line"]
        expanded = False

        if expand and hit.get("section"):
            # Rebuild the run of chunks around this one that share its heading.
            # Only the contiguous run: a heading repeated later in the file is
            # a different section with the same name, and splicing the two
            # would invent a passage that does not exist.
            section = store.section_chunks(hit["path"], hit["section"])
            run: list[dict] = []
            for chunk in section:
                if run and chunk["start_line"] > run[-1]["end_line"] + 1:
                    if run[0]["start_line"] <= start_line <= run[-1]["end_line"]:
                        break
                    run = []
                run.append(chunk)
            if run and run[0]["start_line"] <= start_line <= run[-1]["end_line"] and len(run) > 1:
                merged = "\n".join(chunk["text"] for chunk in run)
                # Never let one expanded section eat the whole budget.
                if len(merged) <= max(request.max_chars // 2, len(text)):
                    text = merged
                    start_line = run[0]["start_line"]
                    end_line = run[-1]["end_line"]
                    expanded = True

        citation = f"{hit['path']}:{start_line}-{end_line}"
        if citation in seen_citations:
            continue
        seen_citations.add(citation)

        # The heading rides on the citation line: it tells a reader (and a
        # generator writing "according to the Deployment section") where in the
        # document this sits, which line numbers alone never do.
        heading = (hit.get("section") or hit.get("symbol") or "").strip().splitlines()[0:1]
        prepared.append(
            {
                "citation": citation,
                "text": text,
                "heading": heading[0][:120] if heading and heading[0] else "",
                "expanded": expanded,
                "path": hit["path"],
                "start_line": start_line,
                "end_line": end_line,
                "score": hit["score"],
            }
        )

    blocks: list[str] = []
    sources: list[dict] = []
    used = 0
    dropped = 0
    for index, item in enumerate(prepared, start=1):
        label = f" ({item['heading']})" if item["heading"] else ""
        block = f"[{index}] {item['citation']}{label}\n{item['text']}"
        if used + len(block) > request.max_chars and blocks:
            dropped = len(prepared) - len(blocks)
            break
        blocks.append(block)
        used += len(block) + 2
        sources.append(
            {
                "n": index,
                "path": item["path"],
                "start_line": item["start_line"],
                "end_line": item["end_line"],
                "score": item["score"],
                "citation": item["citation"],
                "heading": item["heading"],
                "expanded": item["expanded"],
            }
        )

    return {
        "query": request.query,
        "context": "\n\n".join(blocks),
        "sources": sources,
        "ranking": body.get("ranking", []),
        "graph": body.get("graph"),
        "chunks_used": len(blocks),
        # Reported rather than silent: a caller that always wanted 8 chunks
        # should be able to see that it got 5 and why.
        "chunks_dropped_for_budget": dropped,
        "chars": used,
    }


SYSTEM_PROMPT = """You answer questions about a private collection of documents.

The numbered passages attached to each question are the retrieved evidence, and
the only view you have of the collection. Answer from them, and cite the passage
each claim rests on as [1], [2] - a reader checking your answer has the citation
and nothing else.

Say plainly when the passages do not answer the question. A retrieval miss is a
normal outcome and a useful one to report; an answer assembled from general
knowledge reads exactly like a correct one, which is what makes it costly. When
the passages answer part of the question, give that part and name what is
missing.

Quote exact strings - identifiers, part numbers, dates, names - rather than
paraphrasing them, and keep the answer as long as the question needs."""


SEARCH_PROTOCOL = """

One more thing about the search. The passages were found by matching your
question's wording against the documents, so when the two use different words
for the same thing, the right passage can be missing entirely - a manual that
says "torque specification" does not match a question about "how tight", and a
page lettered DAILY QUEST does not match a question about "the System". The
passages being wrong is therefore ordinary, and recoverable.

When they do not contain the answer, and you can guess what the documents
themselves would call it, reply with exactly:

SEARCH: <words likely printed in the document>

on one line, with nothing before or after it. A fresh search runs and you get
new passages. Use the document's likely vocabulary rather than the question's,
and prefer distinctive nouns to a rephrased question. If the passages do answer
the question, ignore all of this and just answer.

Separately: a question about *how many* documents there are cannot be answered
from passages at all, however good they are. Search returns the few passages
most like the question; counting needs every matching document and none of
their text. For "how many X are there", "how many were signed this year", "what
is in folder Y", reply with exactly:

COUNT: <words that appear in those documents' filenames>

on one line. The count is computed over the index and given back to you exactly,
broken down by year - do not estimate it, and do not count anything yourself.
The words should be what the *filenames* contain, so a set of files named
"Signed - A.CISAR.20260225.pdf" is found by COUNT: CISAR."""

LAST_ROUND = """

These passages are what the search found, including any follow-up searches you
asked for. Answer from them now. If they still do not contain the answer, say
so plainly and say what they do cover - that is a useful reply, and better than
a guess dressed up as one."""

# A directive is one short line and nothing else. Anything longer is an answer
# that happens to discuss searching, and treating that as a command would eat
# the response - so the length cap matters as much as the prefix.
_DIRECTIVE_LINE = re.compile(
    r"^\s*(?:>|\*|-)?\s*(SEARCH|COUNT)\s*:\s*(.+?)\s*$", re.IGNORECASE
)


def _directive(text: str) -> tuple[str, str] | None:
    """(kind, argument) the model is asking for, or None if this is an answer."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    if len(first) > 160:
        return None
    match = _DIRECTIVE_LINE.match(first)
    if not match:
        return None
    wanted = match.group(2).strip().strip('"').strip()
    # A bare "SEARCH:" is not a query, and a whole sentence is the model
    # narrating rather than asking.
    if not 2 <= len(wanted) <= 120:
        return None
    return match.group(1).upper(), wanted


def _merge_context(block_a: str, sources_a: list[dict],
                   block_b: str, sources_b: list[dict],
                   max_chars: int) -> tuple[str, list[dict]]:
    """Both rounds' passages as one numbered block.

    Renumbered from one rather than appended, because the model cites by
    number and two independently numbered lists would make [2] ambiguous -
    and the browser resolves those numbers against the list it was sent.
    """
    merged: list[dict] = []
    seen: set[str] = set()
    for source in list(sources_a) + list(sources_b):
        if source["citation"] in seen:
            continue
        seen.add(source["citation"])
        merged.append(dict(source))

    by_citation: dict[str, str] = {}
    for block in (block_a, block_b):
        for piece in block.split("\n\n["):
            piece = piece if piece.startswith("[") else "[" + piece
            head = piece.split("\n", 1)
            if len(head) != 2:
                continue
            label = head[0].split("] ", 1)
            if len(label) != 2:
                continue
            citation = label[1].split(" (")[0].strip()
            by_citation.setdefault(citation, head[1])

    parts: list[str] = []
    kept: list[dict] = []
    used = 0
    for source in merged:
        body = by_citation.get(source["citation"])
        if body is None:
            continue
        heading = f" ({source['heading']})" if source.get("heading") else ""
        number = len(kept) + 1
        text = f"[{number}] {source['citation']}{heading}\n{body}"
        if used + len(text) > max_chars and parts:
            break
        parts.append(text)
        source["n"] = number
        kept.append(source)
        used += len(text) + 2
    return "\n\n".join(parts), kept


def _retrieval_query(messages: list[dict]) -> str:
    """What to search for, given the conversation so far.

    The last user message, except when it is short enough to be a follow-up
    ("what about the other one?", "why?"), which on its own retrieves nothing
    useful. Those carry their subject in the previous turn, so it is prepended.
    A crude rule, but it costs no model call and is easy to reason about when
    the citations look wrong.
    """
    users = [m["content"].strip() for m in messages if m["role"] == "user"]
    if not users:
        return ""
    latest = users[-1]
    if len(latest) < 80 and len(users) > 1:
        return f"{users[-2]}\n{latest}"
    return latest


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    # Whole conversation, oldest first. The server holds no session state: a
    # reload starts a new conversation, and two browsers cannot tread on each
    # other's history.
    messages: list[ChatMessage]
    top_k: int | None = Field(default=None, ge=1, le=50)
    language_filter: str | None = None
    path_prefix: str | None = None
    hybrid: bool | None = None
    hops: int | None = Field(default=None, ge=0, le=3)
    expand: bool | None = None
    # Off asks the generator without retrieval, for a follow-up about the
    # answer itself ("shorten that", "what does that acronym mean") where a
    # fresh search would only crowd the prompt.
    retrieve: bool = True
    # Which generator answers, by the id from GET /chat/models. None takes the
    # default, so a caller written before this existed still works.
    backend: str | None = None


@app.get("/chat/models")
def chat_models():
    """What the dropdown offers, discovered fresh rather than configured.

    A list in a config file is wrong the moment someone pulls a new model.
    Asking Ollama what it holds and looking for the CLIs on PATH is right by
    construction: `ollama pull deepseek-r1` and it is here on the next load.
    """
    available = backends()
    chosen = pick_backend(None)
    return {
        "default": chosen.id if chosen else None,
        "backends": [
            {
                "id": provider.id,
                "label": provider.label or provider.model,
                "model": provider.model,
                "kind": provider.name,
                "endpoint": provider.endpoint,
                "local": provider.local,
            }
            for provider in available
        ],
    }


@app.get("/chat/config")
def chat_config():
    """Whether chat is available, and where the documents would go.

    The destination is reported rather than assumed: this indexes private
    material, and "which host am I about to send it to" is the question anyone
    running it should be able to answer before typing.
    """
    chosen = pick_backend(None)
    if chosen is None:
        return {
            "enabled": False,
            "detail": (
                "no generator found - install Ollama and pull a model, or set "
                "RAG_CHAT_PROVIDER in rag/.env"
            ),
        }
    return {
        "enabled": True,
        "provider": chosen.name,
        "model": chosen.model,
        "endpoint": chosen.endpoint,
        "local": chosen.local,
        "backend": chosen.id,
        "count": len(backends()),
        # A reachability check, not a promise: the model can still be missing
        # or the key rejected at the moment of asking.
        "problem": chosen.check(),
    }


@app.post("/chat")
def chat(request: ChatRequest):
    """A cited answer, streamed as server-sent events.

    Retrieval happens here and the passages are attached to the newest user
    message, so the system prompt stays byte-identical between turns and the
    evidence sits next to the question it answers.

    Events: `sources` once, then `delta` per fragment, then `done`; `error`
    replaces the rest if something fails. Errors arrive on the stream rather
    than as a status code because the failure usually happens after the
    response has started, when the status is already 200.
    """
    generator = pick_backend(request.backend)
    if generator is None:
        if request.backend:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"no generator called '{request.backend}'",
                    "detail": "GET /chat/models lists what is available now",
                },
            )
        return JSONResponse(
            status_code=501,
            content={
                "error": "chat is not configured",
                "detail": (
                    "no generator found - install Ollama and pull a model, or "
                    "set RAG_CHAT_PROVIDER in rag/.env. Search works without it"
                ),
            },
        )

    history = [
        {"role": m.role, "content": m.content}
        for m in request.messages
        if m.role in ("user", "assistant") and m.content.strip()
    ]
    if not history or history[-1]["role"] != "user":
        return JSONResponse(
            status_code=400,
            content={"error": "the last message must be from the user"},
        )

    def retrieve(query: str):
        """One retrieval pass. Returns (block, sources, ranking, graph)."""
        found = context(
            ContextRequest(
                query=query,
                top_k=request.top_k or CONFIG.chat_top_k,
                language_filter=request.language_filter,
                path_prefix=request.path_prefix,
                hybrid=request.hybrid,
                hops=request.hops,
                expand=request.expand,
                max_chars=CONFIG.chat_context_chars,
            )
        )
        if isinstance(found, JSONResponse):
            return found
        return (found["context"], found["sources"],
                found.get("ranking", []), found.get("graph"))

    # Retrieve before opening the stream: a retrieval failure should be an
    # ordinary status code, not an error event inside a 200.
    context_block = ""
    sources: list[dict] = []
    ranking: list[str] = []
    graph_used = None
    if request.retrieve:
        first = retrieve(_retrieval_query(history))
        if isinstance(first, JSONResponse):
            return first
        context_block, sources, ranking, graph_used = first

    # Older turns are replayed for pronouns and follow-ups, but their
    # passages are not: re-sending every turn's evidence would fill the
    # context window with retrieval nobody asked about again.
    base_turns = history[-(CONFIG.chat_history_turns + 1):]
    question = base_turns[-1]["content"]

    # Facts computed about the corpus rather than retrieved from it - counts,
    # so far. Carried separately from the passages because they are not
    # passages: nothing was matched to produce them and there is nothing to
    # cite but the query itself.
    facts: list[str] = []

    def with_passages(block: str) -> list[dict]:
        """The conversation with this round's evidence on the newest turn."""
        preamble = ("\n\n".join(facts) + "\n\n") if facts else ""
        if block:
            body = f"{question}\n\n{preamble}Retrieved passages:\n\n{block}"
        elif facts:
            body = (
                f"{question}\n\n{preamble}"
                "There are no retrieved passages - answer from the counts above."
            )
        elif request.retrieve:
            body = (
                f"{question}\n\n"
                "Retrieved passages: none - the search returned nothing for "
                "this question. Say so rather than answering from general "
                "knowledge."
            )
        else:
            body = question
        return base_turns[:-1] + [{"role": "user", "content": body}]

    def event(name: str, payload: dict) -> str:
        # json.dumps also does the SSE framing work: it escapes the newlines
        # that would otherwise end the data field early.
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    def stream():
        nonlocal context_block, sources, ranking, graph_used

        yield event("sources", {
            "sources": sources,
            "ranking": ranking,
            "graph": graph_used,
            "provider": generator.name,
            "model": generator.model,
            "backend": generator.id,
        })

        # The model may answer, or it may ask for a different search. It can
        # only ask a bounded number of times: a model that keeps searching
        # rather than answering would otherwise never stop, and each round
        # costs a full generation.
        searches_left = CONFIG.chat_max_searches if request.retrieve else 0
        turns = with_passages(context_block)
        tried: list[str] = []
        exhausted = False

        while True:
            # Recomputed every round, and that is the point: once the budget is
            # spent the protocol has to come *out* of the prompt. Leaving it in
            # invites another directive that can no longer be honoured, and an
            # unhonoured directive is streamed to the reader as the answer -
            # which is exactly what "SEARCH: YOU HAVE MET THE QUALIFICATIONS"
            # appearing in the pane looked like.
            prompt = SYSTEM_PROMPT + (SEARCH_PROTOCOL if searches_left else LAST_ROUND)
            directive, opening, rest = None, "", None
            try:
                # Read just enough to tell an answer from a search request.
                # A directive is short and comes first, so a line's worth is
                # always enough - and buffering only that keeps the answer's
                # streaming intact in the ordinary case.
                head = generator.stream(prompt, turns)
                buffer = ""
                for fragment in head:
                    if isinstance(fragment, dict):
                        yield event("status", fragment)
                        continue
                    buffer += fragment
                    if "\n" in buffer or len(buffer) > 120:
                        break
                directive = _directive(buffer)
                opening, rest = buffer, head
            except LlmError as exc:
                yield event("error", {"error": str(exc)})
                return
            except Exception as exc:
                log.exception("chat failed")
                yield event("error", {"error": f"{type(exc).__name__}: {exc}"})
                return

            # A directive with no budget left is not an answer either. Drop it
            # and generate once more without the protocol, rather than showing
            # the reader a command meant for the server.
            if directive and searches_left <= 0:
                if not exhausted:
                    exhausted = True
                    turns = with_passages(context_block)
                    continue
                directive = None

            # A count is not a search: it needs every matching document and
            # none of their text, so it is computed rather than retrieved, and
            # handed back as a fact alongside whatever passages there are.
            if directive and directive[0] == "COUNT" and searches_left > 0 \
                    and directive not in tried:
                searches_left -= 1
                tried.append(directive)
                wanted = directive[1]
                yield event("status", {"counting": wanted})
                try:
                    tally = corpus_module.count(
                        store.indexed_state(), match=wanted, group_by="year"
                    )
                    facts.append(corpus_module.describe(tally))
                    yield event("count", tally)
                except Exception as exc:
                    log.warning("count failed: %s", exc)
                    facts.append(
                        f'The corpus count for "{wanted}" could not be computed: {exc}'
                    )
                turns = with_passages(context_block)
                continue

            if directive and searches_left > 0 and directive not in tried:
                searches_left -= 1
                tried.append(directive)
                directive = directive[1]
                yield event("status", {"searching": directive})
                again = retrieve(directive)
                if isinstance(again, JSONResponse):
                    break  # index went away mid-answer; answer from what we had
                block, more, ranking, graph_used = again
                # Both rounds' passages, renumbered once, so a citation in the
                # final answer still points at something in the list the
                # browser was given.
                context_block, sources = _merge_context(
                    context_block, sources, block, more, CONFIG.chat_context_chars
                )
                yield event("sources", {
                    "sources": sources, "ranking": ranking, "graph": graph_used,
                    "provider": generator.name, "model": generator.model,
                    "backend": generator.id,
                })
                turns = with_passages(context_block)
                continue

            # An answer. Flush what was buffered, then stream the remainder.
            if opening:
                yield event("delta", {"text": opening})
            try:
                for fragment in rest:
                    # A provider may report progress instead of answer text - a
                    # reasoning model can think for minutes before its first
                    # word, and silence that long looks like a hang.
                    if isinstance(fragment, dict):
                        yield event("status", fragment)
                    elif fragment:
                        yield event("delta", {"text": fragment})
            except LlmError as exc:
                yield event("error", {"error": str(exc)})
                return
            except Exception as exc:  # a bug must not look like a hung stream
                log.exception("chat failed")
                yield event("error", {"error": f"{type(exc).__name__}: {exc}"})
                return
            break

        yield event("done", {})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Nothing proxies this today, but a buffering proxy is exactly
            # what turns a streamed answer back into a long silence.
            "X-Accel-Buffering": "no",
        },
    )


class CountRequest(BaseModel):
    # A substring of the path, or a glob if it carries wildcards. Empty counts
    # everything indexed.
    match: str | None = None
    path_prefix: str | None = None
    group_by: str = Field(default="year", pattern="^(year|month|folder|extension|none)$")
    examples: int = Field(default=5, ge=0, le=50)


@app.post("/corpus/count")
def corpus_count(request: CountRequest):
    """How many documents match, grouped - not what any of them say.

    Retrieval cannot answer this and no amount of tuning makes it: search
    returns the passages most like a question, while counting needs every
    document that matches and none of their text. Computed here so the number
    is exact; a model asked to tally filenames is guessing.

    Counts documents rather than chunks, because "how many forms" means files
    and one PDF is many chunks.
    """
    if not store.exists():
        return JSONResponse(
            status_code=503,
            content={"error": "nothing is indexed yet"},
        )
    try:
        documents = store.indexed_state()
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    return corpus_module.count(
        documents,
        match=request.match,
        path_prefix=request.path_prefix,
        group_by=request.group_by,
        examples=request.examples,
    )


class ClassifyRequest(BaseModel):
    # The workbook, absolute or relative to the corpus.
    path: str
    # Which columns hold the text the decision is about. More than one is
    # joined; the failure mode plus its effect usually decides more than
    # either alone.
    columns: list[str]
    # What to retrieve as the criteria. Retrieved once for the whole run.
    criteria_query: str
    labels: list[str] = Field(default_factory=lambda: ["applicable", "not applicable"])
    unclear_label: str = "unclear"
    sheet: str | None = None
    id_column: str | None = None
    # A column of known-correct answers, if the sheet has one. Turns a run into
    # a measurement: the reply carries an agreement rate, which is how the
    # batch size gets chosen on evidence rather than hope.
    truth_column: str | None = None
    batch_size: int = Field(default=25, ge=1, le=200)
    # Batches in flight at once. Worth raising only as far as the model server
    # can genuinely run in parallel - several GPUs serving several instances
    # scale nearly linearly, one instance does not, and past that requests
    # simply queue.
    concurrency: int = Field(default=1, ge=1, le=32)
    # Stop after this many rows. Use it before committing to sixteen thousand.
    limit: int = Field(default=0, ge=0)
    criteria_chars: int = Field(default=6000, ge=500, le=40_000)
    backend: str | None = None


CLASSIFY: dict = {
    "running": False, "started_at": None, "finished_at": None,
    "done": 0, "total": 0, "stage": "", "error": None,
    "request": None, "result": None, "cancel": False,
}


@app.post("/classify")
def classify_start(request: ClassifyRequest):
    """Decide one question about every row of a spreadsheet.

    A bulk job living inside a service built for interactive search, so it is
    deliberately one at a time and on a worker thread: search stays answerable
    while sixteen thousand rows are being judged, and two of these cannot fight
    over the same model.
    """
    if CLASSIFY["running"]:
        return JSONResponse(status_code=409, content={"error": "a classification is already running"})

    generator = pick_backend(request.backend)
    if generator is None:
        return JSONResponse(
            status_code=501,
            content={"error": "no generator configured", "detail": "GET /chat/models"},
        )

    path = request.path
    if not os.path.isabs(path):
        path = os.path.join(CONFIG.repo_path, path)
    if not os.path.isfile(path):
        return JSONResponse(status_code=404, content={"error": f"no such file: {path}"})

    try:
        rows, headers = sheet_module.read_rows(path, request.sheet, request.limit)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if not rows:
        return JSONResponse(status_code=400, content={"error": "the sheet has no data rows"})

    missing = [c for c in request.columns if c not in headers]
    if missing:
        return JSONResponse(
            status_code=400,
            content={"error": f"no such column(s): {', '.join(missing)}",
                     "detail": f"available: {', '.join(headers)}"},
        )

    # The criteria, once. This is the change that matters most: every row was
    # retrieving substantially the same passages, and judging every row against
    # identical wording is worth more than the time it saves.
    found = context(ContextRequest(
        query=request.criteria_query,
        top_k=CONFIG.chat_top_k,
        max_chars=request.criteria_chars,
    ))
    if isinstance(found, JSONResponse):
        return found
    criteria = found["context"]
    if not criteria.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "the criteria search found nothing",
                     "detail": f"nothing matched {request.criteria_query!r} - "
                               f"is the defining document indexed?"},
        )

    cases = classify_module.build_cases(rows, request.columns)
    CLASSIFY.update({
        "running": True, "started_at": time.time(), "finished_at": None,
        "done": 0, "total": len(cases), "stage": "classifying", "error": None,
        "cancel": False, "result": None,
        "request": {**request.model_dump(), "resolved_path": path},
    })

    def ask(prompt: str) -> str:
        parts = []
        for fragment in generator.stream("", [{"role": "user", "content": prompt}]):
            if isinstance(fragment, str):
                parts.append(fragment)
        return "".join(parts)

    def worker() -> None:
        try:
            classify_module.classify_cases(
                cases, criteria, request.labels, ask,
                batch_size=request.batch_size,
                unclear=request.unclear_label,
                on_progress=lambda done, total: CLASSIFY.update({"done": done, "total": total}),
                should_stop=lambda: CLASSIFY["cancel"],
                concurrency=request.concurrency,
            )
            CLASSIFY["result"] = classify_module.results(
                rows, cases, request.id_column, request.truth_column
            )
            CLASSIFY["result"]["sources"] = found["sources"]
            CLASSIFY["stage"] = "done"
        except classify_module.Cancelled:
            CLASSIFY["stage"] = "cancelled"
            # Partial work is still work: keep what was decided before the stop.
            CLASSIFY["result"] = classify_module.results(
                rows, cases, request.id_column, request.truth_column
            )
        except Exception as exc:
            log.exception("classification failed")
            CLASSIFY["error"] = f"{type(exc).__name__}: {exc}"
            CLASSIFY["stage"] = "failed"
        finally:
            CLASSIFY["running"] = False
            CLASSIFY["finished_at"] = time.time()

    threading.Thread(target=worker, name="rag-classify", daemon=True).start()
    return {
        "started": True,
        "rows": len(rows),
        "distinct_cases": len(cases),
        # Reported up front because it predicts the runtime better than the row
        # count does, and it is the number people are surprised by.
        "repeats_collapsed": len(rows) - len(cases),
        "criteria_chars": len(criteria),
        "model": generator.model,
    }


@app.get("/classify")
def classify_status(verdicts: bool = False):
    """Progress, and the verdicts once there are any."""
    state = {key: value for key, value in CLASSIFY.items() if key != "result"}
    elapsed = None
    if CLASSIFY["started_at"]:
        end = CLASSIFY["finished_at"] or time.time()
        elapsed = round(end - CLASSIFY["started_at"], 1)
    state["elapsed_seconds"] = elapsed
    if elapsed and CLASSIFY["done"]:
        rate = CLASSIFY["done"] / elapsed
        state["cases_per_second"] = round(rate, 2)
        remaining = max(CLASSIFY["total"] - CLASSIFY["done"], 0)
        state["eta_seconds"] = round(remaining / rate) if rate else None

    result = CLASSIFY["result"]
    if result:
        state["summary"] = {k: v for k, v in result.items() if k != "verdicts"}
        if verdicts:
            state["verdicts"] = result["verdicts"]
    return state


@app.post("/classify/cancel")
def classify_cancel():
    """Stop after the batch in flight, keeping what has been decided."""
    if not CLASSIFY["running"]:
        return {"cancelled": False, "detail": "nothing is running"}
    CLASSIFY["cancel"] = True
    return {"cancelled": True}


@app.get("/classify/csv")
def classify_csv():
    """The verdicts as a CSV, carrying the columns that identify a row."""
    result = CLASSIFY["result"]
    if not result:
        return JSONResponse(status_code=404, content={"error": "no results yet"})
    request = CLASSIFY["request"] or {}
    try:
        rows, _ = sheet_module.read_rows(
            request.get("resolved_path", ""), request.get("sheet"), request.get("limit", 0)
        )
    except ValueError:
        rows = []
    carry = [c for c in ([request.get("id_column")] if request.get("id_column") else [])
             + list(request.get("columns") or []) if c]
    body = sheet_module.to_csv(result["verdicts"], rows, carry)
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="verdicts.csv"'},
    )


class NeighbourRequest(BaseModel):
    entity: str
    hops: int = Field(default=1, ge=1, le=3)
    limit: int = Field(default=12, ge=1, le=100)


class PathRequest(BaseModel):
    # `from` is a Python keyword, so the field is aliased for the JSON body.
    start: str = Field(alias="from")
    end: str = Field(alias="to")
    max_hops: int = Field(default=3, ge=1, le=4)

    model_config = {"populate_by_name": True}


def _require_graph() -> JSONResponse | None:
    if graph is None:
        return JSONResponse(
            status_code=501,
            content={
                "error": "the entity graph is disabled",
                "detail": "set RAG_GRAPH=1 and reindex to build it",
            },
        )
    if not graph.has_data():
        return JSONResponse(
            status_code=503,
            content={
                "error": "the entity graph is empty",
                "detail": "it is built during ingest - run a full reindex to populate it",
            },
        )
    return None


@app.get("/entities")
def entities(
    limit: int = 50,
    kind: str | None = None,
    contains: str | None = None,
    path_prefix: str | None = None,
    include_common: bool = False,
):
    """The names the index knows about, most-mentioned first.

    Boilerplate is hidden by default - anything appearing in more than
    RAG_GRAPH_MAX_DF of the documents connects everything to everything. Pass
    include_common=true to see what is being filtered and why.
    """
    problem = _require_graph()
    if problem:
        return problem
    return {
        "count": graph.count_chunks(),
        "df_ceiling": graph.df_ceiling(),
        "entities": graph.top_entities(
            limit=min(max(limit, 1), 500),
            kind=kind,
            contains=contains,
            path_prefix=path_prefix,
            include_common=include_common,
        ),
    }


@app.post("/graph/neighbors")
def neighbours(request: NeighbourRequest):
    """What shares documents with this name, one or more hops out."""
    problem = _require_graph()
    if problem:
        return problem

    seeds = graph.resolve_entities(request.entity, limit=3)
    if not seeds:
        return JSONResponse(
            status_code=404,
            content={
                "error": f"no entity matching '{request.entity}'",
                "detail": "GET /entities?contains=... to see what is indexed",
            },
        )

    reached = graph.expand(
        [seed["id"] for seed in seeds], request.hops, CONFIG.graph_fanout
    )
    return {
        "entity": seeds[0]["name"],
        "matched": [{"name": seed["name"], "kind": seed["kind"]} for seed in seeds],
        "hops": request.hops,
        "neighbors": [
            {
                "name": entity["name"],
                "kind": entity["kind"],
                "hop": entity["hop"],
                "shared_chunks": entity.get("shared"),
                "documents": entity["doc_count"],
                # The passages the link rests on. An edge here is never an
                # assertion - it is these citations, and they can be checked.
                "evidence": graph.link(seeds[0]["id"], entity["id"])
                if entity["hop"] == 1
                else [],
            }
            for entity in reached
            if entity["hop"] > 0
        ][: request.limit],
    }


@app.post("/graph/path")
def graph_path(request: PathRequest):
    """How two names connect, through the documents that mention both."""
    problem = _require_graph()
    if problem:
        return problem
    return graph.path_between(request.start, request.end, max_hops=request.max_hops)


@app.get("/queue")
def queue_state(limit: int = 50):
    """What is waiting to be indexed, and what has been given up on.

    The dead-letter list is the point: a file that cannot be read is normally
    a log line nobody reads, and this makes it something you can query, fix,
    and retry deliberately.
    """
    return queue.snapshot(limit=min(max(limit, 1), 500))


class QueueRetryRequest(BaseModel):
    path: str | None = None


@app.post("/queue/retry")
def queue_retry(request: QueueRetryRequest):
    """Put dead letters back in the queue - after fixing whatever broke."""
    moved = queue.retry_dead(request.path)
    return {"requeued": moved}


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
            stats = run_ingest(CONFIG, full=request.full, store=store, embedder=embedder)
            STATE["last_ingest"] = time.time()
            STATE["last_reconcile"] = stats.as_dict()
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
