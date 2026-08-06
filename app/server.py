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
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import CONFIG, corpus_warning
from .embedder import make_embedder
from .graph import open_graph
from .ingest import remove_paths, run_ingest
from .llm import LlmError, make_provider
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

STATE: dict = {
    "ingest_running": False,
    "ingest_error": None,
    "last_ingest": None,
    "last_reconcile": None,
    "started_at": time.time(),
    "warning": None,
}


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
        "rerank": bool(reranker),
        "rerank_model": CONFIG.rerank_model if reranker else None,
        "qdrant": qdrant_ok,
        "vector_store": "embedded" if store.embedded else "server",
        "graph": bool(graph and graph.has_data()),
        "hybrid": bool(graph and graph.has_data() and CONFIG.hybrid),
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
        "indexing": STATE["ingest_running"],
        "last_ingest": STATE["last_ingest"],
        "last_reconcile": STATE["last_reconcile"],
        "uptime_seconds": round(time.time() - STATE["started_at"], 1),
        "error": STATE["ingest_error"],
    }


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


@app.get("/chat/config")
def chat_config():
    """Whether chat is available, and where the documents would go.

    The destination is reported rather than assumed: this indexes private
    material, and "which host am I about to send it to" is the question anyone
    running it should be able to answer before typing.
    """
    if llm is None:
        return {
            "enabled": False,
            "detail": "no generator configured - set RAG_CHAT_PROVIDER in rag/.env",
        }
    return {
        "enabled": True,
        "provider": llm.name,
        "model": llm.model,
        "endpoint": llm.endpoint,
        "local": llm.local,
        # A reachability check, not a promise: the model can still be missing
        # or the key rejected at the moment of asking.
        "problem": llm.check(),
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
    if llm is None:
        return JSONResponse(
            status_code=501,
            content={
                "error": "chat is not configured",
                "detail": (
                    "set RAG_CHAT_PROVIDER (anthropic, openai or ollama) in "
                    "rag/.env - search works without it"
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

    # Retrieve before opening the stream: a retrieval failure should be an
    # ordinary status code, not an error event inside a 200.
    context_block = ""
    sources: list[dict] = []
    ranking: list[str] = []
    graph_used = None
    if request.retrieve:
        found = context(
            ContextRequest(
                query=_retrieval_query(history),
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
        context_block = found["context"]
        sources = found["sources"]
        ranking = found.get("ranking", [])
        graph_used = found.get("graph")

    # Older turns are replayed for pronouns and follow-ups, but their
    # passages are not: re-sending every turn's evidence would fill the
    # context window with retrieval nobody asked about again.
    turns = history[-(CONFIG.chat_history_turns + 1):]
    question = turns[-1]["content"]
    if context_block:
        turns = turns[:-1] + [{
            "role": "user",
            "content": (
                f"{question}\n\n"
                f"Retrieved passages:\n\n{context_block}"
            ),
        }]
    elif request.retrieve:
        turns = turns[:-1] + [{
            "role": "user",
            "content": (
                f"{question}\n\n"
                "Retrieved passages: none - the search returned nothing for "
                "this question. Say so rather than answering from general "
                "knowledge."
            ),
        }]

    def event(name: str, payload: dict) -> str:
        # json.dumps also does the SSE framing work: it escapes the newlines
        # that would otherwise end the data field early.
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    def stream():
        yield event("sources", {
            "sources": sources,
            "ranking": ranking,
            "graph": graph_used,
            "provider": llm.name,
            "model": llm.model,
        })
        try:
            for fragment in llm.stream(SYSTEM_PROMPT, turns):
                if fragment:
                    yield event("delta", {"text": fragment})
        except LlmError as exc:
            yield event("error", {"error": str(exc)})
            return
        except Exception as exc:  # a bug here must not look like a hung stream
            log.exception("chat failed")
            yield event("error", {"error": f"{type(exc).__name__}: {exc}"})
            return
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
