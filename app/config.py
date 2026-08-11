# SPDX-License-Identifier: MPL-2.0
"""Configuration for the one-click RAG stack.

Everything is environment-driven so the same image works for any corpus:
point ``RAG_REPO_PATH`` (host side, in docker-compose) at a
different directory and nothing else changes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger("rag.config")

RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path: str) -> None:
    """Read ``rag/.env`` into the environment.

    docker-compose reads .env by itself; native mode has nothing that would, so
    without this the file documented in .env.example is silently ignored. Real
    environment variables always win — .env is the fallback, not an override.
    """
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as exc:
        log.warning("could not read %s: %s", path, exc)
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(os.path.join(RAG_DIR, ".env"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_ext_map(entries: list[str]) -> dict:
    """Turn ['.sql', '.cfg=conf'] into {'.sql': 'sql', '.cfg': 'conf'}."""
    mapping: dict[str, str] = {}
    for entry in entries:
        ext, _, tag = entry.partition("=")
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        mapping[ext] = (tag.strip() or ext.lstrip(".")).lower()
    return mapping


# Extension -> language tag. The tags match the `language_filter` values the
# project's coders already use (rs, markdown, toml, ld, yaml, json, conf).
LANGUAGE_BY_EXT: dict[str, str] = {
    ".rs": "rs",
    ".md": "markdown",
    ".markdown": "markdown",
    ".toml": "toml",
    ".ld": "ld",
    ".x": "ld",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".conf": "conf",
    ".sh": "sh",
    ".ps1": "ps1",
    ".py": "py",
    ".s": "asm",
    ".asm": "asm",
    ".c": "c",
    ".h": "c",
    ".txt": "text",
    ".rst": "text",
    # Document formats. These are not readable as plain text; app/extract.py
    # turns each into text before chunking.
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".csv": "csv",
    ".tsv": "csv",
    ".html": "html",
    ".htm": "html",
    ".xls": "xlsx",
    ".odt": "docx",
    ".ods": "xlsx",
    ".odp": "pptx",
    ".rtf": "docx",
    ".eml": "email",
    ".msg": "email",
    ".xml": "xml",
    # A comic archive - page images in a zip. Only readable with RAG_OCR=1.
    ".cbz": "comic",
}

# Files with no extension that are still worth indexing.
FILENAME_LANGUAGES: dict[str, str] = {
    "Dockerfile": "conf",
    "Makefile": "conf",
}

DEFAULT_EXCLUDE_DIRS: list[str] = [
    ".git",
    "target",
    "node_modules",
    # Native mode keeps its virtualenv and its vector storage inside rag/,
    # which is inside the repo being indexed. Without these the index ingests
    # site-packages and its own Qdrant segment metadata — the chunk count then
    # drifts on every restart as the store rewrites itself.
    ".venv",
    "venv",
    ".data",
    ".direnv",
    "__pycache__",
    ".cargo-home",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "qdrant_storage",
    # Superseded history. Where a project keeps one, it is usually a large
    # share of the indexable files and almost never the answer to a question
    # about how things work now. Use RAG_EXCLUDE_DIRS_ONLY to index it anyway.
    "archive",
]

# Substrings that mark a path as build output / evidence noise rather than
# source worth embedding.
DEFAULT_EXCLUDE_GLOBS: list[str] = [
    "*.lock",
    "*.log",
    "*.bin",
    "*.img",
    "*.iso",
    "*.elf",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.gz",
    "*.xz",
    "*.o",
    "*.a",
    "*.so",
    "*.rlib",
]


# Two supported topologies, selected by RAG_MODE:
#
#   native (default) - one Python process. Embeddings run in-process through
#                      fastembed/ONNX; the vector store is qdrant-client's
#                      file-backed embedded mode. No daemon, no container.
#   docker           - the compose stack: Ollama for embeddings, a Qdrant
#                      server for storage. Heavier, but fully isolated and
#                      byte-reproducible across machines.
#
# Every individual setting below can still be overridden regardless of mode.
MODE = os.environ.get("RAG_MODE", "native").strip().lower()
_IS_DOCKER = MODE == "docker"

# The corpus defaults to whatever contains rag/ — correct while this folder
# lives inside the repository it indexes, and meaningless once it is moved
# somewhere else, which is what the startup check below catches.
_DEFAULT_REPO = "/repo" if _IS_DOCKER else os.path.dirname(RAG_DIR)
_DEFAULT_DATA = os.path.join(RAG_DIR, ".data")


@dataclass
class Config:
    """Resolved runtime configuration."""

    mode: str = MODE
    repo_path: str = os.environ.get("RAG_REPO_MOUNT", _DEFAULT_REPO)
    repo_label: str = os.environ.get("RAG_REPO_LABEL", "repository")

    # Vector store. An empty qdrant_url means embedded mode backed by
    # qdrant_path; set RAG_QDRANT_URL to talk to a Qdrant server instead.
    qdrant_url: str = os.environ.get(
        "RAG_QDRANT_URL", "http://qdrant:6333" if _IS_DOCKER else ""
    )
    qdrant_path: str = os.environ.get(
        "RAG_QDRANT_PATH", os.path.join(_DEFAULT_DATA, "qdrant")
    )
    collection: str = os.environ.get("RAG_COLLECTION", "codebase")

    # Embeddings: "fastembed" (in-process ONNX) or "ollama" (HTTP daemon).
    embed_backend: str = os.environ.get(
        "RAG_EMBED_BACKEND", "ollama" if _IS_DOCKER else "fastembed"
    ).strip().lower()
    embed_model: str = os.environ.get(
        "RAG_EMBED_MODEL",
        "nomic-embed-text" if _IS_DOCKER else "nomic-ai/nomic-embed-text-v1.5",
    )
    ollama_url: str = os.environ.get("RAG_OLLAMA_URL", "http://ollama:11434")
    auto_pull_model: bool = _env_bool("RAG_AUTO_PULL_MODEL", True)
    # Where fastembed caches its ONNX weights (first run downloads them).
    model_cache: str = os.environ.get(
        "RAG_MODEL_CACHE", os.path.join(_DEFAULT_DATA, "models")
    )

    # Loopback-only by default in native mode; the container needs 0.0.0.0 so
    # the published port can reach it.
    host: str = os.environ.get("RAG_HOST", "0.0.0.0" if _IS_DOCKER else "127.0.0.1")
    port: int = _env_int("RAG_PORT", 49404)

    # Chunking. Sizes are in characters, not tokens: cheap to compute and
    # close enough for a code index.
    chunk_chars: int = _env_int("RAG_CHUNK_CHARS", 1600)
    chunk_overlap_lines: int = _env_int("RAG_CHUNK_OVERLAP_LINES", 8)
    max_file_bytes: int = _env_int("RAG_MAX_FILE_BYTES", 25_000_000)

    # How much of a chunk's location is embedded alongside its text.
    #   full - "dir/sub/file.pdf" (default; makes "where is X done?" work)
    #   name - "file.pdf" only, dropping directory names
    #   off  - the text alone
    # The path is real signal for code, where the directory IS the subsystem.
    # On a document corpus it can hurt: a folder called "Volume_I" pulls every
    # chunk beneath it toward any query mentioning volumes. Changing this
    # changes the vectors, so follow it with `reindex -Full`.
    embed_path: str = os.environ.get("RAG_EMBED_PATH", "full").strip().lower()

    # Which label is embedded alongside the chunk's own text.
    #   symbol  - the chunk's own first line (default)
    #   section - the heading the chunk sits under
    #   off     - neither
    #
    # `section` is the one that ought to win. It says where a passage lives,
    # which the passage never states itself - a paragraph does not mention the
    # chapter it is in - while `symbol` mostly repeats text the vector already
    # holds. Prepending document and section context before embedding is the
    # standard advice, and it is the cheap half of contextual retrieval.
    #
    # It was measured and it did not win. On a heading-rich corpus, ten
    # questions phrased away from the headings' own wording: identical with
    # reranking on (MRR 0.820 either way, no question moving a place), and
    # slightly worse with it off (0.817 -> 0.808). The cross-encoder re-reads
    # query and passage together, so what the vector put in eleventh place is
    # recoverable anyway - which is presumably why the label matters so little
    # here. On a corpus of documents without headings it cannot matter at all:
    # `section` falls back to `symbol`.
    #
    # So the default stays where the evidence is, and the knob stays because
    # ten questions on one corpus is weak evidence, not a law. Worth trying
    # `section` on a large structured corpus with `--compare` before believing
    # either result. Changing this changes every vector: reindex -Full after.
    embed_label: str = os.environ.get("RAG_EMBED_LABEL", "symbol").strip().lower()

    # Cross-encoder reranking. The vector search retrieves `rerank_candidates`
    # and the reranker re-orders them down to top_k. Off by default: it costs a
    # second or two per query on CPU and an extra model download. Worth it when
    # feeding a generator, where the top few chunks decide the answer.
    rerank: bool = _env_bool("RAG_RERANK", False)
    rerank_model: str = os.environ.get(
        "RAG_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"
    )
    rerank_candidates: int = _env_int("RAG_RERANK_CANDIDATES", 40)

    # Keyword search and the entity graph, in one SQLite file beside the
    # vectors. Stdlib-only (the bundled interpreter has FTS5), so this costs
    # nothing in an offline install and no daemon at runtime. Turning it off
    # leaves plain vector search exactly as it was; turning it on takes effect
    # for a file when that file is next indexed, so a corpus indexed before
    # this existed needs `reindex -Full` to populate it.
    graph: bool = _env_bool("RAG_GRAPH", True)
    graph_path: str = os.environ.get(
        "RAG_GRAPH_PATH", os.path.join(_DEFAULT_DATA, "graph.db")
    )
    # Fuse BM25 with the vector ranking instead of using vectors alone. This is
    # what makes exact strings - ticket IDs, part numbers, error codes -
    # findable, which is the one thing embeddings are reliably bad at.
    hybrid: bool = _env_bool("RAG_HYBRID", True)
    # Reciprocal-rank-fusion constant. Ranks, not scores, are combined: a cosine
    # similarity, a BM25 score and an entity-overlap count share no scale, and
    # normalising them against each other invents a comparison that does not
    # exist.
    rrf_k: int = _env_int("RAG_RRF_K", 60)

    # Search again using the words the documents use, when the question's own
    # wording matches little. Deterministic and model-free - two extra BM25
    # passes over SQLite - so it works with any chat model, or none, and helps
    # /search and /context as much as the chat pane. See app/expand.py for why
    # it exists: a question phrased in the reader's vocabulary can miss the
    # passage that answers it entirely.
    expand: bool = _env_bool("RAG_EXPAND", True)
    # How many of the first pass's passages contribute vocabulary, and how many
    # words are taken from them. Both small on purpose: the follow-up is meant
    # to be a sharper query, not a wider one.
    expand_feedback_docs: int = _env_int("RAG_EXPAND_FEEDBACK_DOCS", 5)
    expand_terms: int = _env_int("RAG_EXPAND_TERMS", 8)
    # A harvested word in more than this share of the corpus is furniture, not
    # subject matter, and searching with it retrieves everything. Raise it on a
    # corpus of unrelated documents, where any given word is rarer.
    expand_max_share: float = float(os.environ.get("RAG_EXPAND_MAX_SHARE", "0.08"))

    # An entity in more than this share of the documents is boilerplate - a
    # header, a footer, the corpus's own subject - and connects everything to
    # everything. Raise it on a corpus of unrelated documents.
    graph_max_df: float = float(os.environ.get("RAG_GRAPH_MAX_DF", "0.2"))
    # How many neighbours one entity contributes per hop.
    graph_fanout: int = _env_int("RAG_GRAPH_FANOUT", 8)
    # Default hops for /search and /context. 0 keeps graph expansion opt-in per
    # request: it widens the candidate set, which is what you want for
    # "who else was involved", and noise for "what does this sentence mean".
    graph_hops: int = _env_int("RAG_GRAPH_HOPS", 0)

    # --- an OpenAI-shaped embeddings service (RAG_EMBED_BACKEND=http) -------
    # For a hosted or company-internal endpoint. The base URL, so "/embeddings"
    # is appended: https://ai.internal/v1
    embed_url: str = os.environ.get("RAG_EMBED_URL", "")
    embed_api_key: str = os.environ.get("RAG_EMBED_API_KEY", "")
    # Gateways differ: most take Authorization: Bearer <key>, some want
    # api-key: <key> with no prefix at all.
    embed_auth_header: str = os.environ.get("RAG_EMBED_AUTH_HEADER", "Authorization")
    embed_auth_prefix: str = os.environ.get("RAG_EMBED_AUTH_PREFIX", "Bearer")
    # Retrieval models are trained asymmetrically - nomic expects
    # "search_document: " on passages and "search_query: " on questions, e5
    # wants "passage: " and "query: ". fastembed applies these itself; a raw
    # HTTP endpoint does not, and omitting them degrades ranking while looking
    # like it works. Set them to whatever the endpoint's model expects.
    embed_doc_prefix: str = os.environ.get("RAG_EMBED_DOC_PREFIX", "")
    embed_query_prefix: str = os.environ.get("RAG_EMBED_QUERY_PREFIX", "")
    # Inputs per request. Gateways cap this where a local model does not.
    embed_http_batch: int = _env_int("RAG_EMBED_HTTP_BATCH", 64)
    embed_http_timeout: float = float(os.environ.get("RAG_EMBED_HTTP_TIMEOUT", "120"))
    # For an internal endpoint behind a corporate CA the machine does not trust
    # yet. A real situation and a bad default, hence opt-out rather than opt-in.
    embed_verify_tls: bool = _env_bool("RAG_EMBED_VERIFY_TLS", True)

    # Run the embedding model on the GPU. Needs onnxruntime-gpu, which REPLACES
    # the CPU onnxruntime rather than sitting beside it - so this is off by
    # default and falls back to CPU with a warning when no CUDA provider is
    # present, rather than refusing to start. It speeds up a first index over a
    # large corpus; it does almost nothing for query latency, where embedding
    # one short question is a couple of per cent of the work.
    embed_gpu: bool = _env_bool("RAG_EMBED_GPU", False)

    embed_batch: int = _env_int("RAG_EMBED_BATCH", 32)
    # ONNX inference threads. Defaults to this machine's core count rather than
    # leaving onnxruntime to choose, because a portable copy lands on an unknown
    # machine and a hardcoded number would be wrong on most of them. Embedding
    # is the entire cost of a first ingest, so this is the knob that decides
    # whether a large share takes an afternoon or overnight.
    #
    # Set RAG_EMBED_THREADS=0 to hand the decision back to onnxruntime. Which is
    # faster is machine-specific and has not been measured here - if a big
    # ingest seems slow, it is the first thing worth trying both ways.
    embed_threads: int = _env_int("RAG_EMBED_THREADS", os.cpu_count() or 0)
    snippet_chars: int = _env_int("RAG_SNIPPET_CHARS", 400)

    ingest_on_start: bool = _env_bool("RAG_INGEST_ON_START", True)
    # Reconcile the index against the corpus at startup. The watcher only sees
    # what happens while the process is alive, so without this every file added,
    # edited or deleted while the server was stopped stays wrong until someone
    # notices and reindexes by hand - and nobody notices, because a search that
    # is missing a document looks exactly like a search with no answer.
    reconcile_on_start: bool = _env_bool("RAG_RECONCILE_ON_START", True)
    # Minutes between periodic reconciles. -1 picks a default from `watch`:
    # off when the watcher is trusted, every 15 minutes when it is not, which
    # is what makes a network share stay current despite SMB not delivering
    # change notifications.
    rescan_minutes: int = _env_int("RAG_RESCAN_MINUTES", -1)

    # Durable work queue. Changes seen by the watcher are recorded here before
    # being indexed, so a file that is still being written - or a process that
    # dies mid-embed - costs a retry rather than a permanently stale index.
    queue_path: str = os.environ.get(
        "RAG_QUEUE_PATH", os.path.join(_DEFAULT_DATA, "ingest-queue.db")
    )
    # Attempts before an item is parked in the dead-letter table, where it
    # stops burning retries and starts being evidence (GET /queue).
    queue_max_attempts: int = _env_int("RAG_QUEUE_MAX_ATTEMPTS", 5)
    # How many paths one worker pass takes on. Batching amortises the setup
    # each ingest does; too large and one bad file delays the rest.
    queue_batch: int = _env_int("RAG_QUEUE_BATCH", 25)
    queue_poll_seconds: float = float(os.environ.get("RAG_QUEUE_POLL", "2.0"))

    # Expand a retrieved chunk to the rest of its section before handing it to
    # a generator. A chunk is a retrieval unit, not a unit of meaning: the
    # paragraph that matched is often the middle of the argument.
    context_expand: bool = _env_bool("RAG_CONTEXT_EXPAND", True)

    # --- the chat pane ------------------------------------------------------
    # Which generator writes the answers, or empty for none. Empty is the
    # default and is not a placeholder: this installs on machines with no
    # internet and no local model, and search works exactly as before without
    # it. /chat answers 501 and the browser UI hides the tab.
    #   anthropic  the Claude Messages API
    #   openai     any OpenAI-compatible /chat/completions (OpenAI, Codex
    #              models, llama.cpp, LM Studio, vLLM, Ollama's shim)
    #   ollama     Ollama's own /api/chat
    chat_provider: str = os.environ.get("RAG_CHAT_PROVIDER", "").strip().lower()
    chat_model: str = os.environ.get("RAG_CHAT_MODEL", "")
    # Base URL. Ignored for anthropic; defaults to https://api.openai.com/v1
    # for openai and http://127.0.0.1:11434 for ollama.
    chat_url: str = os.environ.get("RAG_CHAT_URL", "")
    # Never checked into the repo and stripped from the packaged .env - it
    # belongs in rag/.env on the machine that has it, or in the environment.
    chat_api_key: str = os.environ.get("RAG_CHAT_API_KEY", "")
    # Ceiling on the answer. Generous because the newer Claude models count
    # their own reasoning against this budget, so a tight one truncates the
    # answer rather than merely shortening it.
    chat_max_tokens: int = _env_int("RAG_CHAT_MAX_TOKENS", 4096)
    # Low on purpose: the job is to report what the passages say, not to write
    # around them. Not sent to the Claude models, which reject sampling
    # parameters outright.
    chat_temperature: float = float(os.environ.get("RAG_CHAT_TEMPERATURE", "0.2"))
    # Seconds to wait for a whole answer. A 7B model on CPU is slow.
    chat_timeout: float = float(os.environ.get("RAG_CHAT_TIMEOUT", "180"))
    # Retrieval for one chat turn: how many chunks, and how many characters of
    # them survive into the prompt.
    chat_top_k: int = _env_int("RAG_CHAT_TOP_K", 8)
    chat_context_chars: int = _env_int("RAG_CHAT_CONTEXT_CHARS", 12000)
    # Prior turns replayed to the generator. The retrieved passages are the
    # expensive part of each prompt, so history is kept short deliberately.
    chat_history_turns: int = _env_int("RAG_CHAT_HISTORY_TURNS", 6)
    # How many times the model may ask for a different search before it has to
    # answer with what it has. Retrieval matches the question's wording against
    # the documents', and when those differ the right passage is missing
    # entirely - this lets a model that notices say so and try the document's
    # own vocabulary instead. Costs a full generation per round, and a model
    # that kept searching would never answer, hence the cap. 0 disables it.
    chat_max_searches: int = _env_int("RAG_CHAT_MAX_SEARCHES", 2)
    # Watching defaults OFF for a UNC share. SMB does not deliver change
    # notifications dependably, so the watcher would appear to work while
    # quietly missing edits - worse than not running, because the index looks
    # live. Set RAG_WATCH=1 to force it on anyway.
    watch: bool = _env_bool(
        "RAG_WATCH",
        not os.environ.get("RAG_REPO_MOUNT", _DEFAULT_REPO).replace("\\", "/").startswith("//"),
    )
    watch_debounce_seconds: float = float(os.environ.get("RAG_WATCH_DEBOUNCE", "2.0"))

    # ADDED to the defaults, not replacing them. Replacing was a trap: setting
    # RAG_EXCLUDE_DIRS=_to_delete to skip one folder would also have dropped
    # .git, .venv and .data from the exclusions - and the indexer would have
    # walked into its own virtualenv and vector store.
    # RAG_EXCLUDE_DIRS_ONLY replaces the list outright, for anyone who means it.
    exclude_dirs: list[str] = field(
        default_factory=lambda: _env_list(
            "RAG_EXCLUDE_DIRS_ONLY",
            DEFAULT_EXCLUDE_DIRS + _env_list("RAG_EXCLUDE_DIRS", []),
        )
    )
    exclude_globs: list[str] = field(
        default_factory=lambda: _env_list(
            "RAG_EXCLUDE_GLOBS_ONLY",
            DEFAULT_EXCLUDE_GLOBS + _env_list("RAG_EXCLUDE_GLOBS", []),
        )
    )
    # Empty means "every extension we know about".
    include_languages: list[str] = field(
        default_factory=lambda: _env_list("RAG_INCLUDE_LANGUAGES", [])
    )

    # Extra plain-text extensions, so a new file type needs no code change.
    #   RAG_EXTRA_TEXT_EXTS=.sql,.ini,.log
    #   RAG_EXTRA_TEXT_EXTS=.sql=sql,.cfg=conf     (explicit language tag)
    # The tag is what `language_filter` matches on; it defaults to the
    # extension without its dot.
    extra_text_exts: dict = field(
        default_factory=lambda: _parse_ext_map(_env_list("RAG_EXTRA_TEXT_EXTS", []))
    )

    # --- OCR ----------------------------------------------------------------
    # Read text off pages that carry it as pixels: scans, photographed reports,
    # comics. Off by default because it is roughly a thousand times slower than
    # reading a text layer - seconds per page against microseconds - so a folder
    # of scans is an overnight job rather than a pause, and nobody should
    # discover that by starting one unawares. With this off, a PDF whose pages
    # have no text layer is reported in the log and indexed as empty.
    ocr: bool = _env_bool("RAG_OCR", False)
    # A page yielding fewer than this many characters is treated as scanned and
    # sent to OCR. Not zero: a scanned page often carries a stray ligature or a
    # stamped page number in its text layer while the actual words are pixels.
    ocr_min_chars: int = _env_int("RAG_OCR_MIN_CHARS", 20)
    # Height of one recognition band, and how much consecutive bands share.
    # Tall pages are sliced rather than shrunk: fitting a 20,000px webtoon strip
    # into the detector's window squashes the text into illegibility. The
    # overlap stops a line falling in a cut from being lost.
    ocr_band: int = _env_int("RAG_OCR_BAND", 2200)
    ocr_overlap: int = _env_int("RAG_OCR_OVERLAP", 200)
    # Readings below this score are dropped. Artwork is the reason: a detector
    # pointed at a drawing finds "text" in hatching and panel borders, and those
    # come back with low scores and no meaning.
    ocr_min_confidence: float = float(os.environ.get("RAG_OCR_MIN_CONFIDENCE", "0.5"))
    # A guard against a corrupt or synthetic page claiming an enormous size and
    # exhausting memory during render, not a resolution policy.
    ocr_max_megapixels: int = _env_int("RAG_OCR_MAX_MEGAPIXELS", 80)

    # Keep extracted text so a rebuild need not re-read the documents. Indexing
    # is two jobs of very different cost - turning a PDF into text takes
    # seconds or, with OCR, minutes, while chunking and embedding that text
    # takes milliseconds - and a full reindex used to redo both. It exists for
    # when chunking or embedding changes, and neither of those alters what the
    # words on the page are.
    #
    # Keyed by the SHA-256 of the file's bytes and by every setting that
    # changes extraction output, so it survives `reindex -Full` (which is
    # exactly when timestamps are not to be trusted) and invalidates itself
    # when OCR settings change.
    #
    # It holds verbatim document text: never committed, never packaged, and
    # dropped by `down -Wipe`.
    extract_cache: bool = _env_bool("RAG_EXTRACT_CACHE", True)
    extract_cache_path: str = os.environ.get(
        "RAG_EXTRACT_CACHE_PATH", os.path.join(_DEFAULT_DATA, "extract-cache.db")
    )
    # Compressed text, so this goes a long way - prose runs about four to one.
    # The coldest entries are dropped when it is exceeded.
    extract_cache_max_mb: int = _env_int("RAG_EXTRACT_CACHE_MAX_MB", 2048)

    # Where the record of what built the index lives. Compared against the
    # running configuration at startup, so a setting that changes the vectors
    # cannot be flipped without anybody noticing the index is now half one
    # thing and half another.
    manifest_path: str = os.environ.get(
        "RAG_MANIFEST_PATH", os.path.join(_DEFAULT_DATA, "index-manifest.json")
    )

    # Look inside .zip archives and index the documents they contain.
    archives: bool = _env_bool("RAG_ARCHIVES", True)
    archive_max_bytes: int = _env_int("RAG_ARCHIVE_MAX_BYTES", 500_000_000)
    archive_max_members: int = _env_int("RAG_ARCHIVE_MAX_MEMBERS", 2000)


    def __post_init__(self) -> None:
        if self.rescan_minutes < 0:
            # A watched local disk needs no polling; an unwatched share needs
            # it to stay current at all.
            self.rescan_minutes = 0 if self.watch else 15


CONFIG = Config()


def corpus_warning(cfg: Config = CONFIG) -> str | None:
    """Explain an implausible corpus, or None if it looks fine.

    This exists for the moved-folder case: if rag/ is copied out of the
    repository it indexes, the default corpus silently becomes whatever
    directory now contains it — a Desktop, a home folder — and the first
    symptom would otherwise be a confusingly slow ingest of the wrong files.
    """
    if os.environ.get("RAG_REPO_MOUNT"):
        return None  # explicitly chosen: the operator knows what they meant
    if not os.path.isdir(cfg.repo_path):
        return f"corpus path does not exist: {cfg.repo_path}"
    if not os.path.isdir(os.path.join(cfg.repo_path, ".git")):
        return (
            f"indexing {cfg.repo_path}, which is not a git repository — rag/ is "
            "probably outside the project you meant to index. Set RAG_REPO_MOUNT "
            "in rag/.env to the repository path."
        )
    return None
