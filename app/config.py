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
