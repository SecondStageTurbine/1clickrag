# SPDX-License-Identifier: MPL-2.0
"""Walk the mounted repository, chunk it, embed it, and upsert into Qdrant.

Run standalone inside the container:

    python -m app.ingest            # incremental (only changed/new files)
    python -m app.ingest --full -v  # drop the collection and rebuild
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from dataclasses import dataclass, field

from .chunker import chunk_file
from .config import CONFIG, Config
from .embedder import make_embedder
from .graph import open_graph
from .scanner import (
    is_excluded,
    iter_archive_members,
    iter_source_files,
    language_for,
    load_text,
    read_text,
    unsupported_counts,
)
from .store import Store

__all__ = [
    "IngestStats",
    "run_ingest",
    "remove_paths",
    "is_excluded",
    "iter_source_files",
    "language_for",
    "read_text",
    "load_text",
]

log = logging.getLogger("rag.ingest")

# One ingest at a time: the watcher and an API-triggered reindex must not race.
INGEST_LOCK = threading.Lock()


@dataclass
class IngestStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    chunks: int = 0
    errors: int = 0
    # Which files failed, not just how many. The work queue retries by path, so
    # a count would leave it guessing which item to blame for a failed batch.
    failures: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "files_scanned": self.files_scanned,
            "files_indexed": self.files_indexed,
            "files_skipped": self.files_skipped,
            "files_unchanged": self.files_unchanged,
            "files_removed": self.files_removed,
            "chunks": self.chunks,
            "errors": self.errors,
            "failures": self.failures,
        }


def source_mtime(cfg: Config, rel_path: str) -> float | None:
    """Modification time of the real file behind a path, or None if it is gone.

    Archive members resolve to the archive itself: a document inside a zip
    cannot change without the zip changing, and statting the member would mean
    opening the archive for every file in it on every scan.
    """
    from .archive import split_member

    member = split_member(rel_path)
    source = member[0] if member else rel_path
    try:
        return round(os.path.getmtime(os.path.join(cfg.repo_path, source)), 3)
    except OSError:
        return None


def index_file(
    rel_path: str,
    language: str,
    cfg: Config,
    embedder,
    store: Store,
    stats: IngestStats,
) -> None:
    from .archive import split_member

    member = split_member(rel_path)
    source = member[0] if member else rel_path
    abs_path = os.path.join(cfg.repo_path, source)
    if member:
        abs_path = f"{abs_path}!{member[1]}"
    content = load_text(abs_path, cfg.max_file_bytes)
    if content is None:
        stats.files_skipped += 1
        return

    mtime = source_mtime(cfg, rel_path)
    chunks = chunk_file(
        rel_path,
        language,
        content,
        chunk_chars=cfg.chunk_chars,
        overlap_lines=cfg.chunk_overlap_lines,
    )
    if not chunks:
        stats.files_skipped += 1
        return

    for chunk in chunks:
        chunk.source = source

    # What location context goes into the vector - see Config.embed_path.
    #
    # Which label rides along with the text - see Config.embed_label, which
    # also records what happened when the alternative was measured.
    def label(chunk) -> str:
        if cfg.embed_label == "symbol":
            return chunk.symbol or ""
        if cfg.embed_label == "off":
            return ""
        return chunk.section or chunk.symbol or ""

    if cfg.embed_path == "off":
        payloads = [c.text for c in chunks]
    elif cfg.embed_path == "name":
        payloads = [
            f"{os.path.basename(c.path)}\n{label(c)}\n\n{c.text}" for c in chunks
        ]
    else:
        payloads = [f"{c.path}\n{label(c)}\n\n{c.text}" for c in chunks]
    vectors = embedder.embed_documents(payloads, batch_size=cfg.embed_batch)

    store.delete_path(rel_path)
    store.upsert(chunks, vectors, mtime=mtime)

    # The sidecar is updated from the same place, with the same delete-then-
    # insert, so keyword search and the entity graph cannot drift from the
    # vectors across incremental re-ingests. A failure here is logged and
    # swallowed: losing the graph for one file is a degraded search, while
    # letting it abort the ingest would lose the vectors too.
    graph = open_graph(cfg)
    if graph is not None:
        try:
            graph.replace_path(rel_path, chunks)
        except Exception as exc:
            log.warning("graph update failed for %s: %s", rel_path, exc)

    stats.files_indexed += 1
    stats.chunks += len(chunks)


def _reconcile(
    cfg: Config,
    store: Store,
    targets: list[tuple[str, str]],
    stats: IngestStats,
) -> list[tuple[str, str]]:
    """Compare the index against the corpus and return only the stale files.

    This is what makes the index survive being switched off. The watcher sees
    changes only while the process is alive, so anything added, edited or
    deleted in between is invisible to it - and an index quietly missing a
    document is indistinguishable, from the outside, from a question that has
    no answer.

    Also what makes a plain reindex cheap: unchanged files keep their vectors
    instead of being re-embedded, and embedding is the entire cost of a run.
    """
    try:
        indexed = store.indexed_state()
    except Exception as exc:
        # Reconciliation is an optimisation plus a correctness sweep; if the
        # scroll fails, index everything rather than nothing.
        log.warning("could not read the index state, re-indexing everything: %s", exc)
        return targets

    present = {rel_path for rel_path, _ in targets}
    stale: list[tuple[str, str]] = []
    for rel_path, language in targets:
        known = indexed.get(rel_path)
        current = source_mtime(cfg, rel_path)
        if known and current is not None and known.get("mtime") == current:
            stats.files_unchanged += 1
            continue
        stale.append((rel_path, language))

    # Anything the index holds that the corpus no longer offers: deleted,
    # renamed, moved out, newly excluded, or a member of an archive that no
    # longer contains it.
    vanished = sorted({path for path in indexed if path not in present})
    if vanished:
        # A share that failed to mount looks exactly like a corpus where every
        # file was deleted, and acting on that would empty the index in one
        # pass. Requiring the corpus to still hold something makes the
        # difference: an empty walk is treated as a fault, not as an
        # instruction.
        if not present and indexed:
            log.warning(
                "the corpus looks empty (%s) while the index holds %d file(s) - "
                "skipping deletions, since an unreachable folder is not the same "
                "as an emptied one",
                cfg.repo_path,
                len(indexed),
            )
        else:
            graph = open_graph(cfg)
            for rel_path in vanished:
                store.delete_path(rel_path)
                if graph is not None:
                    try:
                        graph.delete_path(rel_path)
                    except Exception as exc:
                        log.warning("graph delete failed for %s: %s", rel_path, exc)
                stats.files_removed += 1
            log.info("removed %d file(s) no longer in the corpus", len(vanished))

    if stats.files_unchanged:
        log.info(
            "%d file(s) unchanged since the last index - skipping", stats.files_unchanged
        )
    return stale


def run_ingest(
    cfg: Config | None = None,
    full: bool = False,
    paths: list[str] | None = None,
    store: Store | None = None,
    embedder=None,
) -> IngestStats:
    """Index the repo (or just ``paths``). Serialised by INGEST_LOCK.

    ``store``/``embedder`` are passed in by the server so both share one
    instance. That is mandatory in embedded-Qdrant mode, where the storage
    directory is held under an exclusive lock by whoever opened it first.
    """
    cfg = cfg or CONFIG
    stats = IngestStats()

    with INGEST_LOCK:
        embedder = embedder or make_embedder(cfg)
        store = store or Store(cfg.qdrant_url, cfg.collection, cfg.qdrant_path)

        if not embedder.wait_until_alive():
            raise RuntimeError(f"embedding backend not reachable at {cfg.ollama_url}")
        if not store.wait_until_alive():
            raise RuntimeError(f"qdrant not reachable at {cfg.qdrant_url}")

        if cfg.auto_pull_model:
            embedder.prepare()

        store.ensure_collection(embedder.dim, recreate=full)

        graph = open_graph(cfg)
        if graph is not None and full:
            # A full rebuild drops the collection; the sidecar has to go with
            # it, or deleted files survive in keyword search and the graph.
            graph.clear()

        if full:
            # The one moment everything in the index provably came from one set
            # of settings. Written here rather than at the end so a rebuild
            # interrupted halfway is still described by what produced it - the
            # chunks that landed did use these values.
            from . import manifest as manifest_module

            manifest_module.save(
                cfg.manifest_path, manifest_module.current(cfg, embedder.dim)
            )

        if paths is not None:
            from .archive import is_archive

            targets = []
            for rel_path in paths:
                if is_excluded(rel_path, cfg):
                    continue
                # A changed archive means re-reading everything inside it.
                if cfg.archives and is_archive(rel_path):
                    store.delete_path(rel_path)
                    targets.extend(iter_archive_members(rel_path, cfg))
                    continue
                language = language_for(rel_path, cfg)
                if language:
                    targets.append((rel_path, language))
        else:
            targets = list(iter_source_files(cfg))
            if not full:
                targets = _reconcile(cfg, store, targets, stats)

        total = len(targets)
        log.info("indexing %d file(s) from %s", total, cfg.repo_path)

        if paths is None:
            skipped_formats = unsupported_counts(cfg)
            if skipped_formats:
                summary = ", ".join(f"{n} {ext}" for ext, n in sorted(skipped_formats.items()))
                log.warning(
                    "NOT indexed - no reliable reader for the legacy binary formats: %s. "
                    "Re-save them as .docx/.pptx to include them.",
                    summary,
                )

        for i, (rel_path, language) in enumerate(targets, start=1):
            stats.files_scanned += 1
            try:
                index_file(rel_path, language, cfg, embedder, store, stats)
            except Exception as exc:
                stats.errors += 1
                stats.failures[rel_path] = str(exc)
                log.warning("failed to index %s: %s", rel_path, exc)
            if total > 50 and i % 50 == 0:
                log.info("  %d/%d files (%d chunks)", i, total, stats.chunks)

        log.info(
            "ingest complete: %d indexed, %d unchanged, %d removed, %d skipped, "
            "%d chunks, %d errors (collection now holds %d points)",
            stats.files_indexed,
            stats.files_unchanged,
            stats.files_removed,
            stats.files_skipped,
            stats.chunks,
            stats.errors,
            store.count(),
        )

    return stats


def remove_paths(cfg: Config, paths: list[str], store: Store | None = None) -> None:
    """Drop deleted files from the index."""
    store = store or Store(cfg.qdrant_url, cfg.collection, cfg.qdrant_path)
    if not store.exists():
        return
    graph = open_graph(cfg)
    with INGEST_LOCK:
        for rel_path in paths:
            store.delete_path(rel_path)
            if graph is not None:
                try:
                    graph.delete_path(rel_path)
                except Exception as exc:
                    log.warning("graph delete failed for %s: %s", rel_path, exc)
            log.info("removed %s from index", rel_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a repository into the RAG index.")
    parser.add_argument("--full", action="store_true", help="drop the collection and rebuild")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    try:
        run_ingest(full=args.full)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:
        # The usual cause in native mode: the running server already holds the
        # embedded store's directory lock.
        log.error("ingest failed: %s", exc)
        log.error(
            "if the RAG server is running, re-index through it instead: "
            "POST /reindex  (or: rag-up reindex)"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
