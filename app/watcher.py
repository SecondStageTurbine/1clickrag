# SPDX-License-Identifier: MPL-2.0
"""Filesystem watcher: re-index changed files, debounced.

Keeps the index live while you edit, so a semantic query never answers from a
stale copy of a file you just saved.
"""

from __future__ import annotations

import logging
import os
import threading

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .config import Config
from .ingest import remove_paths, run_ingest
from .scanner import is_excluded, language_for

log = logging.getLogger("rag.watcher")


class _Handler(FileSystemEventHandler):
    def __init__(self, cfg: Config, store=None, embedder=None, queue=None) -> None:
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        # None falls back to indexing inline, which is what `python -m
        # app.watcher` style use gets; the server always passes one.
        self.queue = queue
        self._lock = threading.Lock()
        self._changed: set[str] = set()
        self._deleted: set[str] = set()
        self._timer: threading.Timer | None = None

    def _relative(self, path: str) -> str | None:
        try:
            rel = os.path.relpath(path, self.cfg.repo_path).replace(os.sep, "/")
        except ValueError:
            return None
        if rel.startswith(".."):
            return None
        if is_excluded(rel, self.cfg) or language_for(rel, self.cfg) is None:
            return None
        return rel

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory or event.event_type not in (
            "created",
            "modified",
            "moved",
            "deleted",
        ):
            return

        with self._lock:
            if event.event_type == "deleted":
                rel = self._relative(event.src_path)
                if rel:
                    self._deleted.add(rel)
                    self._changed.discard(rel)
            elif event.event_type == "moved":
                src = self._relative(event.src_path)
                dest = self._relative(getattr(event, "dest_path", ""))
                if src:
                    self._deleted.add(src)
                if dest:
                    self._changed.add(dest)
            else:
                rel = self._relative(event.src_path)
                if rel:
                    self._changed.add(rel)
                    self._deleted.discard(rel)

            if not self._changed and not self._deleted:
                return
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.cfg.watch_debounce_seconds, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            changed = sorted(self._changed)
            deleted = sorted(self._deleted)
            self._changed.clear()
            self._deleted.clear()
            self._timer = None

        # Record the work, do not do it. A file is often still being written
        # when its change event arrives - Word writes a temp file and renames,
        # a copy over the network lands in pieces - and indexing it here meant
        # one failed read discarded the change permanently. The queue survives
        # both that and the process dying mid-embed.
        if self.queue is not None:
            if deleted:
                self.queue.enqueue(deleted, op="delete")
            if changed:
                self.queue.enqueue(changed, op="index")
                log.info("queued %d changed file(s)", len(changed))
            return

        try:
            if deleted:
                remove_paths(self.cfg, deleted, store=self.store)
            if changed:
                log.info("re-indexing %d changed file(s)", len(changed))
                run_ingest(
                    self.cfg,
                    full=False,
                    paths=changed,
                    store=self.store,
                    embedder=self.embedder,
                )
        except Exception as exc:
            log.warning("watch re-index failed: %s", exc)


def start_watcher(cfg: Config, store=None, embedder=None, queue=None) -> Observer:
    observer = Observer()
    observer.schedule(
        _Handler(cfg, store, embedder, queue), cfg.repo_path, recursive=True
    )
    observer.daemon = True
    observer.start()
    log.info(
        "watching %s (debounce %.1fs)", cfg.repo_path, cfg.watch_debounce_seconds
    )
    return observer
