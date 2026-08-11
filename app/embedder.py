# SPDX-License-Identifier: MPL-2.0
"""Embedding backends. Nothing leaves the machine in either case.

* ``FastEmbedEmbedder`` - in-process ONNX (the fastembed package). No daemon,
  no container; weights are downloaded once into a local cache directory.
* ``OllamaEmbedder``    - a local Ollama HTTP server, used by the compose stack
  and by anyone who already runs Ollama for other reasons.

Both expose the same interface, and both distinguish *document* embedding from
*query* embedding: several strong retrieval models (nomic, bge, e5) are trained
with asymmetric prefixes, and using the document form for queries measurably
degrades ranking.
"""

from __future__ import annotations

import json
import logging
import os
import time

from . import accel

log = logging.getLogger("rag.embedder")


class EmbeddingError(RuntimeError):
    pass


class FastEmbedEmbedder:
    """In-process embeddings via ONNX. The zero-daemon default."""

    backend = "fastembed"

    def __init__(self, model: str, cache_dir: str, threads: int = 0,
                 gpu: bool = False, gpu_setting: str = "RAG_GPU") -> None:
        self.model_name = model
        self.cache_dir = cache_dir
        self.threads = threads
        self.gpu = gpu
        self.gpu_setting = gpu_setting
        self._model = None
        self._dim: int | None = None
        self._provider = accel.CPU

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - install-time issue
                raise EmbeddingError(
                    "the 'fastembed' package is required for RAG_EMBED_BACKEND=fastembed "
                    "(pip install -r requirements.txt), or set RAG_MODE=docker"
                ) from exc
            os.makedirs(self.cache_dir, exist_ok=True)
            log.info(
                "loading embedding model %s (first run downloads it into %s)",
                self.model_name,
                self.cache_dir,
            )
            kwargs = {"model_name": self.model_name, "cache_dir": self.cache_dir}
            if self.threads > 0:
                kwargs["threads"] = self.threads
            if self.gpu:
                kwargs["providers"] = accel.providers(self.gpu_setting)
            self._model = TextEmbedding(**kwargs)
            self._provider = accel.active(self._model, self.gpu)
            log.info(
                "embedding model %s ready on %s", self.model_name, self._provider
            )
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    # An in-process model has no liveness question once it is loaded.
    def alive(self) -> bool:
        return self._model is not None

    def wait_until_alive(self, timeout: float = 0.0) -> bool:
        return True

    def prepare(self) -> None:
        self._load()

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        model = self._load()
        vectors = [list(map(float, v)) for v in model.embed(texts, batch_size=batch_size)]
        if vectors and self._dim is None:
            self._dim = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        vector = list(map(float, next(iter(model.query_embed([text])))))
        if self._dim is None:
            self._dim = len(vector)
        return vector

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed_query("dimension probe")
        assert self._dim is not None
        return self._dim


class OllamaEmbedder:
    """Embeddings from a local Ollama server."""

    backend = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        import httpx

        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self._client = httpx.Client(timeout=timeout)
        self._dim: int | None = None

    def alive(self) -> bool:
        try:
            return self._client.get(f"{self.base_url}/api/tags", timeout=5.0).status_code == 200
        except self._httpx.HTTPError:
            return False

    def wait_until_alive(self, timeout: float = 300.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.alive():
                return True
            time.sleep(2.0)
        return False

    def has_model(self) -> bool:
        try:
            response = self._client.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except (self._httpx.HTTPError, ValueError):
            return False
        # Ollama reports "nomic-embed-text:latest" for a bare "nomic-embed-text".
        return any(
            n == self.model_name or n.split(":")[0] == self.model_name.split(":")[0]
            for n in names
        )

    def prepare(self) -> None:
        """Stream a model pull, logging progress. Idempotent."""
        if self.has_model():
            log.info("embedding model %s already present", self.model_name)
            return
        log.info("pulling embedding model %s (first boot only)...", self.model_name)
        with self._client.stream(
            "POST", f"{self.base_url}/api/pull", json={"model": self.model_name}, timeout=None
        ) as response:
            response.raise_for_status()
            last = ""
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    status = json.loads(line).get("status", "")
                except ValueError:
                    continue
                if status and status != last:
                    log.info("  pull: %s", status)
                    last = status
        if not self.has_model():
            raise EmbeddingError(f"model {self.model_name} still missing after pull")
        log.info("embedding model %s ready", self.model_name)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        # /api/embed (batch) is current; /api/embeddings is the legacy
        # single-input endpoint kept for older servers.
        try:
            response = self._client.post(
                f"{self.base_url}/api/embed", json={"model": self.model_name, "input": texts}
            )
            if response.status_code == 404:
                raise self._httpx.HTTPStatusError(
                    "no /api/embed", request=response.request, response=response
                )
            response.raise_for_status()
            vectors = response.json().get("embeddings")
            if vectors:
                return vectors
        except self._httpx.HTTPStatusError:
            pass

        vectors = []
        for text in texts:
            response = self._client.post(
                f"{self.base_url}/api/embeddings", json={"model": self.model_name, "prompt": text}
            )
            response.raise_for_status()
            vector = response.json().get("embedding")
            if not vector:
                raise EmbeddingError("ollama returned an empty embedding")
            vectors.append(vector)
        return vectors

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._embed_batch(texts[i : i + batch_size]))
        if out and self._dim is None:
            self._dim = len(out[0])
        return out

    def embed_query(self, text: str) -> list[float]:
        # Ollama applies no asymmetric prefix, so query and document embedding
        # are the same call here.
        return self.embed_documents([text], batch_size=1)[0]

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim


class HttpEmbedder:
    """An OpenAI-shaped embeddings endpoint - a hosted or internal service.

    `POST {base}/embeddings` with `{"model": ..., "input": [...]}`, replying
    with `{"data": [{"index": n, "embedding": [...]}]}`. Nearly every internal
    AI gateway speaks this, whatever is behind it, which is why the shape is
    the interface rather than any particular vendor.

    Three things here are not obvious and all three are correctness rather than
    polish:

    **The reply is sorted by index before use.** The `index` field exists
    because responses are not guaranteed to arrive in request order, and a
    server that returns them shuffled would attach every vector to the wrong
    chunk. Nothing would error - search would simply return nonsense, and it
    would take a long time to work out why.

    **The prompt prefix is configurable and matters.** Strong retrieval models
    are trained asymmetrically: nomic wants "search_document: " on passages and
    "search_query: " on questions, e5 wants "passage: " and "query: ".
    `fastembed` applies these itself; a raw HTTP endpoint does not, so pointing
    this at the same model without prefixes measurably degrades ranking while
    looking like it works. Set them to match whatever the endpoint serves.

    **Batches are capped and retried.** Gateways limit inputs per request and
    rate-limit bursts, neither of which a local model does, so a corpus ingest
    that hammers one is the case this has to survive rather than the exception.
    """

    backend = "http"

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 auth_header: str = "Authorization", auth_prefix: str = "Bearer",
                 doc_prefix: str = "", query_prefix: str = "",
                 batch_limit: int = 64, timeout: float = 120.0,
                 verify: bool = True) -> None:
        import httpx

        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.batch_limit = max(1, batch_limit)
        headers = {"content-type": "application/json"}
        if api_key:
            headers[auth_header] = f"{auth_prefix} {api_key}".strip()
        # verify=False exists for an internal endpoint behind a corporate CA
        # that the machine does not trust yet. It is a real situation and a bad
        # default, so it is opt-in and named plainly.
        self._client = httpx.Client(timeout=timeout, headers=headers, verify=verify)
        self._dim: int | None = None

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/embeddings"

    def alive(self) -> bool:
        try:
            self._embed_batch(["ping"])
            return True
        except Exception:
            return False

    def wait_until_alive(self, timeout: float = 300.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.alive():
                return True
            time.sleep(3.0)
        return False

    def prepare(self) -> None:
        # Nothing to download; a failure now is a clearer message than the same
        # failure part-way through an ingest.
        if not self.alive():
            raise EmbeddingError(
                f"no embeddings endpoint answering at {self.endpoint} - check "
                f"RAG_EMBED_URL, the model name, and the API key"
            )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        body = {"model": self.model_name, "input": texts}
        last: Exception | None = None
        for attempt in range(4):
            try:
                response = self._client.post(self.endpoint, json=body)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise self._httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request, response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                break
            except (self._httpx.HTTPError, ValueError) as exc:
                last = exc
                if attempt == 3:
                    raise EmbeddingError(
                        f"{self.endpoint} failed after 4 attempts: {exc}"
                    ) from exc
                time.sleep(1.5 * (2 ** attempt))
        else:  # pragma: no cover - the loop always breaks or raises
            raise EmbeddingError(str(last))

        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            raise EmbeddingError(f"{self.endpoint} returned no embeddings")
        # Sorted by the index the server reports, not by arrival. See the class
        # docstring: out-of-order replies would silently mis-pair every vector.
        try:
            rows = sorted(rows, key=lambda row: int(row.get("index", 0)))
        except (TypeError, ValueError):
            pass
        vectors = [row.get("embedding") for row in rows]
        if any(not isinstance(v, list) or not v for v in vectors):
            raise EmbeddingError(f"{self.endpoint} returned a malformed embedding")
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"asked {self.endpoint} for {len(texts)} embeddings and got "
                f"{len(vectors)} - refusing to guess which text each belongs to"
            )
        return [list(map(float, v)) for v in vectors]

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        width = min(batch_size, self.batch_limit)
        prefixed = [f"{self.doc_prefix}{text}" for text in texts]
        out: list[list[float]] = []
        for start in range(0, len(prefixed), width):
            out.extend(self._embed_batch(prefixed[start:start + width]))
        if out and self._dim is None:
            self._dim = len(out[0])
        return out

    def embed_query(self, text: str) -> list[float]:
        vector = self._embed_batch([f"{self.query_prefix}{text}"])[0]
        if self._dim is None:
            self._dim = len(vector)
        return vector

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed_query("dimension probe")
        assert self._dim is not None
        return self._dim


def make_embedder(cfg):
    if cfg.embed_backend == "ollama":
        return OllamaEmbedder(cfg.ollama_url, cfg.embed_model)
    if cfg.embed_backend in ("http", "openai"):
        if not cfg.embed_url:
            raise EmbeddingError(
                "RAG_EMBED_BACKEND=http needs RAG_EMBED_URL - the base URL of an "
                "OpenAI-shaped embeddings service, e.g. https://ai.internal/v1"
            )
        return HttpEmbedder(
            cfg.embed_url, cfg.embed_model,
            api_key=cfg.embed_api_key,
            auth_header=cfg.embed_auth_header,
            auth_prefix=cfg.embed_auth_prefix,
            doc_prefix=cfg.embed_doc_prefix,
            query_prefix=cfg.embed_query_prefix,
            batch_limit=cfg.embed_http_batch,
            timeout=cfg.embed_http_timeout,
            verify=cfg.embed_verify_tls,
        )
    if cfg.embed_backend == "fastembed":
        return FastEmbedEmbedder(
            cfg.embed_model, cfg.model_cache, getattr(cfg, "embed_threads", 0),
            gpu=accel.gpu_wanted(cfg), gpu_setting=accel.gpu_setting_name(),
        )
    raise EmbeddingError(
        f"unknown RAG_EMBED_BACKEND={cfg.embed_backend!r} "
        f"(expected 'fastembed', 'ollama' or 'http')"
    )
