# SPDX-License-Identifier: MPL-2.0
"""Identity of the embedding configuration, shared by cache and manifest."""

import hashlib
import json
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version


@lru_cache(maxsize=None)
def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def fingerprint(cfg) -> str:
    backend = cfg.embed_backend
    if backend == "openai":
        backend = "http"
    contract = {
        "format": 1,
        "backend": backend,
        "model": cfg.embed_model,
        "revision": getattr(cfg, "embed_revision", ""),
    }
    if backend == "http":
        contract.update(
            endpoint=cfg.embed_url.rstrip("/"),
            document_prefix=cfg.embed_doc_prefix,
            query_prefix=cfg.embed_query_prefix,
        )
    elif backend == "ollama":
        contract["endpoint"] = cfg.ollama_url.rstrip("/")
    elif backend == "fastembed":
        # FastEmbed owns tokenization, pooling, normalization, and prefixes.
        contract["fastembed_version"] = _version("fastembed")
    # Persist only the digest: endpoint URLs may contain credentials.
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
