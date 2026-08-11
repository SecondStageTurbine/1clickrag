# SPDX-License-Identifier: MPL-2.0
"""Choosing an ONNX execution provider, and reporting the one actually chosen.

Both neural models here run on onnxruntime - the embedder and the cross-encoder
reranker - and both want the same two rules: never ask for CUDA without CPU
behind it, and never trust the request as a description of what happened. The
rules are subtle enough that a second copy would drift from the first, so they
live here and both callers import them.

The second rule is the one that matters. Asking for CUDAExecutionProvider and
getting it are different events: onnxruntime falls through the provider list and
logs a warning rather than raising, so a session that quietly landed on the CPU
looks exactly like one that did not. A 1.1x "speedup" measured during
development was CPU against CPU for precisely this reason. Anything reporting a
provider must therefore ask the loaded session, not the configuration.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("rag.accel")

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"

# RAG_EMBED_GPU governed only the embedder before v1.10.0. It still means "use
# the GPU", so it is honoured rather than dropped.
OLD_SETTING = "RAG_EMBED_GPU"
SETTING = "RAG_GPU"


def gpu_wanted(cfg) -> bool:
    """Whether this configuration asked for the GPU, under either name."""
    return bool(getattr(cfg, "gpu", getattr(cfg, "embed_gpu", False)))


def gpu_setting_name() -> str:
    """The variable the user actually set, so warnings name a name they typed."""
    if os.environ.get(SETTING) is not None:
        return SETTING
    if os.environ.get(OLD_SETTING) is not None:
        return OLD_SETTING
    return SETTING


def preload() -> bool:
    """Make the pip-installed CUDA libraries findable before a session is built.

    onnxruntime-gpu does not bundle CUDA or cuDNN; the documented way to supply
    them is `pip install onnxruntime-gpu[cuda,cudnn]`, which puts the DLLs in
    site-packages/nvidia/*/bin. On Windows that directory is not on the DLL
    search path, so the libraries end up present, correct, and invisible - and
    onnxruntime responds by logging "Failed to create CUDAExecutionProvider.
    Require cuDNN 9.* and CUDA 13.*" and quietly running on the CPU.

    That failure deserves spelling out because from the outside it is
    indistinguishable from having no GPU at all: the provider is still offered,
    the session is still created, and every number reads 1.0x. It cost a full
    round of "install the libraries, still on CPU" here before
    onnxruntime.preload_dlls() turned out to be the entire difference.

    Older builds have no preload_dlls, and a machine using a system-wide CUDA
    does not need it, so failing here is not an error - it just leaves the DLL
    search path as whatever it already was.
    """
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - fastembed brings onnxruntime
        return False
    loader = getattr(ort, "preload_dlls", None)
    if not callable(loader):
        return False
    try:
        loader()
    except Exception as exc:  # pragma: no cover - depends on the wheel
        log.debug("onnxruntime.preload_dlls() failed: %s", exc)
        return False
    return True


def providers(setting: str = "RAG_GPU") -> list[str]:
    """CUDA first, CPU behind it - never CUDA alone.

    onnxruntime walks the list in order, so keeping CPU at the end means a
    machine without a working CUDA build gets a slower run instead of a server
    that will not start. Asking for the GPU is a preference about speed; it
    should not become a hard dependency on a driver.

    `setting` names the variable in the warning, so someone who set the older
    RAG_EMBED_GPU is told about the name they actually used.
    """
    # Before asking what is available, make sure the answer can be acted on:
    # get_available_providers() reports CUDA from the wheel's own metadata, so
    # it says yes whether or not the libraries it needs can be found.
    preload()

    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except ImportError:  # pragma: no cover - fastembed brings onnxruntime
        return [CPU]

    if CUDA not in available:
        log.warning(
            "%s is set but this onnxruntime has no CUDA provider (it offers %s). "
            "Install onnxruntime-gpu, which REPLACES the CPU build - falling "
            "back to CPU for now.",
            setting,
            ", ".join(available),
        )
        return [CPU]
    return [CUDA, CPU]


def chosen(gpu: bool, setting: str = SETTING) -> list[str]:
    """The provider list to build a session with, for either answer.

    The `gpu=False` half is not "pass nothing and let onnxruntime decide",
    because on an onnxruntime-gpu build the default is CUDA first - so leaving
    the argument off means the GPU is used whether or not it was asked for, and
    RAG_GPU=0 fails to mean anything. Measured here: with the flag off, both
    models still loaded onto CUDA.

    Being explicit in both directions matters when something else wants the
    card. A local LLM sharing the GPU is the usual case, and these two models
    take about 2.7 GB between them - which is VRAM the generator does not get,
    silently, on the strength of a setting the user believed was off.
    """
    return providers(setting) if gpu else [CPU]


def active(model, requested_gpu: bool = False) -> str:
    """The provider the loaded session actually chose.

    fastembed wraps the session two attributes deep and the exact path has
    changed between versions, so this walks the plausible names rather than
    hardcoding one. Both TextEmbedding and TextCrossEncoder currently answer at
    `.model.model`; the walk costs nothing and survives the next rename.

    Falls back to the request only when no session can be found, which is the
    one case where an honest answer is unavailable.
    """
    seen = set()
    queue = [model]
    while queue:
        obj = queue.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        getter = getattr(obj, "get_providers", None)
        if callable(getter):
            try:
                chosen = getter()
            except Exception:  # pragma: no cover - a session that cannot answer
                continue
            if chosen:
                return chosen[0]
        for attribute in ("model", "session", "_session", "embedding_model"):
            queue.append(getattr(obj, attribute, None))
    return CUDA if requested_gpu else CPU
