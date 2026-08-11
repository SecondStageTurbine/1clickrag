# SPDX-License-Identifier: MPL-2.0
"""Proving the embedding model is actually on the GPU.

    python -m app.gpucheck

The obvious check is the one that lies. `onnxruntime.get_available_providers()`
reporting CUDAExecutionProvider means only that a library is present on disk;
creating a session can still fall back to CPU, and it does so with a warning
rather than an error. A benchmark run in that state reports a small speedup -
1.1x, measured here - which is CPU against CPU and pure noise. Anyone reading
that number concludes the GPU works and stops looking.

So this asks for evidence a CPU-only run cannot fabricate, in rising order of
how hard it is to fake:

1. Which provider the *loaded session* chose, rather than which are installed.
2. Whether GPU memory actually moved while the model loaded and ran.
3. Whether it is faster, warmed up, on identical input.
4. Whether the vectors match the CPU's - because a fast wrong answer is not a
   result, and a silently different embedding would poison an index in a way
   no error message would ever mention.

And a negative control: the same probe is run with CPU forced, and must report
CPU. A check that passes no matter what proves nothing, so this one is made to
fail on demand before its success is believed.

Nothing here is required to run the RAG. It exists because "is it on the GPU"
turned out to be a question that needs an instrument rather than an opinion.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time

ROWS = [
    f"Failure mode {index}: intermittent loss of signal on channel {index % 8}, "
    f"detected by continuity check during power-on self test"
    for index in range(400)
]


def gpu_memory_used() -> list[int]:
    """MiB in use per GPU, or an empty list when nvidia-smi is absent."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [int(line) for line in out.stdout.split() if line.strip().isdigit()]


def gpu_names() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def session_provider(model) -> str:
    """The provider the loaded session is really using.

    fastembed wraps the session a couple of layers deep and the exact shape has
    moved between versions, so this walks rather than assuming a path.
    """
    seen = set()
    stack = [model]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        getter = getattr(node, "get_providers", None)
        if callable(getter):
            try:
                providers = getter()
            except Exception:
                providers = None
            if providers:
                return providers[0]
        for attribute in ("model", "session", "_session", "embedding_model", "_model"):
            child = getattr(node, attribute, None)
            if child is not None:
                stack.append(child)
    return "unknown"


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def run_once(providers: list[str], model_name: str, cache_dir: str,
             batch_size: int) -> dict:
    """Load the model under one provider list and measure it."""
    from fastembed import TextEmbedding

    before = gpu_memory_used()
    started = time.time()
    model = TextEmbedding(model_name=model_name, cache_dir=cache_dir,
                          providers=providers)
    # Warm up: the first call pays for graph optimisation and, on CUDA, kernel
    # autotuning. Timing that instead of the work is how a GPU is made to look
    # slower than a CPU.
    list(model.embed(ROWS[:16], batch_size=16))
    load_seconds = time.time() - started

    resident = gpu_memory_used()

    started = time.time()
    vectors = [list(map(float, v)) for v in model.embed(ROWS, batch_size=batch_size)]
    elapsed = time.time() - started
    during = gpu_memory_used()

    delta = 0
    if before and resident:
        delta = max(r - b for r, b in zip(resident, before))
    peak = 0
    if before and during:
        peak = max(d - b for d, b in zip(during, before))

    return {
        "provider": session_provider(model),
        "load_seconds": round(load_seconds, 2),
        "seconds": round(elapsed, 3),
        "ms_per_row": round(elapsed / len(ROWS) * 1000, 3),
        "gpu_memory_delta_mib": max(delta, peak),
        "dim": len(vectors[0]) if vectors else 0,
        "vectors": vectors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        from .config import CONFIG
        model_name = args.model or CONFIG.embed_model
        cache_dir = args.cache or CONFIG.model_cache
    except Exception:
        model_name = args.model or "nomic-ai/nomic-embed-text-v1.5"
        cache_dir = args.cache or ".data/models"

    report: dict = {"model": model_name, "checks": {}}
    say = (lambda *a: None) if args.json else print

    say(f"\n  model {model_name}\n")

    # 1 - what is installed
    try:
        import onnxruntime as ort
    except ImportError:
        say("  onnxruntime is not installed")
        return 2
    available = ort.get_available_providers()
    report["onnxruntime_version"] = ort.__version__
    report["providers_available"] = available
    say(f"  1. onnxruntime .............. {ort.__version__}")
    say(f"     providers offered ........ {', '.join(available)}")

    cards = gpu_names()
    report["gpus"] = cards
    if cards:
        for card in cards:
            say(f"     gpu ...................... {card}")
    else:
        say("     gpu ...................... nvidia-smi found nothing")

    if "CUDAExecutionProvider" not in available:
        say("\n  CUDAExecutionProvider is not even offered by this build.")
        say("  Install onnxruntime-gpu, which REPLACES onnxruntime:")
        say("      pip uninstall -y onnxruntime")
        say("      pip install --force-reinstall onnxruntime-gpu")
        say("  and match the CUDA major version the wheel expects.")
        report["verdict"] = "no CUDA provider in this onnxruntime build"
        if args.json:
            print(json.dumps({k: v for k, v in report.items()}, indent=2))
        return 1

    # 2 - the negative control, first. A check that cannot fail proves nothing,
    # so establish that this probe reports CPU when CPU is what is running,
    # before trusting it to report CUDA.
    say("\n  2. negative control (CPU forced)")
    cpu = run_once(["CPUExecutionProvider"], model_name, cache_dir, args.batch_size)
    say(f"     session provider ......... {cpu['provider']}")
    say(f"     gpu memory moved ......... {cpu['gpu_memory_delta_mib']} MiB")
    say(f"     speed .................... {cpu['ms_per_row']} ms/row")
    control_ok = cpu["provider"] == "CPUExecutionProvider"
    report["checks"]["negative_control"] = control_ok
    if not control_ok:
        say(f"     -> this probe cannot tell the two apart; nothing below is evidence")
        report["verdict"] = "probe is unreliable - CPU run did not report CPU"
        if args.json:
            print(json.dumps({k: v for k, v in report.items() if k != "vectors"}, indent=2))
        return 1

    # 3 - the real thing
    say("\n  3. CUDA requested")
    gpu = run_once(["CUDAExecutionProvider", "CPUExecutionProvider"],
                   model_name, cache_dir, args.batch_size)
    say(f"     session provider ......... {gpu['provider']}")
    say(f"     gpu memory moved ......... {gpu['gpu_memory_delta_mib']} MiB")
    say(f"     model load ............... {gpu['load_seconds']}s")
    say(f"     speed .................... {gpu['ms_per_row']} ms/row")

    on_gpu = gpu["provider"] == "CUDAExecutionProvider"
    moved = gpu["gpu_memory_delta_mib"] >= 64
    speedup = cpu["seconds"] / gpu["seconds"] if gpu["seconds"] else 0.0
    agreement = min(
        cosine(a, b) for a, b in zip(cpu["vectors"][:50], gpu["vectors"][:50])
    ) if cpu["vectors"] and gpu["vectors"] else 0.0

    say(f"\n  4. speed ..................... {speedup:.1f}x "
        f"(cpu {cpu['ms_per_row']} -> gpu {gpu['ms_per_row']} ms/row)")
    say(f"  5. vectors agree ............. min cosine {agreement:.6f} over 50 rows")

    report["checks"].update({
        "session_used_cuda": on_gpu,
        "gpu_memory_moved": moved,
        "speedup": round(speedup, 2),
        "min_cosine_vs_cpu": round(agreement, 6),
    })
    report["cpu"] = {k: v for k, v in cpu.items() if k != "vectors"}
    report["gpu"] = {k: v for k, v in gpu.items() if k != "vectors"}

    problems = []
    if not on_gpu:
        problems.append(
            f"the session fell back to {gpu['provider']} - onnxruntime logs the "
            f"reason above, usually a missing CUDA library or a wheel built for "
            f"a different CUDA major version"
        )
    if on_gpu and not moved:
        problems.append(
            f"the session claims CUDA but GPU memory barely moved "
            f"({gpu['gpu_memory_delta_mib']} MiB) - suspicious"
        )
    if agreement < 0.999:
        problems.append(
            f"GPU vectors differ from CPU ones (min cosine {agreement:.4f}) - "
            f"an index built on one and queried on the other would rank badly"
        )
    if on_gpu and moved and speedup < 1.2:
        problems.append(
            f"it is on the GPU but no faster ({speedup:.2f}x) - the batch may be "
            f"too small to cover the transfer, try --batch-size 256"
        )

    say("")
    if problems:
        say("  NOT PROVEN:")
        for problem in problems:
            say(f"    - {problem}")
        report["verdict"] = "; ".join(problems)
    else:
        say(f"  PROVEN: embeddings ran on {gpu['provider']}, "
            f"{gpu['gpu_memory_delta_mib']} MiB of GPU memory in use, "
            f"{speedup:.1f}x faster, vectors identical to CPU.")
        report["verdict"] = "confirmed"
    say("")

    if args.json:
        print(json.dumps(report, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
