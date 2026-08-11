# SPDX-License-Identifier: MPL-2.0
"""Proving the ONNX models are actually on the GPU.

    python -m app.gpucheck                  # both models
    python -m app.gpucheck --what rerank    # just the cross-encoder

Both are checked, and the reranker is the one to read first. It is the model
that decides how long a search takes - one forward pass per retrieved candidate
against the embedder's single short one - so a report proving the embedder is
on CUDA while the reranker fell back to CPU is a green light on the wrong lamp.
That is not hypothetical: the reranker had no provider wiring at all until
v1.10.0, which made "the GPU does nothing for query latency" look like a fact
about hardware rather than a missing argument.

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

from . import accel

ROWS = [
    f"Failure mode {index}: intermittent loss of signal on channel {index % 8}, "
    f"detected by continuity check during power-on self test"
    for index in range(400)
]

# The reranker reads a query and a passage together, so it needs both, and the
# passage has to be the length of a real chunk: cross-encoder cost scales with
# the pair's token count, and measuring it on one-line strings would understate
# it by roughly the ratio of the lengths.
QUESTION = "which failure modes are detectable by built-in test?"
PASSAGE = (
    "Failure mode {index}: intermittent loss of signal on channel {index}, "
    "detected by continuity check during power-on self test. The built-in test "
    "asserts a fault flag when the measured continuity falls outside the "
    "qualification limits recorded in the acceptance data package, and the "
    "maintenance manual directs the technician to the connector backshell "
    "before the line replaceable unit itself. Applicability is determined by "
    "whether the monitoring circuit is powered in the operational mode under "
    "which the failure manifests, which for this channel it is. "
) * 2


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


def run_rerank_once(providers: list[str], model_name: str, cache_dir: str,
                    candidates: int) -> dict:
    """Load the cross-encoder under one provider list and measure it.

    Measured per candidate rather than per query, because that is the shape of
    the cost: one forward pass per passage retrieved, so a search reranking 150
    candidates pays this 150 times while the embedder pays once.
    """
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    passages = [PASSAGE.format(index=index) for index in range(candidates)]

    before = gpu_memory_used()
    started = time.time()
    model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir,
                             providers=providers)
    # Same warm-up reasoning as the embedder: the first call pays for graph
    # optimisation and CUDA kernel autotuning, and timing that instead of the
    # work is how a GPU is made to look slower than a CPU.
    list(model.rerank(QUESTION, passages[:8]))
    load_seconds = time.time() - started

    resident = gpu_memory_used()

    started = time.time()
    scores = [float(s) for s in model.rerank(QUESTION, passages)]
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
        "ms_per_candidate": round(elapsed / len(passages) * 1000, 3),
        "candidates": len(passages),
        "gpu_memory_delta_mib": max(delta, peak),
        "scores": scores,
    }


def check_rerank(args, cache_dir: str, say) -> tuple[dict, list[str]]:
    """Negative control, then CUDA, then a verdict - for the cross-encoder."""
    try:
        from .config import CONFIG
        model_name = args.rerank_model or CONFIG.rerank_model
        candidates = args.candidates or CONFIG.rerank_candidates
    except Exception:
        model_name = args.rerank_model or "Xenova/ms-marco-MiniLM-L-6-v2"
        candidates = args.candidates or 40

    report: dict = {"model": model_name, "candidates": candidates, "checks": {}}
    say(f"\n  RERANKER  {model_name}  ({candidates} candidates)")

    say("\n  1. negative control (CPU forced)")
    cpu = run_rerank_once(["CPUExecutionProvider"], model_name, cache_dir, candidates)
    say(f"     session provider ......... {cpu['provider']}")
    say(f"     speed .................... {cpu['ms_per_candidate']} ms/candidate"
        f"  ({cpu['seconds']}s per query)")
    if cpu["provider"] != "CPUExecutionProvider":
        report["verdict"] = "probe is unreliable - CPU run did not report CPU"
        say("     -> this probe cannot tell the two apart; nothing below is evidence")
        return report, [report["verdict"]]
    report["checks"]["negative_control"] = True

    say(f"\n  2. CUDA requested (device {args.device})")
    gpu = run_rerank_once([("CUDAExecutionProvider", {"device_id": args.device}),
                           "CPUExecutionProvider"],
                          model_name, cache_dir, candidates)
    say(f"     session provider ......... {gpu['provider']}")
    say(f"     gpu memory moved ......... {gpu['gpu_memory_delta_mib']} MiB")
    say(f"     speed .................... {gpu['ms_per_candidate']} ms/candidate"
        f"  ({gpu['seconds']}s per query)")

    on_gpu = gpu["provider"] == "CUDAExecutionProvider"
    moved = gpu["gpu_memory_delta_mib"] >= 16
    speedup = cpu["seconds"] / gpu["seconds"] if gpu["seconds"] else 0.0
    # Scores are logits, not a similarity, so they are compared directly. What
    # actually matters is the order they impose - a reranker that agrees to
    # three decimal places but sorts differently has changed the answer.
    drift = max((abs(a - b) for a, b in zip(cpu["scores"], gpu["scores"])), default=0.0)
    order_cpu = sorted(range(len(cpu["scores"])), key=lambda i: -cpu["scores"][i])
    order_gpu = sorted(range(len(gpu["scores"])), key=lambda i: -gpu["scores"][i])
    same_order = order_cpu[:10] == order_gpu[:10]

    say(f"\n  3. speed ..................... {speedup:.1f}x "
        f"({cpu['seconds']}s -> {gpu['seconds']}s per query)")
    say(f"  4. scores agree .............. max drift {drift:.6f}, "
        f"top-10 order {'identical' if same_order else 'DIFFERENT'}")

    report["checks"].update({
        "session_used_cuda": on_gpu,
        "gpu_memory_moved": moved,
        "speedup": round(speedup, 2),
        "max_score_drift": round(drift, 6),
        "same_top10_order": same_order,
    })
    report["cpu"] = {k: v for k, v in cpu.items() if k != "scores"}
    report["gpu"] = {k: v for k, v in gpu.items() if k != "scores"}

    problems = []
    if not on_gpu:
        problems.append(
            f"the reranker session fell back to {gpu['provider']} - this is the "
            f"model that dominates query latency, so this is the fall back that "
            f"costs the most"
            + (f", and device {args.device} does not exist on this host "
               f"({len(gpu_names())} GPU(s) found), which is enough on its own"
               if args.device >= max(len(gpu_names()), 1) else "")
        )
    if on_gpu and not moved:
        problems.append(
            f"the reranker claims CUDA but GPU memory barely moved "
            f"({gpu['gpu_memory_delta_mib']} MiB) - suspicious"
        )
    if not same_order:
        problems.append(
            f"GPU reranking orders the candidates differently (max score drift "
            f"{drift:.4f}) - the same query would return different passages"
        )
    if on_gpu and moved and speedup < 1.2:
        problems.append(
            f"the reranker is on the GPU but no faster ({speedup:.2f}x) - try a "
            f"larger --candidates, the batch may be too small to cover transfer"
        )
    report["verdict"] = "; ".join(problems) if problems else "confirmed"
    return report, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=None)
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=None,
                        help="passages to rerank (default: RAG_RERANK_CANDIDATES)")
    parser.add_argument("--what", choices=("embed", "rerank", "both"), default="both",
                        help="which model to prove (default: both)")
    parser.add_argument("--device", type=int, default=None,
                        help="CUDA device to probe (default: RAG_GPU_DEVICE)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        from .config import CONFIG
        model_name = args.model or CONFIG.embed_model
        cache_dir = args.cache or CONFIG.model_cache
        device = args.device if args.device is not None else CONFIG.gpu_device
    except Exception:
        model_name = args.model or "nomic-ai/nomic-embed-text-v1.5"
        cache_dir = args.cache or ".data/models"
        device = args.device or 0
    args.device = device

    report: dict = {"model": model_name, "checks": {}}
    say = (lambda *a: None) if args.json else print

    if args.what != "rerank":
        say(f"\n  embedding model {model_name}\n")
    else:
        say("")

    # 1 - what is installed
    try:
        import onnxruntime as ort
    except ImportError:
        say("  onnxruntime is not installed")
        return 2
    # The same call the server makes, for the same reason: pip-installed CUDA
    # libraries sit outside the Windows DLL search path, and without this the
    # probe would faithfully measure a fall back that only exists because the
    # probe forgot to look where the libraries are.
    preloaded = accel.preload()
    report["preloaded_dlls"] = preloaded
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

    if args.what == "rerank":
        rerank_report, rerank_problems = check_rerank(args, cache_dir, say)
        report["rerank"] = rerank_report
        say("")
        if rerank_problems:
            say("  NOT PROVEN:")
            for problem in rerank_problems:
                say(f"    - {problem}")
        else:
            say("  PROVEN: reranking ran on CUDA, scores and order identical to CPU.")
        say("")
        if args.json:
            print(json.dumps(report, indent=2))
        return 1 if rerank_problems else 0

    # 2 - the negative control, first. A check that cannot fail proves nothing,
    # so establish that this probe reports CPU when CPU is what is running,
    # before trusting it to report CUDA.
    say("\n  EMBEDDER")
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
    say(f"\n  3. CUDA requested (device {args.device})")
    gpu = run_once([("CUDAExecutionProvider", {"device_id": args.device}),
                    "CPUExecutionProvider"],
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
        say("  embedder NOT PROVEN:")
        for problem in problems:
            say(f"    - {problem}")
        report["verdict"] = "; ".join(problems)
    else:
        say(f"  embedder PROVEN: embeddings ran on {gpu['provider']}, "
            f"{gpu['gpu_memory_delta_mib']} MiB of GPU memory in use, "
            f"{speedup:.1f}x faster, vectors identical to CPU.")
        report["verdict"] = "confirmed"

    # The reranker too, and not as an afterthought: it is the model whose
    # provider a user actually feels. Proving only the embedder is how the
    # original mistake was made - a green report about the cheap model, while
    # the expensive one quietly ran on the CPU.
    if args.what == "both":
        rerank_report, rerank_problems = check_rerank(args, cache_dir, say)
        report["rerank"] = rerank_report
        say("")
        if rerank_problems:
            say("  reranker NOT PROVEN:")
            for problem in rerank_problems:
                say(f"    - {problem}")
        else:
            say(f"  reranker PROVEN: {rerank_report['candidates']} candidates on "
                f"CUDA, {rerank_report['checks']['speedup']}x faster, order "
                f"identical to CPU.")
        problems = problems + rerank_problems
    say("")

    if args.json:
        print(json.dumps(report, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
