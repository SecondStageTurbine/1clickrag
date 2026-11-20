#!/usr/bin/env bash
# Offline native launcher. Does not use or change the OS Python environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RAG_MODE=native RAG_DARK_NETWORK=1
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
PY="$PWD/.venv-rhel/bin/python"
case "${1:-run}" in
  install)
    SYSTEM_PY="${RAG_PYTHON:-/usr/bin/python3.12}"
    "$SYSTEM_PY" -c 'import platform,sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"; assert platform.system() == "Linux" and platform.machine() == "x86_64", "Linux x86_64 required"'
    sha256sum --check --quiet SHA256SUMS
    if [ -e .venv-rhel ]; then
      echo 'Existing .venv-rhel found. Use a fresh extracted release directory for installation.' >&2
      exit 1
    fi
    "$SYSTEM_PY" -m venv .venv-rhel
    "$PY" -m pip install --no-index --find-links vendor/wheels --upgrade pip
    # The frozen set intentionally uses GPU ORT and headless OpenCV in place
    # of CPU ORT and GUI OpenCV. Upstream dependency names still name the latter.
    "$PY" -m pip install --no-index --find-links vendor/wheels --no-deps -r vendor/requirements-rhel.lock
    [ -f .env ] || cp deploy/rhel/rhel.env.example .env
    echo 'Installed. Edit .env, then run: bash deploy/rhel/rag-rhel.sh check'
    exit 0
    ;;
  run|check|check-cpu|benchmark|probe) ;;
  *) echo 'Usage: bash deploy/rhel/rag-rhel.sh install|check|check-cpu|benchmark|probe|run' >&2; exit 2 ;;
esac
[ -x "$PY" ] || { echo 'Run install first.' >&2; exit 1; }
# Linux's loader must see the wheel-provided CUDA shared libraries at startup.
CUDA_LIBS="$("$PY" -c 'import pathlib,sysconfig; p=pathlib.Path(sysconfig.get_paths()["purelib"]); print(":".join(str(x) for x in sorted((p/"nvidia").glob("*/lib"))))')"
export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-run}" in
  check) exec "$PY" deploy/rhel/check.py ;;
  probe)
    # Why did check fall back to CPU? Everything the answer usually hides in.
    nvidia-smi -L 2>&1 || echo 'nvidia-smi not found: NVIDIA driver not installed or not on PATH'
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
    ls -l /dev/nvidia* 2>&1 | head -3
    CUDA_EP="$("$PY" -c 'import onnxruntime,os; print(os.path.dirname(onnxruntime.__file__))')/capi/libonnxruntime_providers_cuda.so"
    ldd "$CUDA_EP" | grep -i 'not found' || echo 'all CUDA provider libraries resolve'
    "$PY" deploy/rhel/check.py 2>&1 | grep -iE 'provider_bridge|Failed to create|fell back|insufficient|not found|PASS|Error' || true
    ;;
  check-cpu) RAG_GPU=0 exec "$PY" deploy/rhel/check.py --cpu ;;
  benchmark) exec "$PY" -m app.gpucheck ;;
  run) exec "$PY" -m app.server ;;
esac
