#!/usr/bin/env bash
# Run in a connected RHEL/Rocky 9 x86_64 Python 3.12 staging container.
set -euo pipefail
cd /bundle
python3.12 -m venv /tmp/rag-build-venv
PY=/tmp/rag-build-venv/bin/python
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements-native.txt
# Replace packages sharing the same import directories; never install both.
"$PY" -m pip uninstall -y onnxruntime opencv-python
"$PY" -m pip install 'onnxruntime-gpu[cuda,cudnn]==1.28.0' \
  'nvidia-cuda-runtime==13.0.96' 'nvidia-cuda-nvrtc==13.0.88' \
  'nvidia-cublas==13.0.2.14' 'nvidia-cufft==12.0.0.61' \
  'nvidia-curand==10.4.0.35' 'nvidia-nvjitlink==13.0.88' \
  'nvidia-cudnn-cu13==9.13.1.26' 'opencv-python-headless==4.12.0.88' 'numpy==2.2.6'  # opencv 4.12 needs numpy<2.3; 4.13+ bundles a Red Hat FIPS-patched libcrypto 1.1.1k that aborts on FIPS hosts (opencv-python#1191)
mkdir -p vendor/wheels
"$PY" -m pip freeze > vendor/requirements-rhel.lock
"$PY" -m pip wheel --no-deps -r vendor/requirements-rhel.lock -w vendor/wheels
"$PY" -m pip download --no-deps --only-binary=:all: pip -d vendor/wheels
"$PY" -m pip list --format=json > vendor/packages.json
cat /etc/os-release > vendor/build-os.txt
python3.12 --version >> vendor/build-os.txt
ldd --version | head -1 >> vendor/build-os.txt
