# RHEL package validation — 2026-09-08

Validated on a local x86_64 Rocky Linux 9 staging container with Python
3.12.14, glibc 2.34, SQLite 3.34.1 and NVIDIA GPU passthrough. Container
networking was disabled (`--network none`) throughout installation and tests.
The container is a staging/test tool only; deployment on RHEL is native Python.

Completed successfully:

- Offline installation from 90 bundled wheels into a fresh virtual environment,
  starting with pip 23.2.1 (the target's reported pip version).
- Dependency version validation with explicit GPU ONNX Runtime and headless
  OpenCV replacements.
- Staged embedding and reranking inference on CPU.
- Staged embedding and reranking inference on CUDA device 0, verified from the
  actual loaded sessions. Local GPU: RTX 5090, driver 616.64.
- OCR execution, PDF page rendering and local Qdrant insert/query.
- Offline application startup and indexing of a synthetic document, followed
  by search, citation preview, local UI asset requests and traversal rejection.
- All 50 existing app tests on Linux with SQLite 3.34.1.
- All 50 existing app tests on the Windows development environment after the
  SQLite compatibility fix.
- Bash syntax checks for the installer/launcher and dependency build script.

The tests found and fixed graph indexing's use of SQL RETURNING, which is
unavailable in RHEL 9's SQLite 3.34. Entity insert/update now uses a separate
lookup inside the existing transaction/lock. The delivered app includes this fix.
Model-cache metadata paths were normalized from Windows separators to Linux
separators; model weights were not changed.

Not yet verified: execution on the user's actual RHEL 9.8 / driver 580.178.04 /
eight-H200 server, production-corpus indexing, or a future Ollama/llama.cpp
answer server. Run `bash deploy/rhel/rag-rhel.sh check` on the target before
accepting the deployment. No H200 throughput or multi-GPU scaling claim is made.

Builder image used locally:
`sha256:c1e9d03835555bb2ed4e883f414f2356e83d2660593e16f38bcfcd2825bfa918`.
The OS description is in `vendor/build-os.txt`; dependency versions are in
`vendor/requirements-rhel.lock`, with file digests in `SHA256SUMS`.
