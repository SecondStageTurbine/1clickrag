# 1clickrag on RHEL 9: offline installation

Target: RHEL 9.8 x86_64, /usr/bin/python3.12, NVIDIA driver 580.178.04,
eight H200 GPUs. The app initially uses CUDA device 0 for embedding and
reranking. No answer-model server or answer-model weights are included.

This bundle includes application source, Linux Python 3.12 wheels, pinned
package versions, CUDA runtime/cuDNN libraries, and these existing models:

- nomic-ai/nomic-embed-text-v1.5
- Xenova/ms-marco-MiniLM-L-6-v2

The OS must already provide Python 3.12 with pip/venv, glibc, libstdc++,
libgcc, zlib and the working NVIDIA driver. No Docker is needed on the target.
Use the existing RHEL packages; the installer does not run dnf or change
/usr/bin/python3 or its packages. An administrator must provide any missing
OS packages from the approved RHEL repository/media.

## Install

Transfer the ZIP and its .sha256 file through your approved transfer process.
In the directory containing both files:

```bash
sha256sum -c 1clickrag-rhel9-py312-cuda13.zip.sha256
unzip 1clickrag-rhel9-py312-cuda13.zip
cd 1clickrag-rhel9-py312-cuda13
bash deploy/rhel/rag-rhel.sh install
```

Installation checks every packaged file, creates .venv-rhel with Python 3.12,
and installs only from the bundled wheels. It does not download anything.
If interrupted, preserve or remove the incomplete .venv-rhel before retrying;
the installer refuses to overwrite an existing environment. For an update,
extract to a fresh release directory and migrate your configuration/data after
stopping the old process. Do not copy a Windows virtual environment or Windows
absolute paths into this release.

Edit `.env` and set `RAG_REPO_MOUNT` to your actual Linux document directory.
It must exist and be readable by the account running 1clickrag. Keep the app
directory writable by that account for its local index and caches.
`RAG_GPU_DEVICE=0` selects the first visible GPU; change it if another device
is allocated to the app. This app does not distribute retrieval across all
eight GPUs. Coordinate GPU allocation with the future answer-model server.

```bash
vi .env
bash deploy/rhel/rag-rhel.sh check
bash deploy/rhel/rag-rhel.sh run
```

The check loads both staged models and requires actual CUDA sessions on the
configured device. It also exercises OCR, PDF rendering, local vector storage,
and imports the server. A CUDA provider listed as available is not sufficient.
If the check fails, keep the complete output for troubleshooting. `check-cpu`
tests the same assets without a GPU; it does not prove GPU operation.
`benchmark` runs the existing comparative CPU/GPU benchmark. If `check` reports
a fall back to CPU, `probe` prints the reason on a few lines: the GPUs the
driver sees, device visibility, and the library the CUDA provider could not load.

The run command stays in the foreground. Open http://127.0.0.1:49404 on that
machine; Ctrl+C stops it. From a workstation with approved SSH access, use:

```bash
ssh -N -L 49404:127.0.0.1:49404 your-user@your-rhel-host
```

Then open http://127.0.0.1:49404 on that workstation. Keep the default loopback
binding unless deploying behind your approved authenticated internal gateway.
The first start builds a new Linux index. Use this RHEL launcher for subsequent
starts; the generic launcher's dependency installer is not the GPU lock installer.

## Connect an answer model later

For Ollama, set these existing settings in `.env` and restart the app:

```dotenv
RAG_CHAT_PROVIDER=ollama
RAG_CHAT_URL=http://your-model-host.lan:11434
RAG_CHAT_MODEL=exact-name-shown-by-your-server
```

For llama.cpp's OpenAI-compatible server:

```dotenv
RAG_CHAT_PROVIDER=openai
RAG_CHAT_URL=http://your-model-host.lan:8080/v1
RAG_CHAT_MODEL=exact-model-alias-configured-on-your-server
```

Use your actual internal address and server model identifier. `openai` selects
the protocol here; the explicit internal URL determines the destination.
If your internal FQDN is outside the built-in local/private host rules, add its
exact hostname to `RAG_DARK_NETWORK_ALLOWED_HOSTS`. Set `RAG_CHAT_API_KEY` only
if your internal service requires authentication. No model is automatically
pulled. Search works before chat is configured, and connecting or changing an
answer model does not require rebuilding embeddings.

## Package maintenance and validation limits

`vendor/requirements-rhel.lock` records exact installed versions. GPU ONNX
Runtime replaces CPU ONNX Runtime; headless OpenCV replaces GUI OpenCV. Their
upstream consumers still declare the CPU/GUI distribution names, so `pip check`
will report those missing names. Do not "repair" this by installing both builds:
they share import directories. The bundled runtime check tests the replacements.
All remaining dependencies are installed from the frozen set with --no-deps.
The runtime check also validates dependency versions using these two substitutions.

CUDA 13.0 runtime/NVRTC and ONNX Runtime 1.28.0 are constrained in build.sh;
the resulting NVIDIA libraries and all other dependencies are pinned in the
lock file. Bundled third-party wheels retain their own license metadata.
See [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
and [NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
The pinned cuDNN release lists RHEL 9, Hopper and this driver family in its
[support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.13.1/reference/support-matrix.html).

An offline installation and CPU smoke test on a RHEL 9-compatible container
cannot certify execution on the target H200s. Run the target GPU check above
before accepting the deployment. See VALIDATION.md for checks actually run.

## Rebuild on connected staging

From the source workspace, with Python 3.11+ for the packaging helper and the
two models already in `.data/models`:

```bash
python deploy/rhel/package.py stage dist/1clickrag-rhel9-py312-cuda13
docker build -f deploy/rhel/Dockerfile.builder -t rag-rhel-builder .
docker run --rm -v "$PWD/dist/1clickrag-rhel9-py312-cuda13:/bundle" rag-rhel-builder
python deploy/rhel/package.py manifest dist/1clickrag-rhel9-py312-cuda13
# Test install and check-cpu in a fresh container with --network none.
python deploy/rhel/package.py archive dist/1clickrag-rhel9-py312-cuda13
```

The rebuild resolves current base-package versions; the delivered lock and
wheels preserve this release's exact versions. Archive creation excludes the
local `.env`, `.venv-rhel`, document index and caches other than model weights.
