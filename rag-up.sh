#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
#
# One-command local RAG (Linux / macOS / WSL).
#
#   ./rag-up.sh --folder DIR index a folder of documents (remembered in .env)
#   ./rag-up.sh              start it - native mode, one Python process, no Docker
#   ./rag-up.sh --docker     start the containerised stack instead
#   ./rag-up.sh status       health + index size
#   ./rag-up.sh reindex      incremental re-ingest   (--full to rebuild)
#   ./rag-up.sh query "..."  one-off search from the shell
#   ./rag-up.sh bundle       vendor wheels + model so it installs with no internet
#   ./rag-up.sh logs         native: tail the log file; docker: follow the container
#   ./rag-up.sh down         stop     (--wipe also drops the index and model cache)

set -euo pipefail

cd "$(dirname "$0")"

# The server reads RAG_PORT from the environment or from .env; this script has
# to resolve it the same way or it polls a port nothing is listening on.
PORT="${RAG_PORT:-}"
if [ -z "$PORT" ] && [ -f .env ]; then
  PORT="$(sed -n 's/^[[:space:]]*RAG_PORT[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' .env | tail -1)"
fi
PORT="${PORT:-49404}"   # must match RAG_PORT default in app/config.py

API="http://127.0.0.1:$PORT"
DATA=".data"
VENV=".venv"
WHEELS="vendor/wheels"
PIDFILE="$DATA/rag.pid"
LOGFILE="$DATA/rag.log"

MODE=native
FOLDER=""
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --docker) MODE=docker ;;
    --native) MODE=native ;;
    --folder) shift; FOLDER="${1:-}" ;;
    --folder=*) FOLDER="${1#--folder=}" ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

# --folder writes the corpus into .env, so the next run needs no arguments and
# status/query/down all agree about which corpus this folder serves.
if [ -n "$FOLDER" ]; then
  if [ ! -d "$FOLDER" ]; then
    echo "error: folder not found: $FOLDER" >&2
    exit 1
  fi
  resolved="$(cd "$FOLDER" && pwd)"
  [ -f .env ] && grep -v -E '^[[:space:]]*RAG_(REPO_MOUNT|REPO_LABEL)[[:space:]]*=' .env > .env.tmp || : > .env.tmp
  {
    cat .env.tmp
    echo "RAG_REPO_MOUNT=$resolved"
    echo "RAG_REPO_LABEL=$(basename "$resolved")"
  } > .env
  rm -f .env.tmp
  echo "==> corpus set to $resolved"
  echo "    (remembered in .env - future runs need no --folder)"
fi

# ---------------------------------------------------------------- helpers

api_get() { curl -fsS --max-time 5 "$API$1" 2>/dev/null; }

api_post() {
  curl -fsS --max-time 30 -X POST "$API$1" \
    -H 'Content-Type: application/json' -d "$2" 2>/dev/null
}

json_field() {
  # $1 = json, $2 = key. Avoids a hard dependency on jq. A JSON null must come
  # back as an empty string, not the literal "None" — callers test emptiness.
  "$PY" -c 'import json,sys
value = json.loads(sys.argv[1]).get(sys.argv[2])
print("" if value is None else value)' "$1" "$2" 2>/dev/null || true
}

find_python() {
  # RAG_PYTHON pins a specific interpreter — the escape hatch when the default
  # one on PATH is too new (or too old) for a dependency to have wheels.
  for candidate in ${RAG_PYTHON:+"$RAG_PYTHON"} python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      command -v "$candidate"
      return 0
    fi
  done
  echo "error: Python 3.10+ is required for native mode." >&2
  echo "Install it, point RAG_PYTHON at an existing interpreter, or run the" >&2
  echo "containerised stack: ./rag-up.sh --docker" >&2
  exit 1
}

install_help() {
  echo >&2
  echo "Dependency install failed. Most often this is a Python version with no" >&2
  echo "prebuilt wheels yet for one of the packages. Two things to try:" >&2
  echo >&2
  echo "  1. Use a different interpreter (3.12 is the safest bet):" >&2
  echo "       rm -rf .venv && RAG_PYTHON=python3.12 ./rag-up.sh" >&2
  echo >&2
  echo "  2. Or run the containerised stack, which brings its own Python:" >&2
  echo "       ./rag-up.sh --docker" >&2
}

need_docker() {
  command -v docker >/dev/null 2>&1 || {
    echo "error: docker is not installed or not on PATH." >&2
    echo "Native mode needs no Docker at all — just run: ./rag-up.sh" >&2
    exit 1
  }
}

compose() {
  need_docker
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  else
    docker-compose "$@"
  fi
}

PY="$(command -v python3 || command -v python || echo python3)"

# Wait for /health to reach a terminal state; prints transitions as they happen.
# Something else may already own this port - notably the hand-rolled RAG this
# replaces, which also lives on 8404 and also answers /health with
# status=healthy. Without this check the launcher reports another service's
# health as its own success while our server has failed to bind.
is_ours() {
  local health
  health="$(api_get /health || true)"
  [ -n "$health" ] && [ "$(json_field "$health" service)" = "rag-local" ]
}

assert_port_free() {
  local health
  health="$(api_get /health || true)"
  [ -z "$health" ] && return 0
  is_ours && return 0
  echo >&2
  echo "error: port $PORT is already serving something else." >&2
  echo "Its /health replies, but it is not this RAG (no service=rag-local)." >&2
  echo "Stop that service, or give this one its own port in rag/.env." >&2
  echo "Pick from the private range 49152-65535, which nothing standard claims:" >&2
  echo "    RAG_PORT=49404" >&2
  exit 1
}

wait_for_health() {
  # Embedding is CPU-bound; a large corpus legitimately takes hours, so the
  # ceiling is generous and overridable rather than a hard hour.
  local limit=$(( ${RAG_START_TIMEOUT_MINUTES:-180} * 60 ))
  local waited=0 health status chunks last="" last_chunks=-1
  while [ "$waited" -lt "$limit" ]; do
    health="$(api_get /health || true)"
    if [ -n "$health" ] && [ "$(json_field "$health" service)" != "rag-local" ]; then
      echo >&2
      echo "error: $API is answering, but it is not this RAG." >&2
      echo "Another service holds the port; set RAG_PORT in rag/.env." >&2
      return 2
    fi
    if [ -n "$health" ]; then
      status="$(json_field "$health" status)"
      if [ "$status" != "$last" ]; then
        echo "    status: $status"
        last="$status"
      fi
      # A big corpus spends a long time here; show it moving.
      if [ "$status" = "indexing" ]; then
        chunks="$(json_field "$health" chunks)"
        if [ "$chunks" != "$last_chunks" ]; then
          echo "      indexed so far: $chunks chunks"
          last_chunks="$chunks"
        fi
      fi
      if [ "$status" = "healthy" ]; then
        chunks="$(json_field "$health" chunks)"
        echo
        echo "    RAG is live:  $API"
        echo "    indexed:      ${chunks} chunks"
        echo
        echo "    Open $API in a browser, or:"
        echo "      ./rag-up.sh query 'where is the IPC rendezvous done?'"
        return 0
      fi
      # A recorded bootstrap error is terminal — the model download failed, or
      # a dependency is misconfigured. Report it now rather than polling for an
      # hour against something that will never come up.
      local err
      err="$(json_field "$health" error)"
      if [ -n "$err" ]; then
        echo
        echo "error: startup failed: $err" >&2
        return 2
      fi
    fi
    sleep 5
    waited=$((waited + 5))
  done
  return 1
}

# ------------------------------------------------------------ native mode

native_venv() {
  # A virtualenv records absolute paths to the Python that built it, so one
  # copied from another machine looks present and fails cryptically. Probe it
  # and rebuild rather than inflicting that error on the user.
  if [ -e "$VENV/bin/python" ] && ! "$VENV/bin/python" -c 'import sys' >/dev/null 2>&1; then
    echo "==> $VENV cannot run on this machine (copied from another PC?) — rebuilding"
    rm -rf "$VENV"
  fi

  PY="$(find_python)"
  if [ ! -x "$VENV/bin/python" ]; then
    echo "==> creating $VENV ($("$PY" --version 2>&1))"
    "$PY" -m venv "$VENV"
  fi
  PY="$PWD/$VENV/bin/python"

  # Only reinstall when the requirements change.
  local stamp="$VENV/.requirements-sha"
  local current
  current="$(cat requirements.txt requirements-native.txt | "$PY" -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.read().encode()).hexdigest())')"
  if [ ! -f "$stamp" ] || [ "$(cat "$stamp")" != "$current" ]; then
    if [ -d "$WHEELS" ] && [ -n "$(ls -A "$WHEELS" 2>/dev/null)" ]; then
      echo "==> installing dependencies from $WHEELS (offline)"
      "$PY" -m pip install --quiet --no-index --find-links "$WHEELS" -r requirements-native.txt || {
        install_help; exit 1; }
    else
      echo "==> installing dependencies (first run only)"
      "$PY" -m pip install --quiet --upgrade pip
      "$PY" -m pip install --quiet -r requirements-native.txt || { install_help; exit 1; }
    fi
    echo "$current" > "$stamp"
  fi
}

# Prepare this folder to be carried to a machine with no internet access:
# vendor the wheels and pre-download the model, so `rag-up` there needs neither
# PyPI nor huggingface.co.
cmd_bundle() {
  native_venv
  mkdir -p "$WHEELS"
  echo "==> building wheels into $WHEELS"
  # pip wheel, not pip download: one dependency here is published only as an
  # sdist, and installing an sdist offline fails because pip's build isolation
  # reaches for setuptools over a network the target does not have.
  "$PY" -m pip wheel --quiet -r requirements-native.txt -w "$WHEELS"
  echo "==> pre-downloading the embedding model"
  RAG_MODE=native "$PY" -c 'from app.config import CONFIG
from app.embedder import make_embedder
make_embedder(CONFIG).prepare()
print("   model cached in", CONFIG.model_cache)'
  echo "==> pre-downloading the reranker (so it can be enabled offline)"
  RAG_MODE=native "$PY" -c 'from app.config import CONFIG
from app.reranker import Reranker
Reranker(CONFIG.rerank_model, CONFIG.model_cache).prepare()
print("   reranker cached:", CONFIG.rerank_model)' || echo "    (reranker download failed - can be fetched later)"
  echo
  echo "    This folder is now self-contained. Copy it to the other machine"
  echo "    WITHOUT .venv (it is machine-specific and will be rebuilt):"
  echo
  echo "      rsync -a --exclude .venv ./ /media/usb/rag/"
  echo
  echo "    Wheels are built for this OS and CPU. If the target machine differs,"
  echo "    re-run the download there, or add pip's --platform/--python-version."
}

native_running() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

native_up() {
  mkdir -p "$DATA"
  if native_running && is_ours; then
    echo "already running (pid $(cat "$PIDFILE")) - $API"
    return 0
  fi

  assert_port_free
  native_venv

  echo "==> starting the RAG server"
  echo "    (first run downloads the embedding model, then indexes the repo;"
  echo "     later starts reuse both and are seconds)"
  RAG_MODE=native "$PY" -m app.server >"$LOGFILE" 2>&1 &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  # No reaper process here on purpose: a background subshell would inherit this
  # script's stdout and keep `./rag-up.sh | tail` from ever finishing. A stale
  # pidfile is harmless — native_running() probes the pid before trusting it.

  local waited=0
  while ! api_get /health >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "error: the server exited during startup. Last lines:" >&2
      tail -20 "$LOGFILE" >&2
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
    [ "$waited" -gt 120 ] && { echo "error: server never answered /health" >&2; return 1; }
  done

  local rc=0
  wait_for_health || rc=$?
  case "$rc" in
    0) echo "    log:  $LOGFILE      stop with: ./rag-up.sh down"; return 0 ;;
    2) echo "    see $LOGFILE for the full traceback." >&2 ;;
    *) echo "note: gave up waiting, but the server may still be indexing." >&2
       echo "     check with: ./rag-up.sh status     log: $LOGFILE" >&2 ;;
  esac
  return 1
}

native_down() {
  if native_running; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "stopped."
  else
    echo "not running."
  fi
  if [ "${1:-}" = "--wipe" ]; then
    rm -rf "$DATA/qdrant" "$DATA/models"
    echo "index and model cache removed."
  fi
}

# ------------------------------------------------------------ docker mode

docker_up() {
  echo "==> building and starting the containerised stack"
  compose up -d --build
  echo "==> waiting for the index"
  wait_for_health || { echo "error: not healthy. Check: ./rag-up.sh --docker logs" >&2; return 1; }
}

# ------------------------------------------------------------------ main

cmd_query() {
  local query="${1:-}"
  if [ -z "$query" ]; then
    echo "usage: ./rag-up.sh query \"your question\"" >&2
    return 1
  fi
  local body
  body="$("$PY" -c 'import json,sys; print(json.dumps({"query": sys.argv[1], "top_k": 5}))' "$query")"
  api_post /search "$body" |
    "$PY" -c '
import json, sys
data = json.load(sys.stdin)
for hit in data.get("results", []):
    print("\n--- %s:%s-%s  (score %s)" % (
        hit["path"], hit["start_line"], hit["end_line"], hit["score"]))
    print(hit["text"])
'
}

case "${1:-up}" in
  up|"")
    if [ "$MODE" = docker ]; then docker_up; else native_up; fi
    ;;
  status)
    api_get /stats || { echo "not responding on $API (start it: ./rag-up.sh)"; exit 1; }
    echo
    ;;
  query)   shift; cmd_query "${1:-}" ;;
  bundle)  cmd_bundle ;;
  reindex)
    shift
    if [ "${1:-}" = "--full" ]; then api_post /reindex '{"full": true}'; else api_post /reindex '{"full": false}'; fi
    echo
    ;;
  logs)
    if [ "$MODE" = docker ]; then compose logs -f rag-api; else tail -f "$LOGFILE"; fi
    ;;
  down)
    shift
    if [ "$MODE" = docker ]; then
      if [ "${1:-}" = "--wipe" ]; then compose down -v; else compose down; fi
    else
      native_down "${1:-}"
    fi
    ;;
  *)
    echo "usage: ./rag-up.sh [--docker] [up|status|query \"...\"|reindex [--full]|bundle|logs|down [--wipe]]" >&2
    exit 1
    ;;
esac
