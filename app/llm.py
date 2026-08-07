# SPDX-License-Identifier: MPL-2.0
"""Answer generation for the chat pane.

Retrieval is what this project does. Generation is not, and that is deliberate:
the index runs offline in one process, and a language model is a separate
decision about cost, privacy and hardware that belongs to whoever installs
this. So the chat pane is a thin client over whatever generator it is pointed
at, and the server ships with none configured - `/chat` answers 501 until
RAG_CHAT_PROVIDER is set.

Three backends cover the ground:

    anthropic  the Claude Messages API
    openai     any OpenAI-compatible /chat/completions endpoint - OpenAI and
               its Codex models, and equally llama.cpp, LM Studio, vLLM,
               text-generation-webui, or Ollama's own compatibility shim
    ollama     Ollama's native /api/chat, for a model on the local network

All three speak HTTP through httpx, which is already a dependency. No vendor
SDK is added: a wheel per provider would have to be bundled for four
interpreter versions in an install that has to work with no internet, and one
uniform streaming loop is easier to reason about than three.

Every provider streams. A grounded answer over several thousand characters of
retrieved context takes long enough that a blank screen reads as a hang, and
a local model on CPU takes long enough to hit an HTTP timeout.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from typing import Iterator

import httpx

log = logging.getLogger("rag.llm")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Sent to the OpenAI-compatible and Ollama backends only. The Claude models
# reject sampling parameters outright (HTTP 400), so `anthropic` omits it.
DEFAULT_TEMPERATURE = 0.2


class LlmError(RuntimeError):
    """A generator problem worth showing the user verbatim.

    Deliberately not a bare RuntimeError: everything raised with this type is
    a sentence someone can act on - a missing key, an unreachable host, a
    model name the server has never heard of - and the chat pane prints it as
    written rather than logging it and showing "something went wrong".
    """


class Provider:
    """One generator this server can put a question to."""

    #: Stable identifier the browser sends back to choose this one.
    id = ""
    #: What the dropdown shows.
    label = ""
    #: Short name, as configured.
    name = ""
    #: The model this will ask for.
    model = ""
    #: Where the documents go. Shown in the UI, unabbreviated and on purpose:
    #: a chat pane that quietly posts a corpus to a third party is the one
    #: failure this tool must never have.
    endpoint = ""
    #: True when the endpoint is this machine or the local network.
    local = False

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        raise NotImplementedError

    def check(self) -> str | None:
        """None when the backend looks usable, else why it does not."""
        return None


def _timeout(seconds: float) -> httpx.Timeout:
    # Read is the long one: a local model on CPU can take a minute to produce
    # its first token. Connect stays short so an unreachable host fails fast
    # rather than looking like a slow model.
    return httpx.Timeout(seconds, connect=10.0, write=30.0, pool=10.0)


def _is_local(url: str) -> bool:
    host = httpx.URL(url).host.lower()
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    # RFC1918 and link-local. A model on the office network is not the
    # internet, and the UI should not warn about it as though it were.
    return (
        host.startswith("10.")
        or host.startswith("192.168.")
        or host.startswith("169.254.")
        or any(host.startswith(f"172.{block}.") for block in range(16, 32))
    )


def _transport_error(exc: httpx.RequestError, endpoint: str, seconds: float,
                     label: str) -> LlmError:
    """Say which of the two very different failures this was.

    "could not reach X" is wrong and misleading when the host answered
    immediately and then spent three minutes thinking: it sends someone to
    check networking and firewalls when the machine is simply slower than the
    deadline. A large model on modest hardware is the ordinary cause, and the
    setting to change is the one named here.
    """
    if isinstance(exc, httpx.TimeoutException):
        return LlmError(
            f"{label} did not finish within {seconds:.0f}s. The host answered, so "
            f"this is the model being slow rather than unreachable - a large "
            f"model can take minutes per answer. Raise RAG_CHAT_TIMEOUT in "
            f"rag/.env, or pick a smaller model."
        )
    return LlmError(f"could not reach {endpoint}: {exc}")


def _raise_for_status(response: httpx.Response, label: str) -> None:
    """Turn a streaming error response into a readable message.

    httpx has not read the body yet on a streamed request, so `.text` is empty
    until `read()` is called - without this the user gets "HTTP 401" and no
    hint that the key is the problem.
    """
    if response.status_code == 200:
        return
    response.read()
    detail = response.text.strip()
    try:
        body = json.loads(detail)
        detail = (
            body.get("error", {}).get("message")
            if isinstance(body.get("error"), dict)
            else body.get("error")
        ) or detail
    except (ValueError, AttributeError):
        pass
    raise LlmError(f"{label} returned HTTP {response.status_code}: {detail[:400]}")


def _sse_payloads(response: httpx.Response) -> Iterator[dict]:
    """Yield the JSON object from each `data:` line of an SSE stream."""
    for line in response.iter_lines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue  # event:, id:, retry: and keep-alive blanks
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except ValueError:
            continue


class AnthropicProvider(Provider):
    """Claude, through the Messages API.

    Streaming is raw SSE rather than the SDK, for the bundling reason in the
    module docstring. The wire format is stable and small: switch on the
    `type` field of each event and keep the text deltas.
    """

    def __init__(self, model: str, api_key: str, max_tokens: int, timeout: float,
                 url: str = "") -> None:
        self.name = "anthropic"
        self.model = model or "claude-opus-5"
        self.endpoint = url or ANTHROPIC_URL
        self.local = _is_local(self.endpoint)
        self._key = api_key
        self._max_tokens = max_tokens
        self._timeout = timeout

    def check(self) -> str | None:
        if not self._key:
            return "no API key - set RAG_CHAT_API_KEY in rag/.env"
        return None

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        if not self._key:
            raise LlmError("no API key - set RAG_CHAT_API_KEY in rag/.env")

        body = {
            "model": self.model,
            # A hard ceiling on thinking *and* answer together. Claude's newer
            # models think by default, so a tight budget truncates the answer
            # rather than merely shortening it.
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": messages,
            "stream": True,
        }
        # No temperature/top_p/top_k: the current Claude models reject them
        # with a 400. The other two backends still get them.
        headers = {
            "x-api-key": self._key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        try:
            with httpx.Client(timeout=_timeout(self._timeout)) as client:
                with client.stream(
                    "POST", self.endpoint, json=body, headers=headers
                ) as response:
                    _raise_for_status(response, "the Claude API")
                    for payload in _sse_payloads(response):
                        kind = payload.get("type")
                        if kind == "content_block_delta":
                            delta = payload.get("delta") or {}
                            # thinking_delta is the other kind that arrives
                            # here; it is not the answer and is empty unless
                            # summaries are asked for, so only text is kept.
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                        elif kind == "message_delta":
                            stop = (payload.get("delta") or {}).get("stop_reason")
                            if stop == "refusal":
                                raise LlmError(
                                    "the model declined to answer this one"
                                )
                            if stop == "max_tokens":
                                yield "\n\n[cut off at the token limit - raise RAG_CHAT_MAX_TOKENS]"
                        elif kind == "error":
                            message = (payload.get("error") or {}).get("message")
                            raise LlmError(message or "the Claude API reported an error")
        except httpx.RequestError as exc:
            raise _transport_error(
                exc, self.endpoint, self._timeout, self.model or self.name
            ) from exc


class OpenAIProvider(Provider):
    """Anything speaking OpenAI's /chat/completions.

    That is one wire format and many servers: OpenAI itself and its Codex
    models, and every local runner worth using - llama.cpp's server, LM
    Studio, vLLM, text-generation-webui, Ollama's shim. Which one is at the
    other end is a base URL, not a code path.
    """

    def __init__(self, model: str, api_key: str, url: str, max_tokens: int,
                 temperature: float, timeout: float) -> None:
        self.name = "openai"
        self.model = model
        base = (url or "https://api.openai.com/v1").rstrip("/")
        self.endpoint = f"{base}/chat/completions"
        self._base = base
        self.local = _is_local(self.endpoint)
        self._key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    def check(self) -> str | None:
        if not self.model:
            return "no model set - RAG_CHAT_MODEL is required for this provider"
        if not self.local and not self._key:
            return "no API key - set RAG_CHAT_API_KEY in rag/.env"
        return None

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        problem = self.check()
        if problem:
            raise LlmError(problem)

        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": True,
        }
        headers = {"content-type": "application/json"}
        if self._key:
            headers["authorization"] = f"Bearer {self._key}"

        try:
            with httpx.Client(timeout=_timeout(self._timeout)) as client:
                with client.stream(
                    "POST", self.endpoint, json=body, headers=headers
                ) as response:
                    _raise_for_status(response, self._base)
                    for payload in _sse_payloads(response):
                        choices = payload.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        # Some local servers stream a separate reasoning field.
                        # It is not the answer, so only `content` is kept.
                        text = delta.get("content")
                        if text:
                            yield text
        except httpx.RequestError as exc:
            raise _transport_error(
                exc, self.endpoint, self._timeout, self.model or self.name
            ) from exc


class OllamaProvider(Provider):
    """Ollama's native /api/chat.

    Ollama also offers an OpenAI-compatible endpoint, and either works. This
    one exists because it is what an Ollama host answers on out of the box,
    and because /api/tags lets a wrong model name produce a list of the right
    ones instead of a bare 404.
    """

    def __init__(self, model: str, url: str, max_tokens: int, temperature: float,
                 timeout: float) -> None:
        self.name = "ollama"
        self.model = model
        base = (url or "http://127.0.0.1:11434").rstrip("/")
        self._base = base
        self.endpoint = f"{base}/api/chat"
        self.local = _is_local(self.endpoint)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    def models(self) -> list[str]:
        """What this host has pulled, or an empty list if it cannot be asked."""
        try:
            response = httpx.get(f"{self._base}/api/tags", timeout=5.0)
            response.raise_for_status()
            return [item["name"] for item in response.json().get("models", [])]
        except Exception:
            return []

    def check(self) -> str | None:
        available = self.models()
        if not available:
            return f"no Ollama at {self._base} - is it running?"
        if not self.model:
            return f"no model set - this host has: {', '.join(available[:8])}"
        # Ollama tags are name:tag; a bare name matches its default tag.
        if not any(
            name == self.model or name.split(":")[0] == self.model
            for name in available
        ):
            return (
                f"'{self.model}' is not pulled on {self._base} - "
                f"available: {', '.join(available[:8])}"
            )
        return None

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        if not self.model:
            raise LlmError("no model set - RAG_CHAT_MODEL is required")

        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": True,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }

        last_status = 0.0
        thought = 0
        try:
            with httpx.Client(timeout=_timeout(self._timeout)) as client:
                with client.stream("POST", self.endpoint, json=body) as response:
                    _raise_for_status(response, self._base)
                    # Newline-delimited JSON, not SSE.
                    for line in response.iter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except ValueError:
                            continue
                        if payload.get("error"):
                            raise LlmError(str(payload["error"]))
                        message = payload.get("message") or {}
                        # A reasoning model streams its thinking first and its
                        # answer afterwards. The thinking is not the answer and
                        # is dropped - but dropping it silently is what makes a
                        # slow model look like a hung one, because minutes pass
                        # with the stream alive and nothing to show for it. So
                        # say that it is working, at a rate nobody has to read.
                        if message.get("thinking") and not message.get("content"):
                            now = time.time()
                            if now - last_status > 2.0:
                                last_status = now
                                thought += len(message["thinking"])
                                yield {"status": "thinking", "chars": thought}
                        text = message.get("content")
                        if text:
                            yield text
                        if payload.get("done"):
                            return
        except httpx.RequestError as exc:
            raise _transport_error(
                exc, self.endpoint, self._timeout, self.model or self.name
            ) from exc


class CliProvider(Provider):
    """A coding agent installed on this machine, driven as a subprocess.

    `claude` and `codex` are not HTTP services - they are signed-in command
    line tools. Someone who has them has a capable model already paid for, and
    telling them to go and find an API key to use it from the pane would be
    silly. So this shells out.

    Two things make that safe and workable rather than a shortcut:

    **The prompt never touches a shell.** argv is a list, so nothing is parsed
    for quotes, pipes or redirection no matter what the retrieved passages
    contain.

    **Length is the real constraint, and it decides the delivery.** A grounded
    prompt runs to ~13,000 characters. `claude` reads stdin, which has no
    limit, so it gets it there. `codex` wants its prompt as an argument, where
    Windows caps a command line at 32,767 - workable, but only by invoking its
    script directly: through the npm .cmd shim the ceiling is cmd.exe's 8,191,
    and a full prompt is rejected outright with no useful error. Measured, not
    assumed: 13,303 characters through the shim exits 1 in 0.0s; the same
    prompt straight to node answers correctly.
    """

    # Below cmd.exe's 8191 there is no argument this cannot carry; above
    # CreateProcess's 32767 there is no way to pass it at all. The gap is where
    # the launcher choice matters.
    ARGV_LIMIT = 30000

    def __init__(self, key: str, label: str, argv: list[str], model: str,
                 timeout: float, prompt_on_stdin: bool) -> None:
        self.id = f"cli:{key}"
        self.label = label
        self.name = f"cli/{key}"
        self.model = model
        self.endpoint = f"{argv[0]} (this machine)"
        self.local = True
        self._argv = argv
        self._timeout = timeout
        self._stdin = prompt_on_stdin

    def check(self) -> str | None:
        return None

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        # These tools take one prompt, not a conversation, so the turns are
        # flattened. Prior turns still matter for follow-ups, and labelling
        # them is what keeps "shorten that" meaningful.
        parts = [system, ""]
        for message in messages[:-1]:
            who = "Question" if message["role"] == "user" else "Your earlier answer"
            parts.append(f"[{who}]\n{message['content']}")
        parts.append(messages[-1]["content"])
        prompt = "\n\n".join(parts)

        argv = list(self._argv)
        if not self._stdin:
            if len(prompt) > self.ARGV_LIMIT:
                raise LlmError(
                    f"{self.label} takes its prompt as a command-line argument, and "
                    f"this one is {len(prompt):,} characters - past the "
                    f"{self.ARGV_LIMIT:,} Windows allows. Lower RAG_CHAT_CONTEXT_CHARS."
                )
            argv.append(prompt)

        # No console window, and no shell: argv goes to CreateProcess as-is.
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
        except OSError as exc:
            raise LlmError(f"could not run {argv[0]}: {exc}") from exc

        # Every pipe gets its own thread, and this is not tidiness - it is the
        # difference between working and hanging.
        #
        # stdin: these tools wait for end-of-input before starting, so the pipe
        # must be closed. `codex` handed an open stdin sat for 90 seconds and
        # produced nothing.
        #
        # stderr: `codex` narrates to it - a banner, the model name, progress.
        # A pipe holds about 64KB, and when it fills the child blocks trying to
        # write. If this thread is meanwhile blocked reading stdout, neither
        # side can move again: a deadlock that survives any timeout expressed
        # as "check the clock between lines", because no line ever arrives.
        # Draining stderr continuously is what prevents it.
        #
        # stdout: read on a thread too, so the deadline below is enforced by a
        # queue poll rather than by a blocking read that may never return.
        lines: "queue.Queue[str | None]" = queue.Queue()
        errors: list[str] = []

        def feed() -> None:
            try:
                if self._stdin:
                    process.stdin.write(prompt)
                process.stdin.close()
            except OSError:
                pass

        def drain_stdout() -> None:
            try:
                for line in process.stdout:
                    lines.put(line)
            except (OSError, ValueError):
                pass
            finally:
                lines.put(None)

        def drain_stderr() -> None:
            try:
                for line in process.stderr:
                    errors.append(line)
                    del errors[:-40]  # only the tail is ever a useful message
            except (OSError, ValueError):
                pass

        for target in (feed, drain_stdout, drain_stderr):
            threading.Thread(target=target, daemon=True).start()

        deadline = time.time() + self._timeout
        produced = False
        try:
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise LlmError(
                        f"{self.label} produced nothing in {self._timeout:.0f}s"
                    )
                try:
                    line = lines.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    continue  # still working; re-check the clock
                if line is None:
                    break
                produced = True
                yield line
        finally:
            if process.poll() is None:
                process.kill()

        code = process.wait()
        if code != 0 and not produced:
            tail = [line.strip() for line in errors if line.strip()]
            reason = tail[-1] if tail else f"exit code {code}"
            raise LlmError(f"{self.label} failed: {reason[:300]}")


def _ollama_models(url: str) -> list[str]:
    try:
        response = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=3.0)
        response.raise_for_status()
        names = [item["name"] for item in response.json().get("models", [])]
    except Exception:
        return []
    # An embedding model answers /api/chat with nonsense rather than an error,
    # so offering one in a chat dropdown produces a puzzling failure rather
    # than a clear one. This server pulls nomic-embed-text itself, so the case
    # is guaranteed, not hypothetical.
    return [name for name in names if "embed" not in name.lower()]


def _cli_backends(cfg) -> list[Provider]:
    """Signed-in coding agents on this machine, if any."""
    found: list[Provider] = []

    claude = shutil.which("claude")
    if claude:
        found.append(CliProvider(
            "claude", "Claude Code (CLI)", [claude, "-p"],
            model="claude-code", timeout=cfg.chat_timeout, prompt_on_stdin=True,
        ))

    # Straight to the script rather than the npm shim, for the argv ceiling
    # explained on CliProvider. Without node there is nothing to run it with.
    codex = shutil.which("codex")
    node = shutil.which("node")
    if codex and node:
        script = os.path.join(
            os.path.dirname(codex), "node_modules", "@openai", "codex", "bin", "codex.js"
        )
        if os.path.isfile(script):
            found.append(CliProvider(
                "codex", "Codex (CLI)", [node, script, "exec"],
                model="codex", timeout=cfg.chat_timeout, prompt_on_stdin=False,
            ))
    return found


def discover(cfg) -> list[Provider]:
    """Every generator this machine can reach, for the chat dropdown.

    Discovery rather than configuration, because the alternative is a list in
    a file that is wrong the moment someone pulls a new model. Ask Ollama what
    it is holding and the answer is right by construction - pull deepseek and
    it is in the dropdown on the next page load, with nothing to edit.

    The explicitly configured provider still comes first when there is one:
    it is the only way to reach a hosted API, and someone who wrote it down
    meant it.
    """
    backends: list[Provider] = []

    configured = make_provider(cfg)
    if configured is not None:
        configured.id = "configured"
        configured.label = f"{configured.model} (configured)"
        backends.append(configured)

    ollama_url = cfg.chat_url or "http://127.0.0.1:11434"
    if (cfg.chat_provider or "").strip().lower() != "ollama":
        ollama_url = "http://127.0.0.1:11434"
    for model in _ollama_models(ollama_url):
        if configured is not None and configured.name == "ollama" and configured.model == model:
            continue  # already listed as the configured one
        provider = OllamaProvider(
            model=model, url=ollama_url, max_tokens=cfg.chat_max_tokens,
            temperature=cfg.chat_temperature, timeout=cfg.chat_timeout,
        )
        provider.id = f"ollama:{model}"
        provider.label = f"{model} (local)"
        backends.append(provider)

    backends.extend(_cli_backends(cfg))
    return backends


def make_provider(cfg) -> Provider | None:
    """Build the configured generator, or None when chat is off.

    Off is the default and the honest one: this ships to machines with no
    internet and no local model, and a chat box that fails on every message
    would be worse than a chat box that says it is not configured.
    """
    name = (cfg.chat_provider or "").strip().lower()
    if not name or name in ("none", "off", "0"):
        return None

    if name in ("anthropic", "claude"):
        return AnthropicProvider(
            model=cfg.chat_model or "claude-opus-5",
            api_key=cfg.chat_api_key,
            max_tokens=cfg.chat_max_tokens,
            timeout=cfg.chat_timeout,
            url=cfg.chat_url,
        )
    if name in ("openai", "codex", "local", "openai-compatible"):
        return OpenAIProvider(
            model=cfg.chat_model,
            api_key=cfg.chat_api_key,
            url=cfg.chat_url,
            max_tokens=cfg.chat_max_tokens,
            temperature=cfg.chat_temperature,
            timeout=cfg.chat_timeout,
        )
    if name == "ollama":
        return OllamaProvider(
            model=cfg.chat_model,
            url=cfg.chat_url or "http://127.0.0.1:11434",
            max_tokens=cfg.chat_max_tokens,
            temperature=cfg.chat_temperature,
            timeout=cfg.chat_timeout,
        )

    log.warning(
        "unknown RAG_CHAT_PROVIDER=%r - expected anthropic, openai or ollama", name
    )
    return None
