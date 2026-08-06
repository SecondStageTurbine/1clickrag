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
    """One generator, reachable over HTTP."""

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
            raise LlmError(f"could not reach {self.endpoint}: {exc}") from exc


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
            raise LlmError(f"could not reach {self.endpoint}: {exc}") from exc


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
                        text = (payload.get("message") or {}).get("content")
                        if text:
                            yield text
                        if payload.get("done"):
                            return
        except httpx.RequestError as exc:
            raise LlmError(f"could not reach {self.endpoint}: {exc}") from exc


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
