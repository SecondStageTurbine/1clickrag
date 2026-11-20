from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.config import Config
from app.embedder import FastEmbedEmbedder
from app.network_policy import (
    NetworkPolicyError,
    is_internal_host,
    require_internal_url,
    require_loopback_bind,
)
from app.reranker import Reranker


class NetworkPolicyTests(unittest.TestCase):
    def test_loopback_private_and_explicit_hosts_are_allowed(self):
        for host in ("localhost", "127.0.0.1", "::1", "10.20.30.40", "ollama"):
            self.assertTrue(is_internal_host(host))
        self.assertTrue(is_internal_host("llm01.example.smil.mil", ["llm01.example.smil.mil"]))

    def test_public_and_unlisted_fqdn_hosts_are_blocked(self):
        for url in (
            "https://api.openai.com/v1",
            "https://8.8.8.8/v1",
            "http://0.0.0.0:8000/v1",
            "https://internal.example.mil/v1",
        ):
            with self.assertRaises(NetworkPolicyError):
                require_internal_url(url, "test")

    def test_malformed_urls_are_blocked(self):
        for url in ("", "file:///tmp/model", "http://localhost:not-a-port"):
            with self.assertRaises(NetworkPolicyError):
                require_internal_url(url, "test")

    def test_remote_bind_requires_explicit_gateway_override(self):
        require_loopback_bind("127.0.0.1")
        with self.assertRaises(NetworkPolicyError):
            require_loopback_bind("0.0.0.0")
        require_loopback_bind("0.0.0.0", allow_remote=True)


class DarkConfigTests(unittest.TestCase):
    def test_dark_mode_disables_model_pull(self):
        cfg = Config(dark_network=True, auto_pull_model=True)
        self.assertFalse(cfg.auto_pull_model)

    def test_cloud_chat_is_rejected(self):
        with self.assertRaises(ValueError):
            Config(dark_network=True, chat_provider="anthropic")
        with self.assertRaises(NetworkPolicyError):
            Config(
                dark_network=True,
                chat_provider="openai",
                chat_url="https://api.openai.com/v1",
            )

    def test_local_openai_compatible_chat_is_allowed(self):
        cfg = Config(
            dark_network=True,
            chat_provider="openai",
            chat_url="http://127.0.0.1:9093/v1",
        )
        self.assertTrue(cfg.dark_network)


class LocalModelOnlyTests(unittest.TestCase):
    def test_embedder_passes_local_files_only(self):
        with tempfile.TemporaryDirectory() as cache:
            with patch("fastembed.TextEmbedding") as model:
                FastEmbedEmbedder("test/model", cache, offline=True).prepare()
                self.assertTrue(model.call_args.kwargs["local_files_only"])

    def test_reranker_passes_local_files_only(self):
        with tempfile.TemporaryDirectory() as cache:
            with patch("fastembed.rerank.cross_encoder.TextCrossEncoder") as model:
                Reranker("test/model", cache, offline=True).prepare()
                self.assertTrue(model.call_args.kwargs["local_files_only"])

    def test_dark_discovery_never_offers_agent_clis(self):
        from app import llm

        cfg = SimpleNamespace(
            dark_network=True,
            chat_provider="",
            chat_url="",
            chat_timeout=1,
        )
        with patch.object(llm, "_ollama_models", return_value=[]), patch.object(
            llm, "_cli_backends", side_effect=AssertionError("CLI discovery must be disabled")
        ):
            self.assertEqual(llm.discover(cfg), [])

    def test_local_openai_provider_reports_a_stopped_runner(self):
        import httpx
        from app.llm import OpenAIProvider

        provider = OpenAIProvider(
            "local-model", "", "http://127.0.0.1:9093/v1", 100, 0.0, 10
        )
        request = httpx.Request("GET", "http://127.0.0.1:9093/v1/models")
        with patch("app.llm.httpx.get", side_effect=httpx.ConnectError("down", request=request)):
            self.assertIn("no local chat service", provider.check())


if __name__ == "__main__":
    unittest.main()
