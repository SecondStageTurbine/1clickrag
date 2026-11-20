"""Offline integration: real extraction/index/API/Ollama HTTP adapter/SSE.

The small deterministic embedder and scripted HTTP model are test doubles.
These tests establish wiring and evidence delivery, NOT real LLM accuracy.
All state is isolated from the installed corpus and .data directory.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import shutil
import socket
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from openpyxl import Workbook

from app.chunker import chunk_file
from app.config import Config
from app.extract import from_xlsx
from app.llm import OllamaProvider


class TestEmbedder:
    dim = 2

    def embed_query(self, query):
        return [1.0, 0.0]


class ChatIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cfg = Config(repo_path=str(cls.root), qdrant_url="",
                     qdrant_path=str(cls.root / "vectors"), collection="chat-test",
                     graph=True, graph_path=str(cls.root / "graph.db"),
                     queue_path=str(cls.root / "queue.db"), rerank=False,
                     hybrid=True, chat_context_chars=12000, chat_max_searches=2)
        spec = importlib.util.spec_from_file_location(
            "app._chat_test_server", Path(__file__).parents[1] / "app/server.py")
        cls.server = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.server
        with patch("app.config.CONFIG", cfg), \
                patch("app.embedder.make_embedder", return_value=TestEmbedder()), \
                patch("app.llm.make_provider", return_value=None):
            spec.loader.exec_module(cls.server)
        cls.server.store.ensure_collection(2)
        cls.client = TestClient(cls.server.app)  # No production ingest/watcher lifespan.

        book = Workbook()
        sheet = book.active
        sheet.title = "Self Test"
        sheet.append(["Fault Condition", "Parameter Name", "ARINC Label", "Artifact"])
        sheet.append(["Low pressure", "Hydraulic Pressure", "213", "BIT Report"])
        book.save(cls.root / "self-test.xlsx")
        book.close()
        cls.index("self-test.xlsx", from_xlsx(str(cls.root / "self-test.xlsx")))
        cls.index("supplier (fixture).txt", "SYNTHETIC TEST DATA ONLY\n"
                  "4900500 Go to figure 2-4, block 1.\n\n"
                  "[diagram node N1]\nPressure below limit\n\n"
                  "[diagram edge E1]\nFrom: Pressure below limit\n"
                  "Relationship: yes_branch\nTo: Set fault\n"
                  "Figure 2-4 TO SHT 3\n\n"
                  "Figure 2-4 Sheet 3: FROM SHT 1. Replace test relay.\n"
                  "Unresolved NO branch; do not invent its action.")
        cls.index("other-revision.txt", "Unrelated revision: do not mix this source.")

        cls.calls = []
        cls.replies = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                cls.calls.append(json.loads(self.rfile.read(length)))
                reply = cls.replies.pop(0) if cls.replies else "No scripted reply"
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for fragment in (reply[:15], reply[15:]):
                    self.wfile.write((json.dumps({"message": {"content": fragment}}) + "\n").encode())
                self.wfile.write(b'{"done":true}\n')

            def log_message(self, *args):
                pass

        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.provider = OllamaProvider("scripted-test-model", f"http://127.0.0.1:{cls.http.server_port}", 1000, 0, 10)
        cls.provider.id = "test-local"
        cls.server.pick_backend = lambda wanted: cls.provider

    @classmethod
    def index(cls, path, text):
        chunks = chunk_file(path, "text", text, chunk_chars=1600, overlap_lines=0)
        cls.server.store.upsert(chunks, [[1.0, 0.0] for _ in chunks], mtime=1.0)
        cls.server.graph.replace_path(path, chunks)

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join(timeout=5)
        cls.client.close()
        cls.server.queue.close()
        cls.server.graph.close()
        cls.server.store.close()
        sys.modules.pop("app._chat_test_server", None)
        cls.temp.cleanup()

    def setUp(self):
        self.calls.clear()
        self.replies.clear()

    def chat(self, mode="general", question="Low pressure", **kwargs):
        response = self.client.post("/chat", json={
            "messages": [{"role": "user", "content": question}], "mode": mode, **kwargs})
        self.assertEqual(response.status_code, 200, response.text)
        events = []
        for frame in response.text.strip().split("\n\n"):
            name, data = frame.split("\n", 1)
            events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
        self.assertFalse(any(name == "error" for name, _ in events), events)
        self.assertEqual(events[-1][0], "done")
        return events

    def test_general_remains_default_and_all_modes_are_discoverable(self):
        data = self.client.get("/chat/modes").json()
        self.assertEqual(data["default"], "general")
        self.assertEqual({m["id"] for m in data["modes"]}, {
            "general", "fault_trace", "diagnostic_review", "symptom_failure_matrix",
            "diagnostic_peer_review", "security_peer_review"})
        self.replies.append("Hydraulic Pressure uses label 213 [1].")
        events = self.chat(path_prefix="self-test.xlsx")
        prompt = self.calls[-1]["messages"][0]["content"]
        self.assertNotIn("Task policy:", prompt)
        self.assertNotIn("Required Output Format", prompt)
        self.assertEqual(events[0][1]["mode"], "general")
        self.assertIn("ARINC Label: 213", self.calls[-1]["messages"][-1]["content"])
        self.assertEqual("".join(d["text"] for n, d in events if n == "delta"),
                         "Hydraulic Pressure uses label 213 [1].")

    def test_extraction_hybrid_retrieval_and_policy_reach_model_for_both_analysis_modes(self):
        for mode, marker in (("diagnostic_review", "6.2 Fault Inventory"),
                             ("symptom_failure_matrix", "E = Explicitly")):
            with self.subTest(mode=mode):
                self.replies.append("Supported mapping [1]. Unknowns remain.")
                events = self.chat(mode, path_prefix="self-test.xlsx", top_k=1)
                messages = self.calls[-1]["messages"]
                self.assertIn(marker, messages[0]["content"])
                self.assertIn("not classification determinations", messages[0]["content"])
                self.assertNotIn("PE002", messages[0]["content"])
                self.assertIn("Parameter Name: Hydraulic Pressure", messages[-1]["content"])
                self.assertIn("Artifact: BIT Report", messages[-1]["content"])
                self.assertTrue({"vector", "keyword", "relationship"}.issubset(events[0][1]["ranking"]))
                self.assertTrue(all(s["path"] == "self-test.xlsx" for s in events[0][1]["sources"]))
                self.assertIn("ARINC Label: 213", events[-1][1]["review_evidence"])

    def test_prior_answer_and_evidence_survive_peer_review_without_recitation_collision(self):
        self.replies.append("Pressure uses label 213 [1].")
        initial = self.chat(path_prefix="self-test.xlsx")
        prior = {"analysis": "Pressure uses label 213 [1].",
                 "evidence": initial[-1][1]["review_evidence"]}
        for mode in ("diagnostic_peer_review", "security_peer_review"):
            self.replies.append("Review finding cites original [P1].")
            events = self.chat(mode, "Check this prior analysis", review=prior, retrieve=False)
            body = self.calls[-1]["messages"][-1]["content"]
            self.assertIn("Pressure uses label 213 [P1]", body)
            self.assertIn("[P1] self-test.xlsx", body)
            self.assertIn("ARINC Label: 213", body)
            self.assertEqual(events[0][1]["sources"], [])
            self.assertEqual(len(self.calls[-1]["messages"]), 2)

    def test_peer_review_fresh_search_preserves_original_evidence_namespace(self):
        self.replies.extend(["SEARCH: Hydraulic Pressure\n", "Use original [P1] and supplemental [1]."])
        self.chat("diagnostic_peer_review", "Check grounding", path_prefix="self-test.xlsx",
                  review={"analysis": "Original claim [1]", "evidence": "[1] original.txt:1-1\nOriginal evidence"})
        self.assertEqual(len(self.calls), 2)
        body = self.calls[-1]["messages"][-1]["content"]
        self.assertIn("[P1] original.txt", body)
        self.assertIn("[1] self-test.xlsx", body)
        self.assertIn("Original claim [P1]", body)

    def test_missing_analysis_rejected_and_missing_evidence_disclosed(self):
        for review in (None, {"analysis": "   "}):
            response = self.client.post("/chat", json={"messages": [{"role": "user", "content": "review"}],
                "mode": "diagnostic_peer_review", "review": review})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.calls, [])
        self.replies.append("Original evidence unavailable; limited review.")
        self.chat("security_peer_review", review={"analysis": "A claim [1]"}, retrieve=False)
        self.assertIn("original grounding cannot be fully verified", self.calls[-1]["messages"][-1]["content"])

    def test_invalid_modes_and_oversized_review_fail_before_model_call(self):
        for extra, status in (({"mode": "not-real"}, 400),
                              ({"mode": "general", "review": {"analysis": "X"}}, 400),
                              ({"mode": "diagnostic_peer_review", "review": {"analysis": "X" * 60001}}, 422)):
            response = self.client.post("/chat", json={"messages": [{"role": "user", "content": "test"}], **extra})
            self.assertEqual(response.status_code, status)
        self.assertEqual(self.calls, [])

    def test_fault_trace_follows_references_and_keeps_diagram_body(self):
        self.replies.extend(["SEARCH: Figure 2-4 Sheet 3\n", "Synthetic action: Replace test relay [1]. NO branch unresolved."])
        with patch.object(self.server, "context", wraps=self.server.context) as retrieval:
            events = self.chat("fault_trace", "Trace 4900500", path_prefix="supplier (fixture).txt", top_k=1)
        queries = [c.args[0].query for c in retrieval.call_args_list]
        self.assertIn("4900500 Figure 2-4 Block 1", queries)
        self.assertIn("Figure 2-4 Sheet 3", queries)
        self.assertTrue(all(c.args[0].max_chars == 24000 for c in retrieval.call_args_list))
        self.assertTrue(all(c.args[0].path_prefix == "supplier (fixture).txt" for c in retrieval.call_args_list))
        body = self.calls[-1]["messages"][-1]["content"]
        self.assertIn("[diagram edge E1]", body)
        self.assertIn("Replace test relay", body)
        self.assertIn("Unresolved NO branch", body)
        self.assertTrue(all(d["mode"] == "fault_trace" for n, d in events if n == "sources"))

    def test_context_merge_preserves_parenthesized_path_and_diagram_headings(self):
        source = {"n": 1, "citation": "supplier (rev A).pdf:1-10", "heading": "Diagram"}
        block = "[1] supplier (rev A).pdf:1-10 (Diagram)\nStart\n\n[diagram node N1]\nSet fault"
        merged, sources = self.server._merge_context(block, [source], block, [source], 24000)
        self.assertEqual(merged, block)
        self.assertEqual(len(sources), 1)

    def test_model_followup_retains_specialist_context_above_general_budget(self):
        def found(path, body):
            return {"context": f"[1] {path}:1-1\n{body}",
                    "sources": [{"n": 1, "citation": f"{path}:1-1", "path": path}]}
        first = found("initial.txt", "A" * 10000)
        second = found("followup.txt", "B" * 7000 + " END OF FOLLOWUP")
        self.replies.extend(["SEARCH: exact followup\n", "Supported partial answer [1] [2]."])
        with patch.object(self.server, "context", side_effect=[first, second]):
            events = self.chat("diagnostic_review")
        self.assertIn("END OF FOLLOWUP", self.calls[-1]["messages"][-1]["content"])
        self.assertGreater(len(events[-1][1]["review_evidence"]), 12000)
        self.assertLessEqual(len(events[-1][1]["review_evidence"]), 24000)

    def test_failed_supplemental_search_produces_limited_answer_not_empty_success(self):
        from fastapi.responses import JSONResponse
        self.replies.extend(["SEARCH: missing reference\n", "Supplemental search failed; original evidence is limited [P1]."])
        with patch.object(self.server, "context", return_value=JSONResponse(status_code=503, content={"error": "unavailable"})):
            events = self.chat("diagnostic_peer_review", review={
                "analysis": "Original claim [1]", "evidence": "[1] example.txt:1-1\nPartial evidence"})
        self.assertEqual(len(self.calls), 2)
        self.assertIn("supplemental search failed", self.calls[-1]["messages"][-1]["content"])
        self.assertTrue(any(n == "delta" and "limited" in d["text"] for n, d in events))

    @unittest.skipUnless(shutil.which("node"), "Node required for UI handler integration")
    def test_ui_mode_selection_answer_review_and_reset_over_http(self):
        import uvicorn
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        service = uvicorn.Server(uvicorn.Config(self.server.app, lifespan="off", log_level="error"))
        thread = threading.Thread(target=service.run, kwargs={"sockets": [sock]}, daemon=True)
        self.replies.extend(["Pressure uses label 213 [1].", "Supported original claim [P1].",
                             "Source marking not specified in original [P1]."])
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not service.started and thread.is_alive() and time.monotonic() < deadline:
                threading.Event().wait(0.01)
            self.assertTrue(service.started, "isolated HTTP test server did not start")
            result = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("chat_ui_harness.cjs")),
                                     f"http://127.0.0.1:{port}"], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("UI handoff checks passed", result.stdout)
            self.assertEqual(len(self.calls), 3)
        finally:
            service.should_exit = True
            thread.join(timeout=10)
            sock.close()


if __name__ == "__main__":
    unittest.main()
