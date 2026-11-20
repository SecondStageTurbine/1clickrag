"""Staging-only end-to-end test, using a disposable synthetic document corpus."""
import os
import tempfile
import time
from pathlib import Path

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    corpus = root / "documents"
    corpus.mkdir()
    (corpus / "deployment.txt").write_text(
        "Deployment instructions\nThe retrieval service uses port 49404.\n"
        "Documents remain on the internal network.\n", encoding="utf-8")
    os.environ.update({
        "RAG_REPO_MOUNT": str(corpus), "RAG_GPU": "0", "RAG_EMBED_THREADS": "2",
        "RAG_DARK_NETWORK": "1", "RAG_MODE": "native", "RAG_CHAT_PROVIDER": "",
        "RAG_MODEL_CACHE": str(Path(".data/models").resolve()),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    for key, name in {
        "RAG_QDRANT_PATH": "qdrant", "RAG_GRAPH_PATH": "graph.db",
        "RAG_QUEUE_PATH": "queue.db", "RAG_EMBED_CACHE_PATH": "embed.db",
        "RAG_EXTRACT_CACHE_PATH": "extract.db", "RAG_MANIFEST_PATH": "manifest.json",
    }.items():
        os.environ[key] = str(root / name)
    from fastapi.testclient import TestClient
    from app.server import app
    with TestClient(app) as client:
        for _ in range(90):
            health = client.get("/health").json()
            if health.get("error"):
                raise RuntimeError(health)
            if health.get("status") == "healthy":
                break
            time.sleep(1)
        else:
            raise AssertionError(health)
        assert client.get("/").status_code == 200
        assert client.get("/preview.js").status_code == 200
        assert client.get("/preview.css").status_code == 200
        response = client.post("/search", json={"query": "What port does retrieval use?", "top_k": 1})
        assert response.status_code == 200, response.text
        hit = response.json()["results"][0]
        assert hit["path"] == "deployment.txt", hit
        preview = client.post("/preview", json={k: hit[k] for k in ("path", "start_line", "end_line")})
        assert preview.status_code == 200, preview.text
        assert "49404" in preview.text
        assert client.post("/preview", json={"path": "../outside.txt", "start_line": 1, "end_line": 2}).status_code == 403
    print("PASS: offline startup, indexing, search, citation preview, UI assets and path rejection")
