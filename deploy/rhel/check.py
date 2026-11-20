"""Exercise staged models, OCR, PDF rendering and local storage without downloads."""
import importlib.metadata
import math
import sys
import tempfile

from packaging.requirements import Requirement

from app.config import CONFIG
from app.embedder import FastEmbedEmbedder
from app.reranker import Reranker

cpu = "--cpu" in sys.argv
# Validate the complete installed dependency graph with the two intentional
# distribution replacements, rather than hiding unrelated missing packages.
replacements = {"onnxruntime": "onnxruntime-gpu", "opencv-python": "opencv-python-headless"}
for distribution in importlib.metadata.distributions():
    for declaration in distribution.requires or []:
        req = Requirement(declaration)
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        name = replacements.get(req.name.lower().replace("_", "-"), req.name)
        version = importlib.metadata.version(name)
        assert version in req.specifier, f"{distribution.name}: {req} incompatible with {name}=={version}"
if not cpu and not CONFIG.gpu:
    raise SystemExit("Set RAG_GPU=1 in .env for the GPU acceptance check.")
embed = FastEmbedEmbedder(CONFIG.embed_model, CONFIG.model_cache, threads=2,
                         gpu=not cpu, device=CONFIG.gpu_device, offline=True)
vectors = embed.embed_documents(["The deployment uses local document retrieval."])
assert vectors and all(math.isfinite(x) for x in vectors[0])
rerank = Reranker(CONFIG.rerank_model, CONFIG.model_cache, threads=2,
                  gpu=not cpu, device=CONFIG.gpu_device, offline=True)
hits = rerank.rerank("local retrieval", [{"text": "Local document retrieval", "score": 0.5}], 1)
assert hits
expected = "CPUExecutionProvider" if cpu else "CUDAExecutionProvider"
for name, model in (("embedder", embed), ("reranker", rerank)):
    # Check the actual session, not just ONNX Runtime's compiled provider list.
    assert model.provider == expected, f"{name} fell back to {model.provider}"
    if not cpu:
        assert model.device_in_use == CONFIG.gpu_device
    print(f"{name}: {model.provider}, device={model.device_in_use}")
import numpy as np
from rapidocr_onnxruntime import RapidOCR
RapidOCR()(np.full((64, 256, 3), 255, dtype=np.uint8))
import pypdfium2 as pdfium
doc = pdfium.PdfDocument.new()
page = doc.new_page(100, 100)
bitmap = page.render(scale=1)
assert bitmap.to_pil().size == (100, 100)
bitmap.close()
page.close()
doc.close()
from qdrant_client import QdrantClient, models
with tempfile.TemporaryDirectory() as path:
    client = QdrantClient(path=path)
    client.create_collection("check", vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE))
    client.upsert("check", [models.PointStruct(id=1, vector=vectors[0])])
    assert client.query_points("check", query=vectors[0], limit=1).points
    client.close()
import app.server
print("PASS: embedding, reranking, OCR, PDF rendering, storage, server import")
