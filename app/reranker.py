# SPDX-License-Identifier: MPL-2.0
"""Optional cross-encoder reranking.

The vector search is a *bi-encoder*: query and chunk are embedded separately and
compared by distance. That is what makes it fast enough to scan a whole corpus,
and it is also its ceiling - the two texts never meet, so the score reflects
topical proximity rather than whether the passage answers the question. Ask "how
should the book be split into volumes?" of a book about splitting text and every
page about chunking scores highly, because they genuinely are about splitting.

A cross-encoder reads the query and the passage *together* and scores the pair.
That is far better at exactly this distinction, and far too slow to run over a
whole corpus - so it runs over the top candidates the vector search already
found. Retrieve wide and cheap, then re-order narrow and accurate.

Off by default: it costs a second or two per query on CPU and another model
download. Worth turning on when feeding a generator, where the quality of the
top few chunks decides the quality of the answer.

It is also, by a wide margin, the most expensive thing in the query path, and
the reason to say so here is that the arithmetic is not obvious. Embedding the
question is one short forward pass - 33ms, a rounding error. Reranking is one
forward pass *per candidate* over passages of chunk_chars each: measured at
415ms for 10 candidates, 1.3s for 40, and 4.5s for 150, and again for every
re-search round. So "the embedding model is on the GPU" says nearly nothing
about how long a search takes, while this does. Hence RAG_GPU covers both.
"""

from __future__ import annotations

import logging

from . import accel

log = logging.getLogger("rag.reranker")


class Reranker:
    def __init__(self, model: str, cache_dir: str, threads: int = 0,
                 gpu: bool = False, gpu_setting: str = "RAG_GPU") -> None:
        self.model_name = model
        self.cache_dir = cache_dir
        self.threads = threads
        self.gpu = gpu
        self.gpu_setting = gpu_setting
        self._model = None
        self._provider = accel.CPU

    def _load(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            log.info("loading reranker %s (first run downloads it)", self.model_name)
            kwargs = {"model_name": self.model_name, "cache_dir": self.cache_dir}
            if self.threads > 0:
                kwargs["threads"] = self.threads
            # Always explicit, both ways - see accel.chosen().
            kwargs["providers"] = accel.chosen(self.gpu, self.gpu_setting)
            self._model = TextCrossEncoder(**kwargs)
            # Asked at load rather than trusted from config: a CUDA request that
            # fell back to CPU warns rather than raises, and reporting the
            # request would hide exactly the case worth seeing.
            self._provider = accel.active(self._model, self.gpu)
            log.info("reranker %s ready on %s", self.model_name, self._provider)
        return self._model

    @property
    def provider(self) -> str:
        return self._provider

    def prepare(self) -> None:
        self._load()

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        """Re-order `hits` by cross-encoder score and return the best `top_k`.

        Falls back to the incoming order if the reranker cannot run: a broken
        reranker should degrade the ranking, never fail the search.
        """
        if not hits:
            return hits
        try:
            model = self._load()
            scores = list(model.rerank(query, [h["text"] for h in hits]))
        except Exception as exc:
            log.warning("rerank failed, keeping vector order: %s", exc)
            return hits[:top_k]

        for hit, score in zip(hits, scores):
            # Keep the vector score visible: a large disagreement between the
            # two is informative, and hiding it would make the reranker's
            # effect impossible to audit from the results alone.
            hit["vector_score"] = hit["score"]
            hit["score"] = round(float(score), 6)

        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]
