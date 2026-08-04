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
"""

from __future__ import annotations

import logging

log = logging.getLogger("rag.reranker")


class Reranker:
    def __init__(self, model: str, cache_dir: str, threads: int = 0) -> None:
        self.model_name = model
        self.cache_dir = cache_dir
        self.threads = threads
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            log.info("loading reranker %s (first run downloads it)", self.model_name)
            kwargs = {"model_name": self.model_name, "cache_dir": self.cache_dir}
            if self.threads > 0:
                kwargs["threads"] = self.threads
            self._model = TextCrossEncoder(**kwargs)
            log.info("reranker %s ready", self.model_name)
        return self._model

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
