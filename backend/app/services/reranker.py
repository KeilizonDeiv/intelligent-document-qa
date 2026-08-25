"""Cross-encoder reranking of retrieved chunks against the query."""

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info("Loading cross-encoder reranker %s", model_name)
        self.model = CrossEncoder(model_name)
        logger.info("Reranker loaded")

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        """Reorder `candidates` by cross-encoder relevance to `query`.

        Each candidate must have a "text" key. Adds a "rerank_score" key
        (raw cross-encoder logit, higher = more relevant) and returns the
        top `top_n` candidates sorted by it.
        """
        if not candidates:
            return []

        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

        return candidates[:top_n]
