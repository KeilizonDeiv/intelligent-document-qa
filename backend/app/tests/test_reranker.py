import pytest

from app.services.reranker import Reranker


@pytest.fixture(scope="module")
def reranker():
    return Reranker()


def test_rerank_orders_by_relevance_to_query(reranker):
    query = "What is the capital of France?"
    candidates = [
        {"id": "a", "text": "Bananas are a good source of potassium."},
        {"id": "b", "text": "Paris is the capital and most populous city of France."},
        {"id": "c", "text": "The stock market fluctuated wildly last week."},
    ]

    results = reranker.rerank(query, candidates, top_n=3)

    assert [r["id"] for r in results][0] == "b"
    assert all("rerank_score" in r for r in results)


def test_rerank_respects_top_n(reranker):
    query = "machine learning frameworks"
    candidates = [
        {"id": str(i), "text": f"Unrelated sentence number {i} about gardening."} for i in range(5)
    ]
    candidates.append({"id": "ml", "text": "PyTorch and TensorFlow are popular machine learning frameworks."})

    results = reranker.rerank(query, candidates, top_n=2)

    assert len(results) == 2
    assert results[0]["id"] == "ml"


def test_rerank_empty_candidates_returns_empty(reranker):
    assert reranker.rerank("anything", [], top_n=5) == []
