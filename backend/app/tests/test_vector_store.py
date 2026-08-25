import pytest

from app.services.document_processor import DocumentChunk
from app.services.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_directory=str(tmp_path / "vector_db"))


def _chunk(text: str, source: str, session_id: str, num: int) -> DocumentChunk:
    return DocumentChunk(
        text=text,
        metadata={"source": source, "chunk_num": num, "session_id": session_id},
    )


def test_add_and_search_returns_relevant_chunk(store):
    chunks = [
        _chunk("The Eiffel Tower is located in Paris.", "a.txt", "s1", 0),
        _chunk("Bananas are rich in potassium.", "b.txt", "s1", 0),
    ]
    store.add_documents(chunks)

    results = store.search("Where is the Eiffel Tower?", n_results=1)

    assert len(results) == 1
    assert "Eiffel" in results[0]["text"]


def test_get_stats_reflects_added_chunks(store):
    chunks = [_chunk("Some content.", "doc.txt", "s1", 0)]
    store.add_documents(chunks)

    stats = store.get_stats()

    assert stats["total_chunks"] == 1
    assert stats["sources"] == ["doc.txt"]


def test_session_filter_isolates_chunks_between_sessions(store):
    store.add_documents([_chunk("Session one content.", "a.txt", "session-1", 0)])
    store.add_documents([_chunk("Session two content.", "b.txt", "session-2", 0)])

    stats_1 = store.get_stats(filter_dict={"session_id": "session-1"})
    stats_2 = store.get_stats(filter_dict={"session_id": "session-2"})

    assert stats_1["total_chunks"] == 1
    assert stats_1["sources"] == ["a.txt"]
    assert stats_2["total_chunks"] == 1
    assert stats_2["sources"] == ["b.txt"]

    results_1 = store.search("content", n_results=5, filter_dict={"session_id": "session-1"})
    assert {r["metadata"]["source"] for r in results_1} == {"a.txt"}


def test_delete_by_source_respects_session_filter(store):
    store.add_documents([_chunk("Doc for session 1.", "shared.txt", "session-1", 0)])
    store.add_documents([_chunk("Doc for session 2.", "shared.txt", "session-2", 0)])

    deleted = store.delete_by_source("shared.txt", filter_dict={"session_id": "session-1"})

    assert deleted == 1
    remaining = store.get_stats(filter_dict={"session_id": "session-2"})
    assert remaining["total_chunks"] == 1


def test_clear_all_with_filter_only_clears_that_session(store):
    store.add_documents([_chunk("A.", "a.txt", "session-1", 0)])
    store.add_documents([_chunk("B.", "b.txt", "session-2", 0)])

    store.clear_all(filter_dict={"session_id": "session-1"})

    assert store.get_stats(filter_dict={"session_id": "session-1"})["total_chunks"] == 0
    assert store.get_stats(filter_dict={"session_id": "session-2"})["total_chunks"] == 1
