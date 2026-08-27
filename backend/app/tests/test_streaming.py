from unittest.mock import MagicMock

import pytest

from app.services.rag_engine import RAGEngine


class _FakeStream:
    def __init__(self, tokens):
        self._tokens = tokens

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def _gen(self):
        for token in self._tokens:
            yield token

    @property
    def text_stream(self):
        return self._gen()


class _FailingStream:
    async def __aenter__(self):
        raise RuntimeError("upstream API error")

    async def __aexit__(self, *exc_info):
        return False


def _make_engine_with_chunks(tokens=None, raise_error=False):
    vector_store = MagicMock()
    vector_store.search.return_value = [
        {
            "id": "chunk-1",
            "text": "Paris is the capital of France.",
            "metadata": {"source": "geo.txt"},
            "relevance_score": 0.9,
        }
    ]

    engine = RAGEngine(vector_store=vector_store, api_key="fake-key", model="claude-sonnet-5")
    engine.client = MagicMock()

    if raise_error:
        engine.client.messages.stream = MagicMock(return_value=_FailingStream())
    else:
        engine.client.messages.stream = MagicMock(return_value=_FakeStream(tokens or ["Paris ", "is ", "the capital."]))

    return engine


@pytest.mark.anyio
async def test_stream_query_emits_sources_then_tokens_then_done():
    engine = _make_engine_with_chunks(tokens=["Paris ", "is ", "the capital."])

    events = [e async for e in engine.stream_query(session_id="s1", question="What is the capital of France?")]

    assert events[0]["type"] == "sources"
    assert events[0]["retrieved_chunks"] == 1

    token_events = [e for e in events[1:-1]]
    assert all(e["type"] == "token" for e in token_events)
    assert "".join(e["text"] for e in token_events) == "Paris is the capital."

    assert events[-1]["type"] == "done"
    assert events[-1]["model"] == "claude-sonnet-5"


@pytest.mark.anyio
async def test_stream_query_records_full_answer_in_history():
    engine = _make_engine_with_chunks(tokens=["Paris ", "is ", "the capital."])

    events = [e async for e in engine.stream_query(session_id="s1", question="What is the capital of France?")]
    assert events[-1]["type"] == "done"

    history = engine.get_conversation_history("s1")
    assert len(history) == 1
    assert history[0]["answer"] == "Paris is the capital."
    assert history[0]["question"] == "What is the capital of France?"


@pytest.mark.anyio
async def test_stream_query_emits_error_event_on_generation_failure():
    engine = _make_engine_with_chunks(raise_error=True)

    events = [e async for e in engine.stream_query(session_id="s1", question="What is the capital of France?")]

    assert events[0]["type"] == "sources"
    assert events[-1]["type"] == "done"
    assert any(e["type"] == "error" for e in events)


@pytest.mark.anyio
async def test_stream_query_no_chunks_returns_fallback_message():
    vector_store = MagicMock()
    vector_store.search.return_value = []

    engine = RAGEngine(vector_store=vector_store, api_key=None)

    events = [e async for e in engine.stream_query(session_id="s1", question="Anything?")]

    assert events[0] == {"type": "sources", "sources": [], "retrieved_chunks": 0}
    assert events[1]["type"] == "token"
    assert "couldn't find" in events[1]["text"]
    assert events[2] == {"type": "done", "model": "none"}


@pytest.fixture
def anyio_backend():
    return "asyncio"
