from unittest.mock import MagicMock

from app.services.query_rewriter import QueryRewriter


def _make_rewriter_with_mock_client(reply_text: str) -> tuple[QueryRewriter, MagicMock]:
    rewriter = QueryRewriter(api_key="fake-key")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=reply_text)]
    mock_client.messages.create.return_value = mock_response
    rewriter.client = mock_client
    return rewriter, mock_client


def test_rewrite_returns_original_question_when_no_history():
    rewriter = QueryRewriter(api_key="fake-key")
    rewriter.client = MagicMock()

    result = rewriter.rewrite("What about it?", history=[])

    assert result == "What about it?"
    rewriter.client.messages.create.assert_not_called()


def test_rewrite_returns_original_question_when_no_api_key():
    rewriter = QueryRewriter(api_key=None)

    history = [{"question": "What is RAG?", "answer": "Retrieval augmented generation."}]
    result = rewriter.rewrite("Tell me more", history=history)

    assert result == "Tell me more"


def test_rewrite_calls_model_and_returns_rewritten_question_when_history_exists():
    rewriter, mock_client = _make_rewriter_with_mock_client("What are the benefits of RAG?")

    history = [{"question": "What is RAG?", "answer": "Retrieval augmented generation."}]
    result = rewriter.rewrite("What about its benefits?", history=history)

    assert result == "What are the benefits of RAG?"
    mock_client.messages.create.assert_called_once()
    _, kwargs = mock_client.messages.create.call_args
    assert "What about its benefits?" in kwargs["messages"][0]["content"]
    assert "What is RAG?" in kwargs["messages"][0]["content"]


def test_rewrite_falls_back_to_original_on_api_error():
    rewriter = QueryRewriter(api_key="fake-key")
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("boom")
    rewriter.client = mock_client

    history = [{"question": "What is RAG?", "answer": "..."}]
    result = rewriter.rewrite("What about it?", history=history)

    assert result == "What about it?"


def test_rewrite_falls_back_when_model_returns_empty_string():
    rewriter, _ = _make_rewriter_with_mock_client("   ")

    history = [{"question": "What is RAG?", "answer": "..."}]
    result = rewriter.rewrite("What about it?", history=history)

    assert result == "What about it?"
