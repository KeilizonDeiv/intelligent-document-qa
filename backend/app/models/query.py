from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    n_results: int = Field(default=5, ge=1, le=20)
    use_reranking: bool = True
    use_context: bool = True


class SourceCitation(BaseModel):
    source: str
    relevance: float
    chunk_id: str
    preview: str
    rerank_score: float | None = None


class SourcesEvent(BaseModel):
    """First SSE event: the sources /api/query is about to answer from."""

    type: Literal["sources"] = "sources"
    sources: list[SourceCitation]
    retrieved_chunks: int


class TokenEvent(BaseModel):
    """One chunk of the streamed answer text."""

    type: Literal["token"] = "token"
    text: str


class DoneEvent(BaseModel):
    """Final SSE event once the answer has finished streaming."""

    type: Literal["done"] = "done"
    model: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


QueryStreamEvent = SourcesEvent | TokenEvent | DoneEvent | ErrorEvent


class ConversationExchange(BaseModel):
    question: str
    answer: str
    sources: list[str]


class ClearHistoryResponse(BaseModel):
    success: bool = True
    message: str
