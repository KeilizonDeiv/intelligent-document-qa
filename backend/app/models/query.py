from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    n_results: int = Field(default=5, ge=1, le=20)
    use_hybrid: bool = True
    use_context: bool = True


class SourceCitation(BaseModel):
    source: str
    relevance: float
    chunk_id: str
    preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    retrieved_chunks: int
    model: str
    relevance_scores: list[float] = []


class ConversationExchange(BaseModel):
    question: str
    answer: str
    sources: list[str]


class ClearHistoryResponse(BaseModel):
    success: bool = True
    message: str
