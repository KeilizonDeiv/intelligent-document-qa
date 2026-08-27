import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_rag_engine, get_session_id, get_vector_store
from app.core.exceptions import InvalidRequestError, NoDocumentsError
from app.models.query import QueryRequest
from app.services.rag_engine import RAGEngine
from app.services.vector_store import VectorStore

router = APIRouter(tags=["query"])


def _format_sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _event_stream(rag_engine: RAGEngine, **kwargs) -> AsyncIterator[str]:
    async for event in rag_engine.stream_query(**kwargs):
        yield _format_sse(event)


@router.post("/api/query")
async def query(
    payload: QueryRequest,
    session_id: str = Depends(get_session_id),
    vector_store: VectorStore = Depends(get_vector_store),
    rag_engine: RAGEngine = Depends(get_rag_engine),
) -> StreamingResponse:
    """Answer a question over the session's documents as a stream of
    Server-Sent Events: one `sources` event, then one or more `token`
    events, then a final `done` event (see app.models.query for the event
    shapes)."""
    question = payload.question.strip()
    if not question:
        raise InvalidRequestError("Question cannot be empty")

    stats = vector_store.get_stats(filter_dict={"session_id": session_id})
    if stats["total_chunks"] == 0:
        raise NoDocumentsError("No documents uploaded yet. Please upload documents first.")

    return StreamingResponse(
        _event_stream(
            rag_engine,
            session_id=session_id,
            question=question,
            n_results=payload.n_results,
            use_reranking=payload.use_reranking,
            conversation_context=payload.use_context,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/sample-questions", response_model=list[str])
async def sample_questions(
    session_id: str = Depends(get_session_id),
    vector_store: VectorStore = Depends(get_vector_store),
):
    stats = vector_store.get_stats(filter_dict={"session_id": session_id})

    if stats["total_chunks"] == 0:
        return [
            "Upload a document to get started!",
            "Try uploading a PDF, DOCX, or TXT file",
            "Then ask questions about its content",
        ]

    return [
        "What are the main topics covered in these documents?",
        "Can you summarize the key points?",
        "What are the most important takeaways?",
        "Are there any specific recommendations or conclusions?",
        "What details are provided about [specific topic]?",
    ]
