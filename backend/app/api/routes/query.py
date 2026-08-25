from fastapi import APIRouter, Depends

from app.api.deps import get_rag_engine, get_session_id, get_vector_store
from app.core.exceptions import InvalidRequestError, NoDocumentsError
from app.models.query import QueryRequest, QueryResponse
from app.services.rag_engine import RAGEngine
from app.services.vector_store import VectorStore

router = APIRouter(tags=["query"])


@router.post("/api/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    session_id: str = Depends(get_session_id),
    vector_store: VectorStore = Depends(get_vector_store),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    question = payload.question.strip()
    if not question:
        raise InvalidRequestError("Question cannot be empty")

    stats = vector_store.get_stats(filter_dict={"session_id": session_id})
    if stats["total_chunks"] == 0:
        raise NoDocumentsError("No documents uploaded yet. Please upload documents first.")

    result = rag_engine.query(
        session_id=session_id,
        question=question,
        n_results=payload.n_results,
        use_reranking=payload.use_reranking,
        conversation_context=payload.use_context,
    )

    return result


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
