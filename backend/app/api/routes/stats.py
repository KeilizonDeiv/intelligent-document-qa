from fastapi import APIRouter, Depends

from app.api.deps import get_rag_engine, get_session_id, get_vector_store
from app.models.document import VectorStoreStats
from app.services.rag_engine import RAGEngine
from app.services.vector_store import VectorStore

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("", response_model=VectorStoreStats)
async def get_stats(
    session_id: str = Depends(get_session_id),
    vector_store: VectorStore = Depends(get_vector_store),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    stats = vector_store.get_stats(filter_dict={"session_id": session_id})
    stats["has_api"] = rag_engine.has_api
    stats["conversation_length"] = rag_engine.history_length(session_id)
    return stats
