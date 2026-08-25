from fastapi import APIRouter, Depends

from app.api.deps import get_rag_engine, get_session_id
from app.models.query import ClearHistoryResponse, ConversationExchange
from app.services.rag_engine import RAGEngine

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=list[ConversationExchange])
async def get_history(
    session_id: str = Depends(get_session_id),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    return rag_engine.get_conversation_history(session_id)


@router.delete("", response_model=ClearHistoryResponse)
async def clear_history(
    session_id: str = Depends(get_session_id),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    rag_engine.clear_history(session_id)
    return ClearHistoryResponse(message="Conversation history cleared")
