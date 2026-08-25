from fastapi import APIRouter, Depends

from app.api.deps import get_rag_engine
from app.models.query import ClearHistoryResponse, ConversationExchange
from app.services.rag_engine import RAGEngine

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=list[ConversationExchange])
async def get_history(rag_engine: RAGEngine = Depends(get_rag_engine)):
    return rag_engine.get_conversation_history()


@router.delete("", response_model=ClearHistoryResponse)
async def clear_history(rag_engine: RAGEngine = Depends(get_rag_engine)):
    rag_engine.clear_history()
    return ClearHistoryResponse(message="Conversation history cleared")
