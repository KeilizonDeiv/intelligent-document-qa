import uuid

from fastapi import Request, Response

from app.services.rag_engine import RAGEngine
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore

SESSION_COOKIE_NAME = "session_id"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def get_document_processor(request: Request) -> DocumentProcessor:
    return request.app.state.document_processor


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_rag_engine(request: Request) -> RAGEngine:
    return request.app.state.rag_engine


def get_session_id(request: Request, response: Response) -> str:
    """Identify the caller so documents/history stay scoped to one browser.

    This is a lightweight session concept for a single-tenant demo app, not
    authentication - anyone who has (or guesses) the cookie value can access
    that session's data. See the README's design notes.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            max_age=SESSION_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
        )
    return session_id
