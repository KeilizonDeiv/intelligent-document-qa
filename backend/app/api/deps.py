from fastapi import Request

from app.services.rag_engine import RAGEngine
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore


def get_document_processor(request: Request) -> DocumentProcessor:
    return request.app.state.document_processor


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_rag_engine(request: Request) -> RAGEngine:
    return request.app.state.rag_engine
