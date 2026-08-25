import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, history, query, stats
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.models.common import HealthResponse
from app.services.document_processor import DocumentProcessor
from app.services.query_rewriter import QueryRewriter
from app.services.rag_engine import RAGEngine
from app.services.reranker import Reranker
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)

    app.state.document_processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    app.state.vector_store = VectorStore(
        persist_directory=str(settings.vector_db_path),
        embedding_model_name=settings.embedding_model,
    )
    app.state.rag_engine = RAGEngine(
        vector_store=app.state.vector_store,
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        reranker=Reranker(model_name=settings.reranker_model),
        query_rewriter=QueryRewriter(api_key=settings.anthropic_api_key, model=settings.anthropic_rewrite_model),
        rerank_candidate_multiplier=settings.rerank_candidate_multiplier,
    )

    logger.info(
        "AI Document Intelligence Assistant started",
        extra={"extra_fields": {"has_api": app.state.rag_engine.has_api}},
    )

    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="AI Document Intelligence Assistant", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(stats.router)
    app.include_router(history.router)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(has_api=app.state.rag_engine.has_api)

    return app


app = create_app()
