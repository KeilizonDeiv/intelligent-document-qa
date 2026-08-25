import logging
import os
import shutil
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import get_document_processor, get_rag_engine, get_session_id, get_vector_store
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.core.files import secure_filename
from app.models.document import (
    ClearDocumentsResponse,
    DeleteDocumentResponse,
    DocumentUploadResponse,
    VectorStoreStats,
)
from app.services.document_processor import DocumentProcessor
from app.services.rag_engine import RAGEngine
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id),
    settings: Settings = Depends(get_settings),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    vector_store: VectorStore = Depends(get_vector_store),
):
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in document_processor.supported_formats:
        raise UnsupportedFileTypeError(
            f"File type not supported. Allowed: {', '.join(document_processor.supported_formats)}"
        )

    filename = secure_filename(file.filename or "upload")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{filename}"

    session_upload_dir = settings.upload_dir / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    filepath = session_upload_dir / unique_filename

    contents = await file.read()
    if len(contents) > settings.max_file_size_bytes:
        raise FileTooLargeError(f"File too large. Maximum size is {settings.max_file_size_mb}MB")

    filepath.write_bytes(contents)

    chunks = document_processor.process_file(str(filepath), session_id=session_id, source_name=filename)

    if not chunks:
        raise EmptyDocumentError("No text could be extracted from the document")

    num_added = vector_store.add_documents(chunks)
    doc_stats = document_processor.get_document_stats(chunks)

    logger.info(
        "Document uploaded",
        extra={"extra_fields": {"filename": filename, "chunks_created": num_added}},
    )

    return DocumentUploadResponse(filename=filename, chunks_created=num_added, stats=doc_stats)


@router.get("", response_model=VectorStoreStats)
async def list_documents(
    session_id: str = Depends(get_session_id),
    vector_store: VectorStore = Depends(get_vector_store),
):
    return vector_store.get_stats(filter_dict={"session_id": session_id})


@router.delete("", response_model=ClearDocumentsResponse)
async def clear_documents(
    session_id: str = Depends(get_session_id),
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store),
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    vector_store.clear_all(filter_dict={"session_id": session_id})
    rag_engine.clear_history(session_id)

    session_upload_dir = settings.upload_dir / session_id
    if session_upload_dir.exists():
        shutil.rmtree(session_upload_dir)

    return ClearDocumentsResponse(message="All documents and history cleared")


@router.delete("/{source}", response_model=DeleteDocumentResponse)
async def delete_document(
    source: str,
    session_id: str = Depends(get_session_id),
    vector_store: VectorStore = Depends(get_vector_store),
):
    deleted_count = vector_store.delete_by_source(source, filter_dict={"session_id": session_id})

    if deleted_count == 0:
        raise DocumentNotFoundError(f"No chunks found for source: {source}")

    return DeleteDocumentResponse(
        deleted_chunks=deleted_count,
        message=f"Deleted {deleted_count} chunks from {source}",
    )
