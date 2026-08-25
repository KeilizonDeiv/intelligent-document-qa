from pydantic import BaseModel


class DocumentStats(BaseModel):
    total_chunks: int
    total_characters: int
    avg_chunk_size: int
    sources: list[str]


class DocumentUploadResponse(BaseModel):
    success: bool = True
    filename: str
    chunks_created: int
    stats: DocumentStats


class VectorStoreStats(BaseModel):
    total_chunks: int
    unique_sources: int
    sources: list[str]
    collection_name: str
    has_api: bool | None = None
    conversation_length: int | None = None


class DeleteDocumentResponse(BaseModel):
    success: bool = True
    deleted_chunks: int
    message: str


class ClearDocumentsResponse(BaseModel):
    success: bool = True
    message: str
