"""ChromaDB-backed embedding storage and semantic search."""

import logging

import chromadb
from sentence_transformers import SentenceTransformer

from app.services.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "./vector_db",
        collection_name: str = "documents",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=persist_directory)

        logger.info("Loading embedding model %s", embedding_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info("Embedding model loaded")

        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def add_documents(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True).tolist()

        self.collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)

        return len(chunks)

    def search(self, query: str, n_results: int = 5, filter_dict: dict | None = None) -> list[dict]:
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict,
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance_score": 1 - results["distances"][0][i],
                }
            )

        return formatted_results

    def get_stats(self, filter_dict: dict | None = None) -> dict:
        if filter_dict:
            sample = self.collection.get(where=filter_dict, limit=10_000)
            count = len(sample["ids"])
            sources = {m.get("source", "unknown") for m in sample["metadatas"]}
        else:
            count = self.collection.count()
            if count > 0:
                sample = self.collection.get(limit=min(100, count))
                sources = {m.get("source", "unknown") for m in sample["metadatas"]}
            else:
                sources = set()

        return {
            "total_chunks": count,
            "unique_sources": len(sources),
            "sources": list(sources),
            "collection_name": self.collection_name,
        }

    def delete_by_source(self, source: str, filter_dict: dict | None = None) -> int:
        where = {"source": source}
        if filter_dict:
            where = {"$and": [where, filter_dict]}

        results = self.collection.get(where=where)

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])

        return 0

    def clear_all(self, filter_dict: dict | None = None) -> None:
        if filter_dict:
            results = self.collection.get(where=filter_dict)
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
            return

        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
