"""Orchestrates retrieval and answer generation via the Claude API."""

import logging

import anthropic

from app.services.query_rewriter import QueryRewriter
from app.services.reranker import Reranker
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        api_key: str | None = None,
        model: str = "claude-sonnet-5",
        reranker: Reranker | None = None,
        query_rewriter: QueryRewriter | None = None,
        rerank_candidate_multiplier: int = 3,
    ):
        self.vector_store = vector_store
        self.model = model
        self.reranker = reranker
        self.query_rewriter = query_rewriter
        self.rerank_candidate_multiplier = rerank_candidate_multiplier

        self.api_key = api_key
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.has_api = True
        else:
            self.client = None
            self.has_api = False
            logger.warning("No Anthropic API key configured - running in demo mode")

        # Keyed by session_id so conversation history and retrieval scoping
        # stay consistent with each other (see VectorStore's session_id
        # metadata filtering).
        self.conversation_history: dict[str, list[dict]] = {}

    def query(
        self,
        session_id: str,
        question: str,
        n_results: int = 5,
        use_reranking: bool = True,
        conversation_context: bool = True,
    ) -> dict:
        history = self.conversation_history.setdefault(session_id, [])
        filter_dict = {"session_id": session_id}

        search_query = question
        if conversation_context and history and self.query_rewriter:
            search_query = self.query_rewriter.rewrite(question, history)
            if search_query != question:
                logger.info("Rewrote follow-up question for retrieval: %r -> %r", question, search_query)

        if use_reranking and self.reranker:
            candidates = self.vector_store.search(
                search_query,
                n_results=n_results * self.rerank_candidate_multiplier,
                filter_dict=filter_dict,
            )
            relevant_chunks = self.reranker.rerank(search_query, candidates, top_n=n_results)
        else:
            relevant_chunks = self.vector_store.search(search_query, n_results=n_results, filter_dict=filter_dict)

        if not relevant_chunks:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "sources": [],
                "retrieved_chunks": 0,
                "model": "none",
                "relevance_scores": [],
            }

        context = self._build_context(relevant_chunks)

        if self.has_api:
            answer = self._generate_with_claude(question, context, history if conversation_context else [])
            model = self.model
        else:
            answer = self._generate_demo_answer(question, relevant_chunks)
            model = "demo"

        if conversation_context:
            history.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": [chunk["metadata"]["source"] for chunk in relevant_chunks],
                }
            )

        return {
            "answer": answer,
            "sources": self._format_sources(relevant_chunks),
            "retrieved_chunks": len(relevant_chunks),
            "model": model,
            "relevance_scores": [chunk["relevance_score"] for chunk in relevant_chunks],
        }

    def _build_context(self, chunks: list[dict]) -> str:
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"]["source"]
            text = chunk["text"]
            score = chunk.get("relevance_score", 0)

            context_parts.append(f"[Source {i}: {source} (Relevance: {score:.2f})]\n{text}\n")

        return "\n".join(context_parts)

    def _generate_with_claude(self, question: str, context: str, history: list[dict]) -> str:
        system_prompt = """You are a helpful AI assistant that answers questions based on provided documents.

INSTRUCTIONS:
1. Answer the question using ONLY the information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite sources by mentioning [Source X] when using information
4. Be concise but comprehensive
5. If multiple sources provide relevant information, synthesize them
6. Do not make up information not present in the context"""

        user_message = f"""Context from documents:
{context}

Question: {question}

Please answer the question based on the context above. Cite your sources."""

        messages = []
        for exchange in history[-3:]:
            messages.append({"role": "user", "content": exchange["question"]})
            messages.append({"role": "assistant", "content": exchange["answer"]})

        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=system_prompt,
                messages=messages,
            )

            return response.content[0].text

        except Exception as e:
            logger.exception("Claude generation failed")
            return f"Error generating response: {e}"

    def _generate_demo_answer(self, question: str, chunks: list[dict]) -> str:
        answer_parts = [
            "**Demo Mode Response** (Configure API key for full Claude-powered answers)\n",
            f"\n**Question:** {question}\n",
            "\n**Relevant Information Found:**\n",
        ]

        for i, chunk in enumerate(chunks[:3], 1):
            source = chunk["metadata"]["source"]
            text = chunk["text"][:200]
            score = chunk.get("relevance_score", 0)

            answer_parts.append(f"\n[Source {i}: {source}] (Relevance: {score:.2f})")
            answer_parts.append(f"{text}...\n")

        answer_parts.append(
            "\n*To get AI-generated answers with Claude, add your ANTHROPIC_API_KEY to the .env file*"
        )

        return "".join(answer_parts)

    def _format_sources(self, chunks: list[dict]) -> list[dict]:
        sources = []

        for chunk in chunks:
            source = {
                "source": chunk["metadata"]["source"],
                "relevance": round(chunk.get("relevance_score", 0), 3),
                "chunk_id": chunk["id"],
                "preview": chunk["text"][:150] + "..." if len(chunk["text"]) > 150 else chunk["text"],
            }
            if "rerank_score" in chunk:
                source["rerank_score"] = round(chunk["rerank_score"], 3)
            sources.append(source)

        return sources

    def clear_history(self, session_id: str) -> None:
        self.conversation_history.pop(session_id, None)

    def get_conversation_history(self, session_id: str) -> list[dict]:
        return [
            {
                "question": exchange["question"],
                "answer": exchange["answer"],
                "sources": list(set(exchange["sources"])),
            }
            for exchange in self.conversation_history.get(session_id, [])
        ]

    def history_length(self, session_id: str) -> int:
        return len(self.conversation_history.get(session_id, []))
