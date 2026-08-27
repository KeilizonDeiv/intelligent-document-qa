"""Orchestrates retrieval and streamed answer generation via the Claude API."""

import asyncio
import logging
from collections.abc import AsyncIterator

import anthropic

from app.services.query_rewriter import QueryRewriter
from app.services.reranker import Reranker
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided documents.

INSTRUCTIONS:
1. Answer the question using ONLY the information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite sources by mentioning [Source X] when using information
4. Be concise but comprehensive
5. If multiple sources provide relevant information, synthesize them
6. Do not make up information not present in the context"""


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
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self.has_api = True
        else:
            self.client = None
            self.has_api = False
            logger.warning("No Anthropic API key configured - running in demo mode")

        # Keyed by session_id so conversation history and retrieval scoping
        # stay consistent with each other (see VectorStore's session_id
        # metadata filtering).
        self.conversation_history: dict[str, list[dict]] = {}

    async def stream_query(
        self,
        session_id: str,
        question: str,
        n_results: int = 5,
        use_reranking: bool = True,
        conversation_context: bool = True,
    ) -> AsyncIterator[dict]:
        """Retrieve context and stream the answer as a sequence of events.

        Yields dicts shaped like the models in app.models.query:
        one "sources" event, then zero or more "token" events, then one
        "done" event (or an "error" event if generation fails after
        sources were already sent).
        """
        history = self.conversation_history.setdefault(session_id, [])
        filter_dict = {"session_id": session_id}

        search_query = question
        if conversation_context and history and self.query_rewriter:
            search_query = await asyncio.to_thread(self.query_rewriter.rewrite, question, history)
            if search_query != question:
                logger.info("Rewrote follow-up question for retrieval: %r -> %r", question, search_query)

        relevant_chunks = await self._retrieve(search_query, n_results, use_reranking, filter_dict)

        if not relevant_chunks:
            yield {"type": "sources", "sources": [], "retrieved_chunks": 0}
            yield {
                "type": "token",
                "text": "I couldn't find any relevant information in the documents to answer your question.",
            }
            yield {"type": "done", "model": "none"}
            return

        yield {
            "type": "sources",
            "sources": self._format_sources(relevant_chunks),
            "retrieved_chunks": len(relevant_chunks),
        }

        context = self._build_context(relevant_chunks)
        answer_parts: list[str] = []

        if self.has_api:
            try:
                async for token in self._stream_from_claude(
                    question, context, history if conversation_context else []
                ):
                    answer_parts.append(token)
                    yield {"type": "token", "text": token}
                model = self.model
            except Exception as e:
                logger.exception("Claude generation failed")
                yield {"type": "error", "message": str(e)}
                yield {"type": "done", "model": self.model}
                return
        else:
            demo_answer = self._generate_demo_answer(question, relevant_chunks)
            answer_parts.append(demo_answer)
            yield {"type": "token", "text": demo_answer}
            model = "demo"

        answer = "".join(answer_parts)
        if conversation_context:
            history.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": [chunk["metadata"]["source"] for chunk in relevant_chunks],
                }
            )

        yield {"type": "done", "model": model}

    async def _retrieve(
        self,
        search_query: str,
        n_results: int,
        use_reranking: bool,
        filter_dict: dict,
    ) -> list[dict]:
        if use_reranking and self.reranker:
            candidates = await asyncio.to_thread(
                self.vector_store.search,
                search_query,
                n_results * self.rerank_candidate_multiplier,
                filter_dict,
            )
            return await asyncio.to_thread(self.reranker.rerank, search_query, candidates, n_results)

        return await asyncio.to_thread(self.vector_store.search, search_query, n_results, filter_dict)

    def _build_context(self, chunks: list[dict]) -> str:
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"]["source"]
            text = chunk["text"]
            score = chunk.get("relevance_score", 0)

            context_parts.append(f"[Source {i}: {source} (Relevance: {score:.2f})]\n{text}\n")

        return "\n".join(context_parts)

    async def _stream_from_claude(self, question: str, context: str, history: list[dict]) -> AsyncIterator[str]:
        user_message = f"""Context from documents:
{context}

Question: {question}

Please answer the question based on the context above. Cite your sources."""

        messages = []
        for exchange in history[-3:]:
            messages.append({"role": "user", "content": exchange["question"]})
            messages.append({"role": "assistant", "content": exchange["answer"]})

        messages.append({"role": "user", "content": user_message})

        async with self.client.messages.stream(
            model=self.model,
            max_tokens=1500,
            system=_SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

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
