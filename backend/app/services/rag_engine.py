"""Orchestrates retrieval and answer generation via the Claude API."""

import logging

import anthropic

from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        api_key: str | None = None,
        model: str = "claude-sonnet-5",
    ):
        self.vector_store = vector_store
        self.model = model

        self.api_key = api_key
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.has_api = True
        else:
            self.client = None
            self.has_api = False
            logger.warning("No Anthropic API key configured - running in demo mode")

        self.conversation_history: list[dict] = []

    def query(
        self,
        question: str,
        n_results: int = 5,
        use_hybrid: bool = True,
        conversation_context: bool = True,
    ) -> dict:
        if use_hybrid:
            relevant_chunks = self.vector_store.hybrid_search(question, n_results=n_results)
        else:
            relevant_chunks = self.vector_store.search(question, n_results=n_results)

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
            answer = self._generate_with_claude(question, context, conversation_context)
            model = self.model
        else:
            answer = self._generate_demo_answer(question, relevant_chunks)
            model = "demo"

        if conversation_context:
            self.conversation_history.append(
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

    def _generate_with_claude(self, question: str, context: str, use_history: bool) -> str:
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
        if use_history and self.conversation_history:
            for exchange in self.conversation_history[-3:]:
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
            sources.append(
                {
                    "source": chunk["metadata"]["source"],
                    "relevance": round(chunk.get("relevance_score", 0), 3),
                    "chunk_id": chunk["id"],
                    "preview": chunk["text"][:150] + "..." if len(chunk["text"]) > 150 else chunk["text"],
                }
            )

        return sources

    def clear_history(self) -> None:
        self.conversation_history = []

    def get_conversation_history(self) -> list[dict]:
        return [
            {
                "question": exchange["question"],
                "answer": exchange["answer"],
                "sources": list(set(exchange["sources"])),
            }
            for exchange in self.conversation_history
        ]
