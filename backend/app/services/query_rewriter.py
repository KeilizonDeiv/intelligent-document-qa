"""Rewrites follow-up questions into standalone queries for retrieval.

Without this, a follow-up like "what about the second one?" gets embedded
and searched literally - which retrieves nothing useful, because the
embedding has no idea what "the second one" refers to. Conversation history
is available to the *generation* call, but retrieval only ever sees the raw
question text. This service closes that gap: given recent history, it asks
a small/cheap model to rewrite the follow-up into a self-contained question
before it's used for embedding search. The rewritten form is used only for
retrieval - generation still sees the user's original wording.
"""

import logging

import anthropic

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """Given this recent conversation:

{history}

Rewrite the follow-up question into a standalone question that includes \
all context needed to understand it on its own, so it can be used as a \
search query without the conversation history. If the follow-up question \
is already standalone, return it unchanged. Reply with ONLY the rewritten \
question and nothing else.

Follow-up question: {question}"""


class QueryRewriter:
    def __init__(self, api_key: str | None, model: str = "claude-haiku-4-5"):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None

    def rewrite(self, question: str, history: list[dict]) -> str:
        """Return a standalone version of `question` given `history`.

        Falls back to the original question if there's no history, no API
        client configured, or the rewrite call fails for any reason -
        retrieval should never hard-fail because of this optimization.
        """
        if not self.client or not history:
            return question

        history_text = "\n".join(
            f"Q: {exchange['question']}\nA: {exchange['answer']}" for exchange in history[-3:]
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[
                    {
                        "role": "user",
                        "content": _REWRITE_PROMPT.format(history=history_text, question=question),
                    }
                ],
            )
            rewritten = response.content[0].text.strip()
            return rewritten or question
        except Exception:
            logger.exception("Query rewrite failed, falling back to original question")
            return question
