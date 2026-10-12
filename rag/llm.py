"""LLM backends.

``GroqLLM`` talks to any OpenAI-compatible chat endpoint (Groq by default —
change ``LLM_BASE_URL``/``LLM_MODEL`` to point elsewhere). When no API key is
configured the pipeline falls back to ``MockLLM``, a deterministic stub that
answers straight from the retrieved context. This keeps the whole app —
including tests — runnable with zero secrets.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Protocol

from rag.config import Settings

logger = logging.getLogger(__name__)

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class ChatLLM(Protocol):
    name: str
    is_mock: bool

    def complete(self, messages: list[Message]) -> str: ...

    def stream(self, messages: list[Message]) -> Iterator[str]: ...


class GroqLLM:
    is_mock = False

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.groq_api_key, base_url=settings.llm_base_url
        )
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self.name = f"groq/{self._model}"

    def complete(self, messages: list[Message]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta


class MockLLM:
    """Deterministic extractive answer built from the retrieved context."""

    is_mock = True
    name = "mock-extractive"

    def _build(self, messages: list[Message]) -> str:
        question = messages[-1]["content"]
        context = ""
        marker = "NGỮ CẢNH:\n"
        if marker in question:
            context = question.split(marker, 1)[1].split("\n\nCÂU HỎI:", 1)[0]
        excerpt = context.strip()
        if not excerpt:
            return (
                "Tôi không tìm thấy thông tin này trong tài liệu của shop. "
                "Bạn vui lòng liên hệ trực tiếp để được hỗ trợ nhé."
            )
        if len(excerpt) > 400:
            excerpt = excerpt[:400].rsplit(" ", 1)[0] + "..."
        return f"Dựa trên tài liệu của shop:\n\n{excerpt}"

    def complete(self, messages: list[Message]) -> str:
        return self._build(messages)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        for word in self._build(messages).split(" "):
            yield word + " "


def build_llm(settings: Settings) -> ChatLLM:
    if settings.groq_api_key:
        return GroqLLM(settings)
    logger.warning(
        "GROQ_API_KEY not set — using MockLLM (deterministic extractive answers)."
    )
    return MockLLM()
