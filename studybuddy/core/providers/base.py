"""LLM provider abstraction — one narrow interface for answer generation."""

from __future__ import annotations

from typing import Iterator, Protocol, TypedDict


class ProviderError(Exception):
    """User-facing provider failure (bad key, unreachable server, ...)."""


class ChatMessage(TypedDict):
    """One prior conversation turn — plain user question or assistant
    answer text, never the retrieved-context-wrapped prompt."""

    role: str  # "user" | "assistant"
    content: str


class ChatProvider(Protocol):
    name: str

    def generate(
        self, prompt: str, history: list[ChatMessage] | None = None
    ) -> str:
        """One-shot completion. `history` carries prior chat turns."""
        ...

    def stream(
        self, prompt: str, history: list[ChatMessage] | None = None
    ) -> Iterator[str]:
        """Yield text chunks as they arrive."""
        ...

    def is_available(self) -> tuple[bool, str]:
        """(ready, human-readable reason if not)."""
        ...
