"""LLM provider abstraction — one narrow interface for answer generation."""

from __future__ import annotations

from typing import Iterator, Protocol


class ProviderError(Exception):
    """User-facing provider failure (bad key, unreachable server, ...)."""


class ChatProvider(Protocol):
    name: str

    def generate(self, prompt: str) -> str:
        """One-shot completion."""
        ...

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield text chunks as they arrive."""
        ...

    def is_available(self) -> tuple[bool, str]:
        """(ready, human-readable reason if not)."""
        ...
