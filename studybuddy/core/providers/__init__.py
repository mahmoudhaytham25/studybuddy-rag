"""Answer-generation providers. Factory builds whichever the settings select."""

from __future__ import annotations

from ..config import Settings
from .base import ChatProvider, ProviderError
from .gemini import GeminiProvider
from .ollama import OllamaProvider

__all__ = [
    "ChatProvider",
    "ProviderError",
    "GeminiProvider",
    "OllamaProvider",
    "make_provider",
]


def make_provider(settings: Settings) -> ChatProvider:
    if settings.provider == "ollama":
        return OllamaProvider(settings.ollama_base_url, settings.ollama_model)
    return GeminiProvider(settings.gemini_api_key, settings.gemini_model)
