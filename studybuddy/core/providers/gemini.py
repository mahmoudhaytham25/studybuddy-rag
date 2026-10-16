"""Gemini provider via the google-genai SDK."""

from __future__ import annotations

from typing import Iterator

from .base import ProviderError

_SYSTEM = (
    "You are StudyBuddy, a study assistant that answers questions strictly "
    "from the provided document excerpts. If the answer is not in the "
    "excerpts, say so plainly instead of guessing. Cite sources inline like "
    "[1], [2] matching the numbered excerpts."
)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ProviderError(
                    "No Gemini API key configured — set one in Settings."
                )
            try:
                from google import genai
            except ImportError as exc:
                raise ProviderError("google-genai is not installed") from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _config(self):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=_SYSTEM, temperature=0.2
        )

    def generate(self, prompt: str) -> str:
        try:
            resp = self._get_client().models.generate_content(
                model=self.model, contents=prompt, config=self._config()
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Gemini request failed: {exc}") from exc
        return resp.text or ""

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            for event in self._get_client().models.generate_content_stream(
                model=self.model, contents=prompt, config=self._config()
            ):
                if event.text:
                    yield event.text
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Gemini stream failed: {exc}") from exc

    def is_available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "Gemini API key not set"
        return True, ""
