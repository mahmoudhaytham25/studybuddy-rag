"""Gemini provider via the google-genai SDK, with a model fallback chain.

Ported from the tour-guide model-server (models/gemini.py + models/llm.py):
each Gemini model carries a SEPARATE quota counter, so when the primary
hits a 429 the request retries on a lower-grade model instead of burning a
65s wait on a quota that can't recover. Model-name errors (404 /
unsupported) and empty responses (safety blocks) also hop the chain —
retrying them on the same model can only fail again. Genuine transient
errors (5xx, timeouts) retry on the same tier with short exponential
backoff.

Set model to "auto" to start at the head of the chain, or name a specific
model to try it first and then cascade downward.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterator

from .base import ChatMessage, ProviderError

logger = logging.getLogger(__name__)

AUTO_MODEL = "auto"

# Ordered descending by generation — verified against the models list
# endpoint for a live key. Override with STUDYBUDDY_GEMINI_FALLBACK_MODELS.
_DEFAULT_FALLBACK = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# Per-minute quota exhaustion needs a long wait, not jitter — free-tier
# quotas reset every 60s.
RATE_LIMIT_WAIT_SECONDS = 65.0
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0
GENERATE_TIMEOUT_MS = 120_000

_SYSTEM = (
    "You are StudyBuddy, a study assistant that answers questions strictly "
    "from the provided document excerpts. If the answer is not in the "
    "excerpts, say so plainly instead of guessing. Cite sources inline like "
    "[1], [2] matching the numbered excerpts."
)


def fallback_models() -> list[str]:
    configured = os.environ.get("STUDYBUDDY_GEMINI_FALLBACK_MODELS", "")
    if configured.strip():
        return [m.strip() for m in configured.split(",") if m.strip()]
    return list(_DEFAULT_FALLBACK)


class EmptyResponseError(RuntimeError):
    """Response carried no text — safety block, RECITATION, or MAX_TOKENS
    with no parts. A tier failure: hop the chain rather than surface it."""


def _error_status_code(error: Exception) -> int | None:
    code = getattr(error, "code", None)
    return code if isinstance(code, int) and code > 0 else None


def is_rate_limit_error(error: Exception) -> bool:
    code = _error_status_code(error)
    if code is not None:
        return code == 429
    message = str(error).lower()
    return "429" in message or "quota" in message


def is_model_unavailable(error: Exception) -> bool:
    code = _error_status_code(error)
    if code == 404:
        return True
    message = str(error).lower()
    return (
        "not found" in message
        or "not supported" in message
        or "does not exist" in message
    )


def is_retryable_error(error: Exception) -> bool:
    """429/5xx/timeouts are transient; other 4xx rejections fail
    identically on every attempt."""
    code = _error_status_code(error)
    if code is not None:
        return code == 429 or code >= 500
    message = str(error).lower()
    if is_rate_limit_error(error):
        return True
    if any(
        m in message
        for m in ("400", "401", "403", "404", "invalid", "permission",
                  "not found")
    ):
        return False
    return any(
        m in message
        for m in ("500", "502", "503", "504", "internal", "unavailable",
                  "timeout", "timed out", "deadline")
    )


def _should_hop_tier(error: Exception, has_fallback: bool) -> bool:
    return has_fallback and (
        isinstance(error, EmptyResponseError)
        or is_rate_limit_error(error)
        or is_model_unavailable(error)
    )


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str = AUTO_MODEL):
        self.api_key = api_key
        self.model = model or AUTO_MODEL
        self._client = None

    # -- plumbing -------------------------------------------------------------

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
            system_instruction=_SYSTEM,
            temperature=0.2,
            http_options=types.HttpOptions(timeout=GENERATE_TIMEOUT_MS),
        )

    def models_to_try(self) -> list[str]:
        """[configured model] + remaining chain, or the full chain on auto."""
        chain = fallback_models()
        if self.model == AUTO_MODEL:
            return chain
        return [self.model] + [m for m in chain if m != self.model]

    # -- generation -----------------------------------------------------------

    @staticmethod
    def _contents(prompt: str, history: list[ChatMessage] | None = None):
        """The `contents` argument for generate_content: the bare prompt
        string when there's no history (unchanged request shape), else a
        role-labelled Content list so prior turns become real chat context
        ("assistant" maps to Gemini's "model" role)."""
        if not history:
            return prompt
        from google.genai import types

        contents = [
            types.Content(
                role="model" if turn.get("role") == "assistant" else "user",
                parts=[types.Part.from_text(text=turn.get("content", ""))],
            )
            for turn in history
        ]
        contents.append(
            types.Content(
                role="user", parts=[types.Part.from_text(text=prompt)]
            )
        )
        return contents

    def generate(
        self, prompt: str, history: list[ChatMessage] | None = None
    ) -> str:
        def call(model: str) -> str:
            resp = self._get_client().models.generate_content(
                model=model,
                contents=self._contents(prompt, history),
                config=self._config(),
            )
            text = resp.text
            if not text or not text.strip():
                raise EmptyResponseError(
                    f"{model} returned an empty response (safety block or "
                    f"no text parts)"
                )
            return text

        return self._run_chain(call)

    def stream(
        self, prompt: str, history: list[ChatMessage] | None = None
    ) -> Iterator[str]:
        yielded_any = False

        def call(model: str):
            for event in self._get_client().models.generate_content_stream(
                model=model,
                contents=self._contents(prompt, history),
                config=self._config(),
            ):
                if event.text:
                    yield event.text

        for model in self.models_to_try():
            delay = INITIAL_BACKOFF_SECONDS
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    for token in call(model):
                        yielded_any = True
                        yield token
                    return
                except Exception as exc:
                    if yielded_any:
                        # Tokens already shown — a mid-answer restart would
                        # garble the transcript. Surface the failure.
                        raise self._provider_error(exc) from exc
                    if self._hop_or_raise(exc, model, attempt, delay):
                        break  # next model tier
                    delay *= 2
        raise ProviderError(
            "Gemini fallback chain exhausted — all models failed"
        )

    # -- chain driver -----------------------------------------------------------

    def _run_chain(self, call):
        """Run `call(model)` across the fallback chain for one-shot calls."""
        last_error: Exception | None = None
        for model in self.models_to_try():
            delay = INITIAL_BACKOFF_SECONDS
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    return call(model)
                except ProviderError:
                    raise
                except Exception as exc:
                    last_error = exc
                    if self._hop_or_raise(exc, model, attempt, delay):
                        break  # next model tier
                    delay *= 2
        raise self._provider_error(
            last_error or RuntimeError("Gemini fallback chain exhausted")
        )

    def _hop_or_raise(
        self, exc: Exception, model: str, attempt: int, delay: float
    ) -> bool:
        """Decide the fate of a failed attempt. Returns True to hop to the
        next model tier; returns False after sleeping (retry same tier);
        raises when the error should surface immediately."""
        models = self.models_to_try()
        model_index = models.index(model)
        has_fallback = model_index < len(models) - 1
        if _should_hop_tier(exc, has_fallback):
            if isinstance(exc, EmptyResponseError):
                reason = "empty response"
            elif is_rate_limit_error(exc):
                reason = "rate-limited"
            else:
                reason = "unavailable"
            logger.warning(
                "%s %s — falling back to %s",
                model, reason, models[model_index + 1],
            )
            return True
        if attempt == MAX_RETRIES or not is_retryable_error(exc):
            raise self._provider_error(exc) from exc
        wait = RATE_LIMIT_WAIT_SECONDS if is_rate_limit_error(exc) else delay
        logger.warning(
            "%s failed (attempt %d/%d): %s — retrying in %.0fs",
            model, attempt, MAX_RETRIES, str(exc)[:200], wait,
        )
        time.sleep(wait)
        return False

    # -- misc -----------------------------------------------------------------

    def is_available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "Gemini API key not set"
        return True, ""

    @staticmethod
    def _provider_error(exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        return ProviderError(f"Gemini request failed: {exc}")
