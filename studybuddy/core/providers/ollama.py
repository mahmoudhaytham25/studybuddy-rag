"""Ollama provider — fully local answering via a running Ollama server.
Plain HTTP against the native API; no SDK dependency.
"""

from __future__ import annotations

import json
from typing import Iterator

import httpx

from .base import ProviderError
from .gemini import _SYSTEM


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _payload(self, prompt: str, stream: bool) -> dict:
        return {
            "model": self.model,
            "stream": stream,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.2},
        }

    def generate(self, prompt: str) -> str:
        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json=self._payload(prompt, stream=False),
                timeout=120,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(self._explain(exc)) from exc
        return resp.json().get("message", {}).get("content", "")

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=self._payload(prompt, stream=True),
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise ProviderError(self._explain(exc)) from exc

    def is_available(self) -> tuple[bool, str]:
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            resp.raise_for_status()
        except httpx.HTTPError:
            return False, f"Ollama not reachable at {self.base_url}"
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        if not any(m == self.model or m.startswith(self.model + ":")
                   for m in models):
            return False, f"model '{self.model}' not pulled in Ollama"
        return True, ""

    def _explain(self, exc: httpx.HTTPError) -> str:
        if isinstance(exc, httpx.ConnectError):
            return (
                f"Cannot reach Ollama at {self.base_url} — is it running?"
            )
        return f"Ollama request failed: {exc}"
