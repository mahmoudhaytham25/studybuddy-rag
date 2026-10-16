import json

import httpx
import pytest

from studybuddy.core.config import Settings
from studybuddy.core.providers import (
    GeminiProvider,
    OllamaProvider,
    ProviderError,
    make_provider,
)


def test_factory_selects_provider():
    s = Settings(provider="ollama")
    assert isinstance(make_provider(s), OllamaProvider)
    s.provider = "gemini"
    assert isinstance(make_provider(s), GeminiProvider)


def test_ollama_unavailable_without_server():
    p = OllamaProvider("http://localhost:9", "llama3.1:8b")
    ok, reason = p.is_available()
    assert not ok and "Ollama" in reason


def test_ollama_generate_parses_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "hi there"}}
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Client(
        transport=transport).post(*a, **k))
    p = OllamaProvider("http://x", "m")
    assert p.generate("q") == "hi there"


def test_ollama_stream_yields_tokens(monkeypatch):
    lines = [
        json.dumps({"message": {"content": "Hello"}, "done": False}),
        json.dumps({"message": {"content": " world"}, "done": False}),
        json.dumps({"done": True}),
    ]

    class FakeResp:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(lines)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: FakeResp())
    p = OllamaProvider("http://x", "m")
    assert list(p.stream("q")) == ["Hello", " world"]


def test_ollama_connect_error_message(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)
    p = OllamaProvider("http://x", "m")
    with pytest.raises(ProviderError, match="Cannot reach Ollama"):
        p.generate("q")


def test_gemini_requires_key():
    p = GeminiProvider("", "gemini-2.5-flash")
    ok, reason = p.is_available()
    assert not ok
    with pytest.raises(ProviderError, match="API key"):
        p.generate("q")
