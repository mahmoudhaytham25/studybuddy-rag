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
    p = GeminiProvider("", "auto")
    ok, reason = p.is_available()
    assert not ok
    with pytest.raises(ProviderError, match="API key"):
        p.generate("q")


# -- Gemini fallback chain ----------------------------------------------------

import types as _types

import studybuddy.core.providers.gemini as gemini_mod


class ApiError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class FakeModels:
    """Records calls; each entry in `script` is an Exception to raise or a
    string to return as response.text."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        action = self.script.get(model, "ok-text")
        if isinstance(action, Exception):
            raise action
        return _types.SimpleNamespace(text=action)

    def generate_content_stream(self, model, contents, config):
        self.calls.append(model)
        action = self.script.get(model, ["ok "])
        if isinstance(action, Exception):
            raise action
        for chunk in action:
            if isinstance(chunk, Exception):
                raise chunk
            yield _types.SimpleNamespace(text=chunk)


def _provider(script, model="auto"):
    p = GeminiProvider("key", model)
    p._client = _types.SimpleNamespace(models=FakeModels(script))
    return p


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(gemini_mod.time, "sleep", lambda *_: None)


def test_models_to_try_auto_is_full_chain():
    p = GeminiProvider("key", "auto")
    assert p.models_to_try() == gemini_mod.fallback_models()
    assert p.models_to_try()[0] == "gemini-3.8-flash"


def test_models_to_try_named_first_then_chain():
    p = GeminiProvider("key", "gemini-2.5-flash")
    tried = p.models_to_try()
    assert tried[0] == "gemini-2.5-flash"
    assert tried.count("gemini-2.5-flash") == 1
    assert "gemini-3.8-flash" in tried


def test_generate_returns_text():
    p = _provider({"gemini-3.8-flash": "answer text"})
    assert p.generate("q") == "answer text"
    assert p._client.models.calls == ["gemini-3.8-flash"]


def test_404_hops_to_next_model():
    p = _provider({"gemini-3.8-flash": ApiError("not found", code=404)})
    assert p.generate("q") == "ok-text"
    assert p._client.models.calls[:2] == ["gemini-3.8-flash", "gemini-3.7-flash"]


def test_429_hops_instead_of_waiting():
    p = _provider({"gemini-3.8-flash": ApiError("quota", code=429)})
    assert p.generate("q") == "ok-text"
    assert p._client.models.calls == ["gemini-3.8-flash", "gemini-3.7-flash"]


def test_empty_response_hops():
    p = _provider({"gemini-3.8-flash": ""})
    assert p.generate("q") == "ok-text"


def test_400_fails_immediately_no_hop():
    p = _provider({"gemini-3.8-flash": ApiError("invalid request", code=400)})
    with pytest.raises(ProviderError):
        p.generate("q")
    assert p._client.models.calls == ["gemini-3.8-flash"]


def test_503_retries_same_tier_then_raises():
    p = _provider({"gemini-3.8-flash": ApiError("unavailable", code=503)})
    with pytest.raises(ProviderError):
        p.generate("q")
    # exhausted MAX_RETRIES on tier 1; transient errors never hop tiers
    assert (
        p._client.models.calls
        == ["gemini-3.8-flash"] * gemini_mod.MAX_RETRIES
    )


def test_stream_hops_before_first_token():
    p = _provider(
        {
            "gemini-3.8-flash": ApiError("not found", code=404),
            "gemini-3.7-flash": ["Hello ", "world"],
        }
    )
    assert "".join(p.stream("q")) == "Hello world"


def test_stream_mid_answer_failure_does_not_restart():
    p = _provider(
        {"gemini-3.8-flash": ["partial ", ApiError("boom", code=500)]}
    )
    with pytest.raises(ProviderError):
        list(p.stream("q"))


def test_fallback_env_override(monkeypatch):
    monkeypatch.setenv("STUDYBUDDY_GEMINI_FALLBACK_MODELS", "m1,m2")
    p = GeminiProvider("key", "auto")
    assert p.models_to_try() == ["m1", "m2"]
