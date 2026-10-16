"""RAG orchestration tests — embeddings/rerank mocked so no model downloads."""

import pytest

import studybuddy.core.rag as rag
from studybuddy.core.chunking import Chunk
from studybuddy.core.config import Settings
from studybuddy.core.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    chunks = [
        Chunk(
            doc_path=f"/tmp/d{i}.md",
            doc_title=f"doc{i}",
            page=i,
            section="S",
            content=f"chunk {i} about agile practices and software testing",
            ord=i,
        )
        for i in range(5)
    ]
    emb = [[0.1 + 0.01 * i] * 384 for i in range(5)]
    for i, c in enumerate(chunks):
        s.replace_document(c.doc_path, c.doc_title, "h", [c], [emb[i]])
    yield s
    s.close()


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "the answer [1]"

    def stream(self, prompt):
        self.last_prompt = prompt
        yield "the "
        yield "answer"

    def is_available(self):
        return True, ""


def test_retrieve_limits_to_top_k(store, monkeypatch):
    monkeypatch.setattr(rag, "embed_query", lambda q: [0.1] * 384)
    monkeypatch.setattr(rag, "rerank", lambda q, d, top_n: [(i, 1.0) for i in range(top_n)])
    s = Settings(top_k=2, rerank=True)
    hits = rag.retrieve(store, s, "agile")
    assert len(hits) == 2


def test_answer_includes_citations(store, monkeypatch):
    monkeypatch.setattr(rag, "embed_query", lambda q: [0.1] * 384)
    prov = FakeProvider()
    s = Settings(top_k=3, rerank=False)
    ans = rag.answer(store, prov, s, "what is agile?")
    assert ans.text == "the answer [1]"
    assert ans.sources
    assert "[1]" in prov.last_prompt
    assert "EXCERPTS" in prov.last_prompt


def test_answer_empty_library(tmp_path):
    store = Store(tmp_path / "e.db")
    prov = FakeProvider()
    ans = rag.answer(store, prov, Settings(), "anything")
    assert "indexed material" in ans.text
    store.close()


def test_prompt_formatting():
    hits = [
        rag.ChunkHit(
            chunk_id=1, doc_path="/a", doc_title="Lecture 1", page=7,
            section="Intro", content="body", score=0.9,
        )
    ]
    p = rag.build_prompt("q?", hits)
    assert "[1] (Lecture 1 — p. 7)" in p
    assert "q?" in p
