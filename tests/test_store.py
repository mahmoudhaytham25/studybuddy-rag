import pytest

from studybuddy.core.chunking import Chunk
from studybuddy.core.store import Store, _fts_query


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def _chunks(n: int, prefix: str = "content") -> list[Chunk]:
    return [
        Chunk(
            doc_path="/tmp/doc.md",
            doc_title="doc",
            page=1,
            section="S1",
            content=f"{prefix} number {i} about agile testing methods",
            ord=i,
        )
        for i in range(n)
    ]


def _embeds(n: int, seed: float = 0.1) -> list[list[float]]:
    return [[seed + i * 0.001] * 384 for i in range(n)]


def test_replace_and_list(store):
    store.replace_document("/tmp/doc.md", "doc", "hash1", _chunks(3), _embeds(3))
    rec = store.get_document("/tmp/doc.md")
    assert rec.n_chunks == 3 and rec.sha256 == "hash1"
    assert store.count_chunks() == 3
    assert store.list_documents()[0].title == "doc"


def test_replace_is_clean(store):
    store.replace_document("/tmp/doc.md", "doc", "h1", _chunks(5), _embeds(5))
    store.replace_document("/tmp/doc.md", "doc", "h2", _chunks(2), _embeds(2))
    assert store.count_chunks() == 2
    assert store.get_document("/tmp/doc.md").sha256 == "h2"


def test_delete_document(store):
    store.replace_document("/tmp/doc.md", "doc", "h1", _chunks(3), _embeds(3))
    store.delete_document("/tmp/doc.md")
    assert store.count_chunks() == 0
    assert store.get_document("/tmp/doc.md") is None


def test_vector_search_returns_ids(store):
    store.replace_document("/tmp/doc.md", "doc", "h", _chunks(3), _embeds(3))
    hits = store.vector_search([0.1] * 384, k=2)
    assert len(hits) == 2
    assert all(isinstance(h[0], int) for h in hits)


def test_fts_search_keyword(store):
    chunks = _chunks(3)
    chunks[0].content = "unique zebra terminology appears here"
    store.replace_document("/tmp/doc.md", "doc", "h", chunks, _embeds(3))
    hits = store.fts_search("zebra terminology", k=5)
    assert hits  # found via bm25
    assert _fts_query("zebra!!! ---") == '"zebra"'


def test_hybrid_search_returns_hits(store):
    store.replace_document("/tmp/doc.md", "doc", "h", _chunks(4), _embeds(4))
    hits = store.hybrid_search("agile testing", [0.1] * 384, k=3)
    assert len(hits) == 3
    assert hits[0].doc_title == "doc"
    assert "agile" in hits[0].content


def test_mismatched_lengths_rejected(store):
    with pytest.raises(ValueError):
        store.replace_document(
            "/tmp/doc.md", "doc", "h", _chunks(2), _embeds(1)
        )
