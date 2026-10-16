"""Indexing pipeline — embeddings mocked (no model download in unit tests)."""

import pytest

import studybuddy.core.index as index_mod
from studybuddy.core.index import index_paths, needs_reindex, prune_missing
from studybuddy.core.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "i.db")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    monkeypatch.setattr(
        index_mod,
        "embed_texts",
        lambda texts: [[0.1] * 384 for _ in texts],
    )


def test_index_markdown(store, tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Lecture\n\n" + "Agile testing content. " * 50)
    result = index_paths(store, [f], chunk_size=500, chunk_overlap=50)
    assert result.indexed == [str(f)]
    assert store.count_chunks() > 0


def test_incremental_skips_unchanged(store, tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("content " * 100)
    index_paths(store, [f])
    result = index_paths(store, [f])
    assert result.skipped == [str(f)]
    assert not result.indexed
    # Changed content re-indexes
    f.write_text("different " * 100)
    result = index_paths(store, [f])
    assert result.indexed == [str(f)]
    assert not needs_reindex(store, f)


def test_failed_file_reported(store, tmp_path):
    bad = tmp_path / "bad.md"
    bad.write_text("   ")
    result = index_paths(store, [bad])
    assert str(bad) in result.failed


def test_prune_missing(store, tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text("text " * 50)
    b.write_text("text " * 50)
    index_paths(store, [a, b])
    removed = prune_missing(store, {str(a)})
    assert removed == [str(b)]
    assert [r.path for r in store.list_documents()] == [str(a)]


def test_progress_callback(store, tmp_path):
    f = tmp_path / "n.md"
    f.write_text("text " * 50)
    calls = []
    index_paths(store, [f], progress=lambda n, d, t: calls.append((n, d, t)))
    assert calls[-1] == ("", 1, 1)
