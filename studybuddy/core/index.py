"""Indexing pipeline: load -> chunk -> embed -> store, with hash-based
incremental reindexing (a changed file re-embeds; an unchanged one is skipped).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .chunking import chunk_document
from .documents import DocumentError, collect_files, file_sha256, load_document
from .embeddings import embed_texts
from .store import Store

_EMBED_BATCH = 64


@dataclass
class IndexResult:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


ProgressCb = Callable[[str, int, int], None]  # (filename, done, total)


def needs_reindex(store: Store, path: Path) -> bool:
    rec = store.get_document(str(path))
    if rec is None:
        return True
    try:
        return rec.sha256 != file_sha256(path)
    except OSError:
        return False


def index_paths(
    store: Store,
    paths: list[Path],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    progress: ProgressCb | None = None,
    force: bool = False,
    pdf_parser: str = "pymupdf4llm",
) -> IndexResult:
    files = collect_files(paths)
    result = IndexResult()
    total = len(files)
    for i, f in enumerate(files):
        if progress:
            progress(f.name, i, total)
        try:
            if not force and not needs_reindex(store, f):
                result.skipped.append(str(f))
                continue
            doc = load_document(f, pdf_parser=pdf_parser)
            chunks = chunk_document(doc, chunk_size, chunk_overlap)
            if not chunks:
                result.failed[str(f)] = "no usable text"
                continue
            embeddings = _embed_batched(c.embed_text for c in chunks)
            store.replace_document(
                str(f), doc.title, file_sha256(f), chunks, embeddings
            )
            result.indexed.append(str(f))
        except (DocumentError, OSError) as exc:
            result.failed[str(f)] = str(exc)
    if progress:
        progress("", total, total)
    return result


def _embed_batched(texts) -> list[list[float]]:
    out: list[list[float]] = []
    batch: list[str] = []
    for t in texts:
        batch.append(t)
        if len(batch) >= _EMBED_BATCH:
            out.extend(embed_texts(batch))
            batch.clear()
    if batch:
        out.extend(embed_texts(batch))
    return out


def prune_missing(store: Store, keep_paths: set[str]) -> list[str]:
    """Drop index entries for files no longer in the library."""
    removed = []
    for rec in store.list_documents():
        if rec.path not in keep_paths:
            store.delete_document(rec.path)
            removed.append(rec.path)
    return removed
