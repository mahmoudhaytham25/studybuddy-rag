"""Embedding and reranking models via fastembed (ONNX runtime — no PyTorch,
which keeps the packaged app small). Models download on first use and are
cached under the app data dir so the app works fully offline afterwards.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .config import EMBEDDING_MODEL, RERANKER_MODEL, data_dir

_embedder = None
_reranker = None


def _cache_dir() -> Path:
    d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(
            model_name=EMBEDDING_MODEL, cache_dir=str(_cache_dir())
        )
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _reranker = TextCrossEncoder(
            model_name=RERANKER_MODEL, cache_dir=str(_cache_dir())
        )
    return _reranker


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    embedder = _get_embedder()
    return [v.tolist() for v in embedder.embed(list(texts))]


def embed_query(text: str) -> list[float]:
    embedder = _get_embedder()
    return next(iter(embedder.query_embed(text))).tolist()


def rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    """Return (index, score) pairs for the best top_n documents."""
    if not documents:
        return []
    model = _get_reranker()
    scored = sorted(
        enumerate(model.rerank(query, documents)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return [(i, float(s)) for i, s in scored[:top_n]]


def preload(reranker: bool = False) -> None:
    """Force model downloads up front (first-run experience / warmup)."""
    _get_embedder()
    if reranker:
        _get_reranker()


def reset_for_tests() -> None:
    global _embedder, _reranker
    _embedder = _reranker = None
    os.environ.pop("FASTEMBED_CACHE_DIR", None)
