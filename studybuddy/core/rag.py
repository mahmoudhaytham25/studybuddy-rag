"""Retrieval + answer orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .config import Settings
from .embeddings import embed_query, rerank
from .providers import ChatProvider
from .store import ChunkHit, Store

_PROMPT_TEMPLATE = """\
Answer the question using only the numbered document excerpts below. \
Cite sources inline with [n] matching the excerpt numbers. If the \
excerpts don't contain the answer, say so and suggest what to try.

=== EXCERPTS ===
{context}

=== QUESTION ===
{question}
"""


@dataclass
class Answer:
    text: str
    sources: list[ChunkHit] = field(default_factory=list)


def retrieve(store: Store, settings: Settings, question: str) -> list[ChunkHit]:
    embedding = embed_query(question)
    hits = store.hybrid_search(question, embedding, k=settings.top_k * 3)
    if settings.rerank and len(hits) > settings.top_k:
        ranked = rerank(
            question, [h.content for h in hits], top_n=settings.top_k
        )
        hits = [hits[i] for i, _ in ranked]
    else:
        hits = hits[: settings.top_k]
    return hits


def build_prompt(question: str, hits: list[ChunkHit]) -> str:
    context = "\n\n".join(
        f"[{i}] ({h.citation()})\n{h.content}" for i, h in enumerate(hits, 1)
    )
    return _PROMPT_TEMPLATE.format(context=context, question=question)


def answer(
    store: Store, provider: ChatProvider, settings: Settings, question: str
) -> Answer:
    hits = retrieve(store, settings, question)
    if not hits:
        return Answer(
            "I don't have any indexed material yet — add some documents "
            "to your library first."
        )
    return Answer(provider.generate(build_prompt(question, hits)), hits)


def answer_stream(
    store: Store, provider: ChatProvider, settings: Settings, question: str
) -> tuple[Iterator[str], list[ChunkHit]]:
    """Return (token iterator, source hits) — hits are resolved eagerly so the
    caller can show citations immediately."""
    hits = retrieve(store, settings, question)
    if not hits:
        def empty() -> Iterator[str]:
            yield (
                "I don't have any indexed material yet — add some documents "
                "to your library first."
            )

        return empty(), []
    return provider.stream(build_prompt(question, hits)), hits
