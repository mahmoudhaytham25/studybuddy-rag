"""Structure-aware chunking.

Rather than fixed-size character windows, markdown text is first split along its
heading hierarchy — each chunk records a section breadcrumb like
"Lecture 3 > 3.2 Agile Methods" — and only then are oversized sections split
with a recursive separator fallback. This keeps chunks semantically coherent and
gives retrieval (and citations) real context instead of "somewhere in page 12".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .documents import PageContent, ParsedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    doc_path: str
    doc_title: str
    page: int | None
    section: str  # heading breadcrumb, "" if no headings seen
    content: str
    ord: int = 0

    @property
    def embed_text(self) -> str:
        """Text actually embedded: a lightweight contextual prefix (the cheap
        cousin of Anthropic-style contextual retrieval — no extra LLM calls)."""
        parts = [f"Document: {self.doc_title}"]
        if self.section:
            parts.append(f"Section: {self.section}")
        if self.page:
            parts.append(f"Page: {self.page}")
        return " | ".join(parts) + "\n" + self.content

    def citation(self) -> str:
        loc = f"p. {self.page}" if self.page else self.section or ""
        return f"{self.doc_title}{' — ' + loc if loc else ''}"


def chunk_document(
    doc: ParsedDocument, chunk_size: int = 1000, overlap: int = 200
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in doc.pages:
        chunks.extend(_chunk_page(doc, page, chunk_size, overlap))
    for i, c in enumerate(chunks):
        c.ord = i
    return chunks


def _chunk_page(
    doc: ParsedDocument, page: PageContent, chunk_size: int, overlap: int
) -> list[Chunk]:
    sections = _split_by_headings(page.text)
    out: list[Chunk] = []
    for breadcrumb, text in sections:
        for piece in recursive_split(text, chunk_size, overlap):
            piece = piece.strip()
            if piece:
                out.append(
                    Chunk(
                        doc_path=str(doc.path),
                        doc_title=doc.title,
                        page=page.page,
                        section=breadcrumb,
                        content=piece,
                    )
                )
    return out


def _split_by_headings(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into (breadcrumb, body) pairs at heading boundaries."""
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []  # index = heading level - 1
    buf: list[str] = []

    def flush() -> None:
        body = "".join(buf).strip()
        if body:
            sections.append((" > ".join(heading_stack), body))
        buf.clear()

    for line in markdown.splitlines(keepends=True):
        m = _HEADING_RE.match(line.rstrip("\n"))
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip().strip("#").strip()
            heading_stack[level - 1 :] = [title]
        else:
            buf.append(line)
    flush()
    return sections


def recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Classic recursive-character splitter: try paragraph, line, sentence,
    word, then hard character boundaries."""
    if len(text) <= chunk_size:
        return [text]

    separator = _pick_separator(text, chunk_size)
    if separator == "":
        pieces = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    else:
        raw = text.split(separator)
        pieces = _merge_splits(raw, separator, chunk_size)

    out: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            out.append(piece)
        else:
            out.extend(recursive_split(piece, chunk_size, overlap))
    if overlap > 0:
        out = _apply_overlap(out, chunk_size, overlap)
    return out


def _pick_separator(text: str, chunk_size: int) -> str:
    for sep in _DEFAULT_SEPARATORS:
        if sep == "":
            return ""
        head = text[: 4 * chunk_size]
        if sep in head:
            return sep
    return ""


def _merge_splits(splits: list[str], separator: str, chunk_size: int) -> list[str]:
    """Greedily pack adjacent splits up to chunk_size."""
    merged: list[str] = []
    current = ""
    for s in splits:
        candidate = s if not current else current + separator + s
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = s
    if current:
        merged.append(current)
    return merged


def _apply_overlap(chunks: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Prepend the tail of each chunk to the next for boundary continuity."""
    if len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        out.append(prev[-overlap:] + cur)
    return out
