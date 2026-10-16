"""Document loading: PDF -> structured Markdown via pymupdf4llm, plus .md/.txt.

Every loader returns a ParsedDocument with per-page text so citations can carry
page numbers. The markdown structure (headings, tables, lists) is preserved so
the chunker can split along section boundaries rather than mid-paragraph.

An optional Docling backend (better layout/table recovery, pulls PyTorch —
install the ``docling`` extra) can be selected per Settings.pdf_parser. It is
imported lazily so the app works — and packages — without it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .config import SUPPORTED_EXTENSIONS


class DocumentError(Exception):
    pass


@dataclass
class PageContent:
    page: int | None  # 1-based page number, None for unpaged formats
    text: str


@dataclass
class ParsedDocument:
    path: Path
    title: str
    pages: list[PageContent] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_document(
    path: Path, pdf_parser: str = "pymupdf4llm"
) -> ParsedDocument:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentError(f"Unsupported file type: {ext or path.name}")
    if not path.is_file():
        raise DocumentError(f"File not found: {path}")
    if ext == ".pdf":
        if pdf_parser == "docling":
            return _load_pdf_docling(path)
        if pdf_parser != "pymupdf4llm":
            raise DocumentError(f"Unknown PDF parser: {pdf_parser}")
        return _load_pdf(path)
    return _load_text(path)


def _load_pdf(path: Path) -> ParsedDocument:
    import pymupdf4llm

    try:
        page_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    except Exception as exc:  # pymupdf raises various file errors
        raise DocumentError(f"Could not parse PDF {path.name}: {exc}") from exc

    pages = [
        PageContent(page=i + 1, text=chunk.get("text") or "")
        for i, chunk in enumerate(page_chunks)
    ]
    if not any(p.text.strip() for p in pages):
        raise DocumentError(f"No extractable text in {path.name} (scanned PDF?)")
    return ParsedDocument(path=path, title=path.stem, pages=pages)


def _load_pdf_docling(path: Path) -> ParsedDocument:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise DocumentError(
            "Docling parser selected but not installed — "
            'pip install "studybuddy-rag[docling]"'
        ) from exc

    try:
        doc = DocumentConverter().convert(str(path)).document
    except Exception as exc:  # backend/model errors vary widely
        raise DocumentError(f"Could not parse PDF {path.name}: {exc}") from exc

    # Export per physical page so citations keep real page numbers: the
    # serializer filters items by provenance internally, unlike the
    # from_element/to_element slicing whose indices don't match
    # iterate_items() (table cells inflate the element count).
    page_nos = sorted(getattr(doc, "pages", None) or _prov_pages(doc))
    pages = [
        PageContent(page=n, text=doc.export_to_markdown(page_no=n).strip())
        for n in page_nos
    ]
    if not any(p.text for p in pages):
        raise DocumentError(f"No extractable text in {path.name} (scanned PDF?)")
    return ParsedDocument(path=path, title=path.stem, pages=pages)


def _prov_pages(doc) -> set[int]:
    """Page numbers that carry content, taken from item provenance (first
    provenance entry wins for items spanning pages)."""
    return {
        item.prov[0].page_no
        for item, _level in doc.iterate_items()
        if getattr(item, "prov", None)
    }


def _load_text(path: Path) -> ParsedDocument:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise DocumentError(f"Could not read {path.name}: {exc}") from exc
    if not text.strip():
        raise DocumentError(f"{path.name} is empty")
    return ParsedDocument(
        path=path, title=path.stem, pages=[PageContent(page=None, text=text)]
    )


def collect_files(paths: list[Path]) -> list[Path]:
    """Expand a mix of files and directories into a sorted list of supported files."""
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(
                f for f in sorted(p.rglob("*"))
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            out.append(p)
    return sorted(set(out))
