from studybuddy.core.chunking import (
    chunk_document,
    recursive_split,
    _split_by_headings,
)
from studybuddy.core.documents import PageContent, ParsedDocument
from pathlib import Path


def _doc(text: str, title: str = "notes") -> ParsedDocument:
    return ParsedDocument(
        path=Path(f"/tmp/{title}.md"),
        title=title,
        pages=[PageContent(page=1, text=text)],
    )


def test_heading_breadcrumbs(sample_markdown):
    chunks = chunk_document(_doc(sample_markdown), chunk_size=400, overlap=50)
    sections = {c.section for c in chunks}
    assert any("Chapter 1" in s for s in sections)
    assert any("1.1 Requirements" in s for s in sections)
    # Chunks keep doc metadata
    assert all(c.doc_title == "notes" for c in chunks)
    assert all(c.page == 1 for c in chunks)
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_embed_text_has_context_prefix(sample_markdown):
    chunks = chunk_document(_doc(sample_markdown), chunk_size=400, overlap=50)
    c = next(c for c in chunks if "Requirements" in c.section)
    assert c.embed_text.startswith("Document: notes")
    assert "Section: " in c.embed_text
    assert c.embed_text.endswith(c.content)


def test_no_headings_still_chunks():
    text = "alpha bravo charlie " * 200
    chunks = chunk_document(_doc(text), chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(c.section == "" for c in chunks)
    joined = " ".join(c.content for c in chunks)
    assert "alpha" in joined and "charlie" in joined


def test_recursive_split_respects_size():
    text = "word " * 500  # 2500 chars
    parts = recursive_split(text, chunk_size=300, overlap=0)
    assert all(len(p) <= 300 for p in parts)
    assert " ".join(parts).split() == text.split()


def test_recursive_split_hard_cut_no_separators():
    text = "x" * 1000
    parts = recursive_split(text, chunk_size=300, overlap=0)
    assert all(len(p) <= 300 for p in parts)
    assert "".join(parts) == text


def test_overlap_carries_tail():
    text = " ".join(str(i) for i in range(200))
    parts = recursive_split(text, chunk_size=200, overlap=40)
    assert len(parts) > 1
    for prev, cur in zip(parts, parts[1:]):
        assert cur[:40] in prev or prev[-40:] == cur[:40]


def test_split_by_headings_nested(sample_markdown):
    sections = _split_by_headings(sample_markdown)
    crumbs = [c for c, _ in sections]
    assert "Chapter 1: Software Engineering" in crumbs
    assert "Chapter 1: Software Engineering > 1.1 Requirements" in crumbs
    assert "Chapter 2: Agile Methods" in crumbs


def test_citation_formats():
    doc = _doc("# T\n\nbody")
    chunks = chunk_document(doc, chunk_size=500, overlap=0)
    assert "notes" in chunks[0].citation()
    assert "p. 1" in chunks[0].citation()
