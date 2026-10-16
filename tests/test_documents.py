import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from studybuddy.core.config import Settings
from studybuddy.core.documents import (
    DocumentError,
    collect_files,
    file_sha256,
    load_document,
)


def _make_pdf(path: Path, texts: list[str]) -> None:
    doc = pymupdf.open()
    for t in texts:
        page = doc.new_page()
        page.insert_text((72, 72), t)
    doc.save(path)
    doc.close()


def test_load_pdf_pages(tmp_path):
    pdf = tmp_path / "lec.pdf"
    _make_pdf(pdf, ["# Intro\nSoftware engineering basics", "Second page content"])
    doc = load_document(pdf)
    assert doc.title == "lec"
    assert len(doc.pages) == 2
    assert doc.pages[0].page == 1
    assert "Software engineering" in doc.pages[0].text


def test_load_md_and_txt(tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# Title\n\nbody text", encoding="utf-8")
    txt = tmp_path / "notes.txt"
    txt.write_text("plain text body", encoding="utf-8")
    for f in (md, txt):
        doc = load_document(f)
        assert doc.pages[0].page is None
        assert doc.text.strip()


def test_empty_file_raises(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("   ", encoding="utf-8")
    with pytest.raises(DocumentError):
        load_document(f)


def test_unsupported_and_missing(tmp_path):
    with pytest.raises(DocumentError):
        load_document(tmp_path / "x.docx")
    with pytest.raises(DocumentError):
        load_document(tmp_path / "ghost.md")


def test_collect_files_mixed(tmp_path):
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-")
    (tmp_path / "c.docx").write_bytes(b"zip")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.txt").write_text("x")
    files = collect_files([tmp_path])
    names = {f.name for f in files}
    assert names == {"a.md", "b.pdf", "d.txt"}


def test_sha256_stable(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert file_sha256(f) == file_sha256(f)
    f.write_text("changed")
    rec = file_sha256(f)
    f.write_text("hello")
    assert file_sha256(f) != rec


def _install_fake_docling(
    monkeypatch: pytest.MonkeyPatch, page_texts: dict[int, str]
) -> None:
    """Register a stub `docling` package in sys.modules — exposes just the
    surface _load_pdf_docling uses (DocumentConverter().convert() ->
    result.document with .pages and .export_to_markdown(page_no=...))."""

    class FakeDoc:
        pages = {n: object() for n in page_texts}

        def export_to_markdown(self, page_no=None):
            return page_texts.get(page_no, "")

    class FakeConverter:
        def convert(self, _src):
            return SimpleNamespace(document=FakeDoc())

    docling = types.ModuleType("docling")
    converter = types.ModuleType("docling.document_converter")
    converter.DocumentConverter = FakeConverter
    docling.document_converter = converter
    monkeypatch.setitem(sys.modules, "docling", docling)
    monkeypatch.setitem(sys.modules, "docling.document_converter", converter)


def test_docling_dispatch_keeps_real_page_numbers(tmp_path, monkeypatch):
    pdf = tmp_path / "tables.pdf"
    _make_pdf(pdf, ["irrelevant — backend is stubbed"])
    _install_fake_docling(
        monkeypatch,
        {1: "# Title\n\nPage one", 2: "", 3: "| a | b |\n|---|---|\n| 1 | 2 |"},
    )
    doc = load_document(pdf, pdf_parser="docling")
    assert [p.page for p in doc.pages] == [1, 2, 3]
    assert "Page one" in doc.pages[0].text
    assert doc.pages[1].text == ""  # contentless page still gets a PageContent
    assert "| a | b |" in doc.pages[2].text


def test_docling_selected_but_missing(tmp_path, monkeypatch):
    pdf = tmp_path / "x.pdf"
    _make_pdf(pdf, ["hello"])
    # None in sys.modules forces ImportError even if docling were installed.
    monkeypatch.setitem(sys.modules, "docling", None)
    with pytest.raises(DocumentError, match=r"studybuddy-rag\[docling\]"):
        load_document(pdf, pdf_parser="docling")


def test_unknown_pdf_parser_rejected(tmp_path):
    pdf = tmp_path / "x.pdf"
    _make_pdf(pdf, ["hi"])
    with pytest.raises(DocumentError, match="Unknown PDF parser"):
        load_document(pdf, pdf_parser="bogus")


def test_settings_pdf_parser_roundtrip(tmp_path):
    cfg = tmp_path / "settings.json"
    s = Settings(pdf_parser="docling")
    s.save(cfg)
    assert Settings.load(cfg).pdf_parser == "docling"
    # Unknown keys still ignored; absent key falls back to default.
    cfg.write_text('{"provider": "ollama"}', encoding="utf-8")
    assert Settings.load(cfg).pdf_parser == "pymupdf4llm"


def test_docling_real_parse(tmp_path):
    pytest.importorskip("docling")  # optional extra — skipped by default
    pdf = tmp_path / "real.pdf"
    _make_pdf(pdf, ["Docling real parse text"])
    doc = load_document(pdf, pdf_parser="docling")
    assert doc.pages[0].page == 1
    assert "Docling real parse text" in doc.text
