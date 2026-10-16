from pathlib import Path

import pymupdf
import pytest

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
