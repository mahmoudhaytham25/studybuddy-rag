"""UI smoke tests — run against the offscreen Qt platform."""

import os
import sys

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from studybuddy.core.config import Settings
from studybuddy.core.store import ChunkHit, Store


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "ui.db")
    yield s
    s.close()


def _hit(
    path="/tmp/notes.md", title="notes", page=None, section=""
) -> ChunkHit:
    return ChunkHit(
        chunk_id=1,
        doc_path=path,
        doc_title=title,
        page=page,
        section=section,
        content="chunk body",
        score=1.0,
    )


def test_main_window_builds(qapp, store):
    from studybuddy.ui.main_window import MainWindow

    w = MainWindow(Settings(document_paths=[]), store)
    assert w.windowTitle() == "StudyBuddy"
    assert w.docs.list.count() == 0
    w.close()


def test_chat_panel_flow(qapp, store):
    from studybuddy.ui.chat_panel import ChatPanel

    panel = ChatPanel()
    seen = []
    panel.ask.connect(seen.append)
    panel.input.setPlainText("hello?")
    panel._emit()
    assert seen == ["hello?"]
    assert panel.input.toPlainText() == ""

    panel.add_user_message("hello?")
    panel.start_answer()
    panel.append_token("partial")
    panel.append_token(" answer")
    panel.finish_answer([_hit(page=3)])
    html = panel.history.toHtml()
    assert "partial answer" in html
    assert "p. 3" in html
    # Source entries are clickable file links.
    assert "file:///tmp/notes.md" in html


def test_markdown_rendered_citations_kept(qapp, store):
    from studybuddy.ui.chat_panel import ChatPanel

    panel = ChatPanel()
    panel.add_user_message("q")
    panel.start_answer()
    panel.append_token("## Recap\n\n**bold** and `code` [1]\n\n- one\n- two")
    panel.finish_answer([_hit()])
    text = panel.history.toPlainText()
    html = panel.history.toHtml()
    # Rendered, not literal markdown syntax.
    assert "bold" in text and "**" not in text
    assert "Recap" in text and "##" not in text
    assert "<li" in html  # real list, not "- one" text
    # [n] citation markers survive verbatim.
    assert "[1]" in text
    # Sources section still appended.
    assert "Sources" in text


def test_chat_history_records_turns_and_caps(qapp, store):
    from studybuddy.ui.chat_panel import MAX_HISTORY_TURNS, ChatPanel

    panel = ChatPanel()
    total = MAX_HISTORY_TURNS + 2
    for i in range(total):
        panel.add_user_message(f"q{i}")
        panel.start_answer()
        panel.append_token(f"a{i}")
        panel.finish_answer([])
    hist = panel.chat_history()
    assert len(hist) == 2 * MAX_HISTORY_TURNS
    # Oldest turns dropped, most recent kept — plain text, not HTML.
    assert hist[0] == {"role": "user", "content": "q2"}
    assert hist[-2] == {"role": "user", "content": f"q{total - 1}"}
    assert hist[-1] == {"role": "assistant", "content": f"a{total - 1}"}
    assert "<" not in hist[-1]["content"]


def test_failed_answer_not_recorded(qapp, store):
    from studybuddy.ui.chat_panel import ChatPanel

    panel = ChatPanel()
    panel.add_user_message("q")
    panel.start_answer()
    panel.append_token("half")
    panel.add_error("boom")
    assert panel.chat_history() == []


def test_clear_chat_resets(qapp, store):
    from studybuddy.ui.chat_panel import ChatPanel

    panel = ChatPanel()
    panel.add_user_message("q")
    panel.start_answer()
    panel.append_token("a")
    panel.finish_answer([])
    assert panel.chat_history()
    panel.clear_chat()
    assert panel.chat_history() == []
    assert "q" not in panel.history.toPlainText()


def test_source_url_targets(qapp):
    from studybuddy.ui.chat_panel import source_url

    pdf = source_url("/tmp/My Doc.pdf", 3)
    assert pdf.startswith("file:///")
    assert pdf.endswith("#page=3")
    # Non-PDFs and missing pages open the file bare.
    assert source_url("/tmp/notes.md", 5) == "file:///tmp/notes.md"
    assert source_url("/tmp/x.pdf", None) == "file:///tmp/x.pdf"


def test_link_click_missing_file_notifies(qapp, monkeypatch):
    import studybuddy.ui.chat_panel as chat_panel_mod
    from studybuddy.ui.chat_panel import ChatPanel

    opened = []
    monkeypatch.setattr(chat_panel_mod, "_open_url", opened.append)
    panel = ChatPanel()
    seen = []
    panel.notice.connect(seen.append)
    panel._link_clicked(
        QUrl("file:///definitely/missing-file.pdf#page=2")
    )
    assert seen and "missing-file.pdf" in seen[0]
    assert opened == []


def test_link_click_existing_file_opens(qapp, monkeypatch, tmp_path):
    import studybuddy.ui.chat_panel as chat_panel_mod
    from studybuddy.ui.chat_panel import ChatPanel

    f = tmp_path / "doc.pdf"
    f.write_text("x")
    opened = []
    monkeypatch.setattr(chat_panel_mod, "_open_url", opened.append)
    panel = ChatPanel()
    url = QUrl.fromLocalFile(str(f))
    url.setFragment("page=2")
    panel._link_clicked(url)
    assert len(opened) == 1
    assert opened[0].fragment() == "page=2"


def test_settings_dialog_collects(qapp, store):
    from studybuddy.ui.settings_dialog import SettingsDialog

    s = Settings(provider="gemini")
    d = SettingsDialog(s)
    d.gemini_key.setText("k123")
    d._collect()
    assert s.gemini_api_key == "k123"
    assert s.provider == "gemini"
    d.close()
