"""UI smoke tests — run against the offscreen Qt platform."""

import os
import sys

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from studybuddy.core.config import Settings
from studybuddy.core.store import Store


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "ui.db")
    yield s
    s.close()


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
    panel.finish_answer(["doc — p. 3"])
    html = panel.history.toHtml()
    assert "partial answer" in html and "doc" in html


def test_settings_dialog_collects(qapp, store):
    from studybuddy.ui.settings_dialog import SettingsDialog

    s = Settings(provider="gemini")
    d = SettingsDialog(s)
    d.gemini_key.setText("k123")
    d._collect()
    assert s.gemini_api_key == "k123"
    assert s.provider == "gemini"
    d.close()
