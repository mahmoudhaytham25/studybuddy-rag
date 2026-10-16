"""Main window: library panel + chat panel, owns Store/Settings/workers."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow, QMessageBox, QSplitter

from ..core.config import Settings
from ..core.index import prune_missing
from ..core.providers import make_provider
from ..core.store import Store
from .chat_panel import ChatPanel
from .document_panel import DocumentPanel
from .settings_dialog import SettingsDialog
from .workers import ChatWorker, IndexWorker


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, store: Store):
        super().__init__()
        self.settings = settings
        self.store = store
        self._workers: list = []  # keep refs so threads aren't GC'd

        self.setWindowTitle("StudyBuddy")
        self.resize(1080, 700)

        splitter = QSplitter(Qt.Horizontal)
        self.docs = DocumentPanel(store)
        self.chat = ChatPanel()
        splitter.addWidget(self.docs)
        splitter.addWidget(self.chat)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 780])
        self.setCentralWidget(splitter)

        settings_action = QAction("Settings…", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self._open_settings)
        self.addAction(settings_action)
        self.menuBar().addMenu("&File").addAction(settings_action)

        self.docs.add_requested.connect(self._index_paths)
        self.docs.remove_requested.connect(self._remove_doc)
        self.chat.ask.connect(self._ask)

        # Index whatever is already in the library (and prune vanished files).
        if settings.document_paths:
            self._index_paths([Path(p) for p in settings.document_paths])

    # -- indexing -----------------------------------------------------------

    def _index_paths(self, paths: list[Path]) -> None:
        known = set(self.settings.document_paths)
        known.update(str(p) for p in paths)
        self.settings.document_paths = sorted(known)
        self.settings.save()
        self.docs.set_busy("…", 0, 1)
        self._spawn(
            IndexWorker(self.store, self.settings, paths),
            progressed=self.docs.set_busy,
            finished_ok=self._index_done,
            failed=lambda msg: self.docs.set_idle(f"Indexing failed: {msg}"),
        )

    def _index_done(self, result) -> None:
        prune_missing(self.store, set(self.settings.document_paths))
        parts = [f"{len(result.indexed)} indexed"]
        if result.skipped:
            parts.append(f"{len(result.skipped)} unchanged")
        if result.failed:
            parts.append(f"{len(result.failed)} failed")
        summary = " · ".join(parts)
        if result.failed:
            first = next(iter(result.failed.items()))
            summary += f" — {Path(first[0]).name}: {first[1]}"
        self.docs.set_idle(summary)

    def _remove_doc(self, path: str) -> None:
        self.store.delete_document(path)
        self.settings.document_paths = [
            p for p in self.settings.document_paths if p != path
        ]
        self.settings.save()
        self.docs.reload()

    # -- chat ---------------------------------------------------------------

    def _ask(self, question: str) -> None:
        provider = make_provider(self.settings)
        ok, reason = provider.is_available()
        if not ok:
            QMessageBox.warning(self, "Provider not ready", reason)
            self._open_settings()
            return
        self.chat.add_user_message(question)
        self.chat.start_answer()
        self.chat.set_busy(True)
        self._spawn(
            ChatWorker(self.store, provider, self.settings, question),
            token=self.chat.append_token,
            done=self._answer_done,
            failed=self._answer_failed,
        )

    def _answer_done(self, hits) -> None:
        self.chat.finish_answer([h.citation() for h in hits])
        self.chat.set_busy(False)

    def _answer_failed(self, msg: str) -> None:
        self.chat.add_error(msg)
        self.chat.set_busy(False)

    # -- misc -----------------------------------------------------------------

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        dialog.exec()

    def _spawn(self, worker, **signals) -> None:
        for name, slot in signals.items():
            getattr(worker, name).connect(slot)
        self._workers.append(worker)
        worker.finished.connect(lambda: self._workers.remove(worker))
        worker.start()

    def closeEvent(self, event) -> None:
        for w in list(self._workers):
            w.wait(3000)
        self.store.close()
        super().closeEvent(event)


_STYLESHEET = """
QMainWindow, QDialog { background: #1e1f24; }
QWidget { color: #e8e6e1; font-size: 13px; }
#panelTitle { font-size: 16px; font-weight: 600; padding: 4px 0 8px 0; }
#statusLabel { color: #9b978d; font-size: 12px; }
QListWidget, QTextBrowser, QTextEdit, QLineEdit {
    background: #26272e; border: 1px solid #3a3b44; border-radius: 8px;
    padding: 6px; selection-background-color: #d4a253;
}
QPushButton {
    background: #3a3b44; border: none; border-radius: 8px; padding: 8px 14px;
}
QPushButton:hover { background: #4a4b56; }
QPushButton:pressed { background: #d4a253; color: #1e1f24; }
QPushButton:disabled { color: #6d6a62; }
QPushButton[default="true"], QPushButton:default {
    background: #d4a253; color: #1e1f24; font-weight: 600;
}
QProgressBar {
    background: #26272e; border-radius: 6px; text-align: center; height: 12px;
}
QProgressBar::chunk { background: #d4a253; border-radius: 6px; }
QSplitter::handle { background: #3a3b44; width: 2px; }
QSpinBox, QComboBox {
    background: #26272e; border: 1px solid #3a3b44; border-radius: 6px;
    padding: 4px 8px;
}
"""
