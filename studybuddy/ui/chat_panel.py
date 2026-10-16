"""Chat panel — question input, streaming answer display, source citations."""

from __future__ import annotations

import html

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ChatPanel(QWidget):
    ask = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("💬 Ask your notes")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.history = QTextBrowser()
        self.history.setOpenExternalLinks(True)
        layout.addWidget(self.history, 1)

        row = QHBoxLayout()
        self.input = QTextEdit()
        self.input.setPlaceholderText(
            "Ask about anything in your library…  (Enter to send)"
        )
        self.input.setFixedHeight(56)
        self.send = QPushButton("Ask")
        self.send.setDefault(True)
        row.addWidget(self.input, 1)
        row.addWidget(self.send)
        layout.addLayout(row)

        self.send.clicked.connect(self._emit)
        self.input.installEventFilter(self)

        self._pending_html = ""
        self._transcript: list[str] = []

    # -- input --------------------------------------------------------------

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent, Qt

        if (
            obj is self.input
            and event.type() == QEvent.KeyPress
            and event.key() in (Qt.Key_Return, Qt.Key_Enter)
            and not event.modifiers() & Qt.ShiftModifier
        ):
            self._emit()
            return True
        return super().eventFilter(obj, event)

    def _emit(self) -> None:
        text = self.input.toPlainText().strip()
        if text:
            self.input.clear()
            self.ask.emit(text)

    # -- display --------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        self._append_block("user", f"<b>You</b><br>{html.escape(text)}")

    def start_answer(self) -> None:
        self._pending_html = ""
        self._append_block("assistant", "<b>StudyBuddy</b><br><i>Thinking…</i>")

    def append_token(self, tok: str) -> None:
        self._pending_html += html.escape(tok)
        self._replace_last_block(
            f"<b>StudyBuddy</b><br>{self._render_md_lite(self._pending_html)}"
        )

    def finish_answer(self, citations: list[str]) -> None:
        cite = ""
        if citations:
            items = "".join(f"<li>{html.escape(c)}</li>" for c in citations)
            cite = f"<br><span class='src'><b>Sources</b><ul>{items}</ul></span>"
        self._replace_last_block(
            f"<b>StudyBuddy</b><br>{self._render_md_lite(self._pending_html)}{cite}"
        )
        self._pending_html = ""

    def add_error(self, msg: str) -> None:
        self._replace_last_block(
            f"<b>StudyBuddy</b><br><span class='err'>{html.escape(msg)}</span>"
        )
        self._pending_html = ""

    def set_busy(self, busy: bool) -> None:
        self.send.setEnabled(not busy)
        self.input.setEnabled(not busy)

    # -- internals --------------------------------------------------------------

    def _append_block(self, cls: str, inner: str) -> None:
        self._transcript.append(inner)
        self._rerender()

    def _replace_last_block(self, inner: str) -> None:
        # QTextBrowser can't edit a block in place; keep the transcript and
        # re-render. Cheap enough at chat sizes.
        self._transcript[-1] = inner
        self._rerender()

    def _rerender(self) -> None:
        self.history.setHtml(
            "".join(f"<div class='msg'>{h}</div>" for h in self._transcript)
        )
        self._scroll_end()

    def _scroll_end(self) -> None:
        bar = self.history.verticalScrollBar()
        bar.setValue(bar.maximum())

    @staticmethod
    def _render_md_lite(text: str) -> str:
        """Bold/inline-code on already-HTML-escaped text — enough for model
        output, no markdown renderer needed."""
        import re

        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        return text.replace("\n", "<br>")
