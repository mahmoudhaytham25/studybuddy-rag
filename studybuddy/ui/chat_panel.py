"""Chat panel — question input, streaming answer display, source citations."""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices, QTextDocument
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ..core.store import ChunkHit

# Prior turns fed back to the provider — bounded so long sessions can't
# blow up the prompt.
MAX_HISTORY_TURNS = 8


def source_url(doc_path: str, page: int | None) -> str:
    """Clickable target for a citation: a file:// URL, with a `#page=N`
    hint for PDFs (viewers that honour page anchors jump straight there;
    the rest just open the file)."""
    url = QUrl.fromLocalFile(doc_path)
    if page and doc_path.lower().endswith(".pdf"):
        url.setFragment(f"page={page}")
    return url.toString()


def _open_url(url: QUrl) -> None:
    """Seam for tests — the one place we shell out to the OS viewer."""
    QDesktopServices.openUrl(url)


def _md_to_html(markdown: str) -> str:
    """Render one answer's markdown to an HTML body fragment using Qt's
    native parser — headers, bold, lists, code blocks — no extra
    dependency. `[n]` citation markers are not links, so they survive
    verbatim."""
    doc = QTextDocument()
    doc.setMarkdown(markdown)
    full = doc.toHtml()
    start = full.find("<body")
    end = full.rfind("</body>")
    if start < 0 or end < 0:
        return html.escape(markdown).replace("\n", "<br>")
    start = full.find(">", start) + 1
    return full[start:end].strip()


class ChatPanel(QWidget):
    ask = Signal(str)
    notice = Signal(str)  # transient status-bar message (e.g. missing file)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()
        title = QLabel("💬 Ask your notes")
        title.setObjectName("panelTitle")
        header.addWidget(title, 1)
        self.new_chat = QPushButton("New chat")
        header.addWidget(self.new_chat)
        layout.addLayout(header)

        self.history = QTextBrowser()
        # Links are opened via _link_clicked so file citations get an
        # existence check and a page anchor instead of blind navigation.
        self.history.setOpenLinks(False)
        self.history.setOpenExternalLinks(False)
        self.history.anchorClicked.connect(self._link_clicked)
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
        self.new_chat.clicked.connect(self.clear_chat)
        self.input.installEventFilter(self)

        self._pending_md = ""  # raw markdown of the in-flight answer
        self._current_question = ""  # question awaiting its answer
        self._chat_history: list[dict[str, str]] = []  # provider turns
        self._transcript: list[str] = []  # rendered HTML blocks

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
        self._current_question = text
        self._append_block("user", f"<b>You</b><br>{html.escape(text)}")

    def start_answer(self) -> None:
        self._pending_md = ""
        self._append_block(
            "assistant", "<b>StudyBuddy</b><br><i>Thinking…</i>"
        )

    def append_token(self, tok: str) -> None:
        # Accumulate RAW markdown; re-render the block on each token.
        self._pending_md += tok
        self._replace_last_block(
            f"<b>StudyBuddy</b><br>{_md_to_html(self._pending_md)}"
        )

    def finish_answer(self, sources: list[ChunkHit]) -> None:
        cite = ""
        if sources:
            items = "".join(
                f'<li><a href="'
                f'{html.escape(source_url(s.doc_path, s.page), quote=True)}'
                f'">{html.escape(s.citation())}</a></li>'
                for s in sources
            )
            cite = (
                f"<br><span class='src'><b>Sources</b>"
                f"<ul>{items}</ul></span>"
            )
        self._replace_last_block(
            f"<b>StudyBuddy</b><br>{_md_to_html(self._pending_md)}{cite}"
        )
        self._record_turn()
        self._pending_md = ""

    def add_error(self, msg: str) -> None:
        self._replace_last_block(
            f"<b>StudyBuddy</b><br><span class='err'>{html.escape(msg)}</span>"
        )
        self._pending_md = ""
        self._current_question = ""  # failed turns don't enter history

    def set_busy(self, busy: bool) -> None:
        self.send.setEnabled(not busy)
        self.input.setEnabled(not busy)
        # Mid-answer clears would strand the in-flight worker's tokens.
        self.new_chat.setEnabled(not busy)

    # -- conversation history ---------------------------------------------------

    def chat_history(self) -> list[dict[str, str]]:
        """Snapshot of prior turns as {role, content} for the provider —
        plain question/answer text, never the retrieved-context prompt."""
        return list(self._chat_history)

    def clear_chat(self) -> None:
        """New conversation: wipe the transcript and provider history."""
        self._transcript.clear()
        self._chat_history.clear()
        self._pending_md = ""
        self._current_question = ""
        self.history.clear()

    def _record_turn(self) -> None:
        if not self._current_question:
            return
        self._chat_history.append(
            {"role": "user", "content": self._current_question}
        )
        self._chat_history.append(
            {"role": "assistant", "content": self._pending_md}
        )
        del self._chat_history[: -2 * MAX_HISTORY_TURNS]
        self._current_question = ""

    # -- internals --------------------------------------------------------------

    def _link_clicked(self, url: QUrl) -> None:
        if url.isLocalFile() and not Path(url.toLocalFile()).exists():
            self.notice.emit(f"File not found: {url.toLocalFile()}")
            return
        _open_url(url)

    def _append_block(self, cls: str, inner: str) -> None:
        self._transcript.append(inner)
        self._rerender()

    def _replace_last_block(self, inner: str) -> None:
        # QTextBrowser can't edit a block in place; keep the transcript and
        # re-render. Cheap enough at chat sizes.
        if not self._transcript:
            return
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
