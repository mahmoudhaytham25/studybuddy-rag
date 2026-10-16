"""Library panel — add/remove documents, see index status, progress."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.config import SUPPORTED_EXTENSIONS
from ..core.documents import collect_files
from ..core.store import Store


class DocumentPanel(QWidget):
    add_requested = Signal(list)  # list[Path] to index
    remove_requested = Signal(str)  # doc path to drop
    refresh_requested = Signal()

    def __init__(self, store: Store, parent=None):
        super().__init__(parent)
        self._store = store
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("📚 Library")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self.list = QListWidget()
        layout.addWidget(self.list, 1)

        btns = QHBoxLayout()
        self.add_files = QPushButton("Add Files")
        self.add_folder = QPushButton("Add Folder")
        self.remove = QPushButton("Remove")
        btns.addWidget(self.add_files)
        btns.addWidget(self.add_folder)
        btns.addWidget(self.remove)
        layout.addLayout(btns)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setObjectName("statusLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)

        self.add_files.clicked.connect(self._pick_files)
        self.add_folder.clicked.connect(self._pick_folder)
        self.remove.clicked.connect(self._remove_selected)

        self.reload()

    # -- public ---------------------------------------------------------------

    def reload(self) -> None:
        self.list.clear()
        docs = self._store.list_documents()
        for rec in docs:
            QListWidgetItem(f"{rec.title}  ({rec.n_chunks} chunks)", self.list)
        n = self._store.count_chunks()
        self.status.setText(
            f"{len(docs)} documents · {n} chunks indexed"
            if docs
            else "No documents yet — add PDFs, Markdown or text notes."
        )

    def set_busy(self, name: str, done: int, total: int) -> None:
        self.progress.setVisible(True)
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.status.setText(f"Indexing {name}… ({done}/{total})")

    def set_idle(self, summary: str) -> None:
        self.progress.setVisible(False)
        self.status.setText(summary)
        self.reload()

    # -- internals ------------------------------------------------------------

    def _pick_files(self) -> None:
        globs = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add study documents", "", f"Documents ({globs})"
        )
        if files:
            self.add_requested.emit([Path(f) for f in files])

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add a folder")
        if folder:
            files = collect_files([Path(folder)])
            if files:
                self.add_requested.emit(files)
            else:
                self.status.setText("No supported documents in that folder.")

    def _remove_selected(self) -> None:
        item = self.list.currentItem()
        if not item:
            return
        row = self.list.currentRow()
        docs = self._store.list_documents()
        if 0 <= row < len(docs):
            self.remove_requested.emit(docs[row].path)
