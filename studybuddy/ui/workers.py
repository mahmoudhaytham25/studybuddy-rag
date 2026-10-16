"""QThread workers — indexing and answer generation stay off the UI thread."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..core.config import Settings
from ..core.index import index_paths
from ..core.rag import answer_stream
from ..core.store import Store


class IndexWorker(QThread):
    progressed = Signal(str, int, int)  # filename, done, total
    finished_ok = Signal(object)  # IndexResult
    failed = Signal(str)

    def __init__(self, store: Store, settings: Settings, paths: list[Path]):
        super().__init__()
        self._store, self._settings, self._paths = store, settings, paths

    def run(self) -> None:
        try:
            result = index_paths(
                self._store,
                self._paths,
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
                pdf_parser=self._settings.pdf_parser,
                progress=lambda name, done, total: self.progressed.emit(
                    name, done, total
                ),
            )
        except Exception as exc:  # model download / db errors
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(result)


class ChatWorker(QThread):
    token = Signal(str)
    done = Signal(list)  # list[ChunkHit]
    failed = Signal(str)

    def __init__(
        self,
        store: Store,
        provider,
        settings: Settings,
        question: str,
        history: list | None = None,
    ):
        super().__init__()
        self._store, self._provider, self._settings = store, provider, settings
        self._question = question
        self._history = history  # snapshot of prior turns, list[ChatMessage]

    def run(self) -> None:
        try:
            stream, hits = answer_stream(
                self._store,
                self._provider,
                self._settings,
                self._question,
                history=self._history,
            )
            for tok in stream:
                self.token.emit(tok)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.done.emit(hits)
