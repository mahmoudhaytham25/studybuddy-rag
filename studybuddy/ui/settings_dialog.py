"""Settings dialog: provider, credentials, retrieval tuning."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from ..core.config import Settings
from ..core.providers import make_provider
from ..core.providers.base import ProviderError


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(440)
        self._settings = settings

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.provider = QComboBox()
        self.provider.addItems(["Gemini API", "Ollama (local)"])
        self.provider.setCurrentIndex(1 if settings.provider == "ollama" else 0)
        form.addRow("Provider", self.provider)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._gemini_page(settings))
        self.stack.addWidget(self._ollama_page(settings))
        self.provider.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.stack.setCurrentIndex(self.provider.currentIndex())

        layout.addLayout(form)
        layout.addWidget(self.stack)

        tune = QFormLayout()
        self.top_k = QSpinBox(minimum=1, maximum=20, value=settings.top_k)
        self.chunk_size = QSpinBox(
            minimum=200, maximum=4000, singleStep=100, value=settings.chunk_size
        )
        self.chunk_overlap = QSpinBox(
            minimum=0, maximum=1000, singleStep=50, value=settings.chunk_overlap
        )
        self.rerank = QCheckBox("Rerank results (better answers, slightly slower)")
        self.rerank.setChecked(settings.rerank)
        tune.addRow("Sources per answer", self.top_k)
        tune.addRow("Chunk size", self.chunk_size)
        tune.addRow("Chunk overlap", self.chunk_overlap)
        tune.addRow("", self.rerank)
        layout.addLayout(tune)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _gemini_page(self, s: Settings) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.gemini_key = QLineEdit(s.gemini_api_key)
        self.gemini_key.setEchoMode(QLineEdit.Password)
        self.gemini_key.setPlaceholderText("AIza…")
        self.gemini_model = QComboBox()
        self.gemini_model.setEditable(True)
        from ..core.providers.gemini import AUTO_MODEL, fallback_models

        self.gemini_model.addItem("auto — newest first, falls back on quota")
        for m in fallback_models():
            self.gemini_model.addItem(m)
        # prefill: either "auto" or a saved custom model name
        idx = 0 if s.gemini_model == AUTO_MODEL else -1
        if idx == -1:
            self.gemini_model.setCurrentText(s.gemini_model)
        else:
            self.gemini_model.setCurrentIndex(0)
        hint = QLabel(
            "Free key from Google AI Studio → aistudio.google.com · "
            "each model has a separate quota, exhausted ones fall back"
        )
        hint.setWordWrap(True)
        form.addRow("API key", self.gemini_key)
        form.addRow("Model", self.gemini_model)
        form.addRow(hint)
        return page

    def _ollama_page(self, s: Settings) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.ollama_url = QLineEdit(s.ollama_base_url)
        self.ollama_model = QLineEdit(s.ollama_model)
        self.ollama_status = QLabel("")
        test_btn = QPushButton("Test connection")
        test_btn.clicked.connect(self._test_ollama)
        row = QHBoxLayout()
        row.addWidget(self.ollama_url)
        row.addWidget(test_btn)
        form.addRow("Server URL", row)
        form.addRow("Model", self.ollama_model)
        form.addRow(self.ollama_status)
        return page

    def _test_ollama(self) -> None:
        s = self._collect()
        provider = make_provider(s)
        ok, reason = provider.is_available()
        self.ollama_status.setText("✓ Connected" if ok else f"✗ {reason}")

    def _collect(self) -> Settings:
        s = self._settings
        s.provider = "ollama" if self.provider.currentIndex() == 1 else "gemini"
        s.gemini_api_key = self.gemini_key.text().strip()
        model_text = self.gemini_model.currentText().strip()
        s.gemini_model = (
            model_text if model_text.startswith("gemini-") else "auto"
        )
        s.ollama_base_url = self.ollama_url.text().strip()
        s.ollama_model = self.ollama_model.text().strip()
        s.top_k = self.top_k.value()
        s.chunk_size = self.chunk_size.value()
        s.chunk_overlap = self.chunk_overlap.value()
        s.rerank = self.rerank.isChecked()
        return s

    def _save(self) -> None:
        s = self._collect()
        if s.chunk_overlap >= s.chunk_size:
            QMessageBox.warning(
                self, "Invalid", "Chunk overlap must be smaller than chunk size."
            )
            return
        if s.provider == "gemini" and not s.gemini_api_key:
            QMessageBox.warning(
                self, "Missing key", "Enter a Gemini API key or switch to Ollama."
            )
            return
        s.save()
        self.accept()
