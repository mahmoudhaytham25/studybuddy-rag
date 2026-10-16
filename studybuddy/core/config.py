"""Application settings and data-directory layout.

Settings persist as JSON in the platform-appropriate config dir;
the vector store and cached models live in the user data dir.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP_NAME = "StudyBuddy"
APP_AUTHOR = "MahmoudHaytham"

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
RERANKER_MODEL = "BAAI/bge-reranker-base"

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


def data_dir() -> Path:
    return Path(user_data_dir(APP_NAME, APP_AUTHOR))


def db_path() -> Path:
    return data_dir() / "studybuddy.db"


@dataclass
class Settings:
    provider: str = "gemini"  # "gemini" | "ollama"
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    # Indexing / retrieval tuning
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 6
    rerank: bool = True
    # Documents added to the library (absolute paths)
    document_paths: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        cfg = path or _config_file()
        if not cfg.exists():
            return cls()
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path | None = None) -> None:
        cfg = path or _config_file()
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _config_file() -> Path:
    return Path(user_config_dir(APP_NAME, APP_AUTHOR)) / "settings.json"
