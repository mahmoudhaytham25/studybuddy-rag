"""Persistent vector + full-text store.

A single SQLite file holds everything: the document registry, chunk records,
vector embeddings (sqlite-vec) and a BM25 full-text index (FTS5). Hybrid
retrieval fuses vector and BM25 rankings with Reciprocal Rank Fusion — the
modern default that fixes pure-vector search's weakness with rare terms,
identifiers and exact phrases.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

# Some Python builds (python-build-standalone, some pyenv builds) compile
# sqlite3 without loadable-extension support — pysqlite3 wheels bundle
# their own SQLite with it enabled.
import sqlite3 as _stdlib_sqlite3

if hasattr(_stdlib_sqlite3.Connection, "enable_load_extension"):
    sqlite3 = _stdlib_sqlite3
else:
    try:
        import pysqlite3 as sqlite3  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError(
            "This Python's sqlite3 lacks loadable-extension support "
            "(required by sqlite-vec). Use a standard python.org/Homebrew "
            "interpreter, or `pip install pysqlite3-binary`."
        ) from exc

from .chunking import Chunk
from .config import EMBEDDING_DIM

_RRF_K = 60


@dataclass
class DocumentRecord:
    path: str
    title: str
    sha256: str
    n_chunks: int
    indexed_at: float


@dataclass
class ChunkHit:
    chunk_id: int
    doc_path: str
    doc_title: str
    page: int | None
    section: str
    content: str
    score: float

    def citation(self) -> str:
        loc = f"p. {self.page}" if self.page else self.section or ""
        return f"{self.doc_title}{' — ' + loc if loc else ''}"


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents(
                    path TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    n_chunks INTEGER NOT NULL,
                    indexed_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks(
                    id INTEGER PRIMARY KEY,
                    doc_path TEXT NOT NULL REFERENCES documents(path)
                        ON DELETE CASCADE,
                    ord INTEGER NOT NULL,
                    page INTEGER,
                    section TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_path);
                """
            )
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
                f"embedding float[{EMBEDDING_DIM}])"
            )
            try:
                self.conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
                    "content, doc_title, section)"
                )
                self.fts_enabled = True
            except sqlite3.OperationalError:
                self.fts_enabled = False
        self.conn.execute("PRAGMA foreign_keys = ON")

    # -- document registry ----------------------------------------------------

    def get_document(self, path: str) -> DocumentRecord | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE path = ?", (path,)
        ).fetchone()
        return DocumentRecord(**dict(row)) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY title"
        ).fetchall()
        return [DocumentRecord(**dict(r)) for r in rows]

    def count_chunks(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def delete_document(self, path: str) -> None:
        with self._lock, self.conn:
            ids = [
                r[0]
                for r in self.conn.execute(
                    "SELECT id FROM chunks WHERE doc_path = ?", (path,)
                )
            ]
            self._delete_chunk_rows(ids)
            self.conn.execute("DELETE FROM documents WHERE path = ?", (path,))

    def _delete_chunk_rows(self, ids: list[int]) -> None:
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        self.conn.execute(f"DELETE FROM chunks WHERE id IN ({marks})", ids)
        self.conn.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({marks})", ids)
        if self.fts_enabled:
            self.conn.execute(
                f"DELETE FROM chunks_fts WHERE rowid IN ({marks})", ids
            )

    # -- indexing -------------------------------------------------------------

    def replace_document(
        self, path: str, title: str, sha256: str,
        chunks: list[Chunk], embeddings: list[list[float]],
    ) -> None:
        """Atomically replace a document's chunks (reindex) or insert it."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks/embeddings length mismatch")
        with self._lock, self.conn:
            old_ids = [
                r[0]
                for r in self.conn.execute(
                    "SELECT id FROM chunks WHERE doc_path = ?", (path,)
                )
            ]
            self._delete_chunk_rows(old_ids)
            self.conn.execute(
                "INSERT OR REPLACE INTO documents"
                "(path, title, sha256, n_chunks, indexed_at)"
                " VALUES (?,?,?,?,?)",
                (path, title, sha256, len(chunks), time.time()),
            )
            for chunk, emb in zip(chunks, embeddings):
                cur = self.conn.execute(
                    "INSERT INTO chunks(doc_path, ord, page, section, content)"
                    " VALUES (?,?,?,?,?)",
                    (path, chunk.ord, chunk.page, chunk.section, chunk.content),
                )
                rowid = cur.lastrowid
                self.conn.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?,?)",
                    (rowid, sqlite_vec.serialize_float32(emb)),
                )
                if self.fts_enabled:
                    self.conn.execute(
                        "INSERT INTO chunks_fts(rowid, content, doc_title,"
                        " section) VALUES (?,?,?,?)",
                        (rowid, chunk.content, title, chunk.section),
                    )

    # -- retrieval ------------------------------------------------------------

    def _hits(self, ids_scores: list[tuple[int, float]]) -> list[ChunkHit]:
        if not ids_scores:
            return []
        marks = ",".join("?" * len(ids_scores))
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({marks})",
            [i for i, _ in ids_scores],
        ).fetchall()
        by_id = {r["id"]: r for r in rows}
        return [
            ChunkHit(
                chunk_id=i,
                doc_path=by_id[i]["doc_path"],
                doc_title=self._title_for(by_id[i]["doc_path"]),
                page=by_id[i]["page"],
                section=by_id[i]["section"],
                content=by_id[i]["content"],
                score=s,
            )
            for i, s in ids_scores
            if i in by_id
        ]

    def _title_for(self, doc_path: str) -> str:
        row = self.conn.execute(
            "SELECT title FROM documents WHERE path = ?", (doc_path,)
        ).fetchone()
        return row[0] if row else Path(doc_path).stem

    def vector_search(
        self, embedding: list[float], k: int
    ) -> list[tuple[int, float]]:
        rows = self.conn.execute(
            "SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ?"
            " ORDER BY distance LIMIT ?",
            (sqlite_vec.serialize_float32(embedding), k),
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def fts_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.fts_enabled:
            return []
        match = _fts_query(query)
        if not match:
            return []
        try:
            rows = self.conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS rank FROM chunks_fts"
                " WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r[0], float(r[1])) for r in rows]

    def hybrid_search(
        self, query: str, embedding: list[float], k: int
    ) -> list[ChunkHit]:
        """Fuse vector and BM25 rankings with Reciprocal Rank Fusion."""
        vec = self.vector_search(embedding, k * 4)
        fts = self.fts_search(query, k * 4)
        rrf: dict[int, float] = {}
        for rank, (cid, _) in enumerate(vec):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        for rank, (cid, _) in enumerate(fts):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        ranked = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return self._hits(ranked)

    def close(self) -> None:
        self.conn.close()


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 AND/OR query: quote each token, OR them."""
    tokens = [t for t in _tokenize(query) if t]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens[:12])


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[\w']+", text.lower())
