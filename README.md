# StudyBuddy

A local-first desktop RAG study assistant. Point it at your lecture PDFs and
notes, and ask questions with answers grounded in — and cited back to — your
own material.

![StudyBuddy](screenshots/app.png)

Evolution of a Jupyter-notebook experiment (fixed-size chunks, pure vector
search, hard-coded paths) into a proper desktop application with a modern
retrieval pipeline.

## Features

- **Desktop app** — PySide6/Qt; a real window next to your study PDFs, not a
  browser tab. Indexing and answer generation run on background threads.
- **Structured PDF parsing** — `pymupdf4llm` converts pages to Markdown
  (headings, tables, lists) so chunking can follow document structure.
- **Structure-aware chunking** — splits along the heading hierarchy first
  (each chunk keeps a `Lecture 3 > 3.2 Agile` breadcrumb), then size-splits
  within sections. Chunks get a contextual `Document/Section/Page` prefix
  baked into their embedding — the zero-LLM-call variant of contextual
  retrieval.
- **Hybrid retrieval** — vector search (sqlite-vec) **+** BM25 keyword search
  (SQLite FTS5) fused with Reciprocal Rank Fusion, then optional cross-encoder
  reranking (`bge-reranker-base`, ONNX).
- **Cited answers** — every answer lists the document + page/section sources.
- **Two providers** — Gemini API (free tier) by default, or fully-offline
  **Ollama** for zero-cloud usage. Switchable in Settings.
- **Incremental indexing** — files are content-hashed; unchanged files never
  re-embed. Everything lives in one SQLite file under the OS app-data dir.
- **Local embeddings** — `all-MiniLM-L6-v2` via fastembed/ONNX. No PyTorch —
  keeps the packaged app small. Models cache after first download; retrieval
  then works offline.

## Install & run

```bash
git clone https://github.com/mahmoudhaytham25/studybuddy-rag
cd studybuddy-rag
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m studybuddy        # or: studybuddy
```

Requires Python ≥ 3.10 whose `sqlite3` supports loadable extensions
(standard python.org / Homebrew builds do; some pyenv/standalone builds
don't — `pip install pysqlite3-binary` as a fallback).

## Getting answers

1. **File → Settings** — pick a provider:
   - **Gemini**: paste a free key from
     [Google AI Studio](https://aistudio.google.com) (default `gemini-2.5-flash`)
   - **Ollama**: `ollama serve` + `ollama pull llama3.1:8b`, then Test connection
2. **Add Files / Add Folder** — drop in PDFs, Markdown or TXT notes.
   First run downloads the embedding model (~100 MB, cached afterwards).
3. Ask questions — answers stream in with `Sources` cited per response.

## How it works

```
PDF/MD/TXT ── pymupdf4llm ──▶ structured markdown
        │
        ▼  heading-aware split + contextual prefix          ┌─ sqlite-vec ──┐
chunks ─▶ embed (MiniLM-L6-v2, ONNX) ─▶ SQLite ─────────────┤               ├─▶ RRF
        │                                                    └─ FTS5 BM25 ──┘   fuse
        ▼                                                                        │
query ─▶ hybrid candidates ─▶ cross-encoder rerank ─▶ top-k ─▶ prompt ─▶ LLM
                                                               (Gemini | Ollama)
```

Layout:

```
studybuddy/
  core/            provider-agnostic pipeline (no Qt imports)
    documents.py   parse → ParsedDocument(pages, markdown)
    chunking.py    heading-aware splitter + contextual prefixes
    embeddings.py  fastembed wrapper (embed + rerank, lazy-loaded)
    store.py       SQLite + sqlite-vec + FTS5, RRF hybrid search
    index.py       hash-diffed incremental indexing
    retrieval→rag  prompt build + streamed answers with citations
    providers/     ChatProvider protocol · Gemini · Ollama
  ui/              PySide6: MainWindow, DocumentPanel, ChatPanel,
                   SettingsDialog, QThread workers
tests/             pytest suite (core fully covered; UI smoke via offscreen Qt)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Packaging (WIP)

`pip install -e ".[package]"` then PyInstaller — a proper
`studybuddy.spec` and signed installers land in a later milestone.

## Stack

Python · PySide6 · PyMuPDF/pymupdf4llm · fastembed (ONNX) · sqlite-vec ·
SQLite FTS5 · google-genai · Ollama · PyInstaller · pytest · GitHub Actions
