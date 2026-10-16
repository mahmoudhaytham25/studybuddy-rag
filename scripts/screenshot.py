#!/usr/bin/env python3
"""Regenerate ``screenshots/app.png`` with a realistic, populated app state.

Run from anywhere inside the repo venv — no arguments needed:

    python scripts/screenshot.py [output.png]

Builds four sample study documents (one real generated PDF plus
Markdown/TXT notes) in a temp dir, indexes them into a temp SQLite store
through the real ``index.index_paths`` pipeline, then drives the real
``ChatPanel`` streaming path with a canned markdown answer whose Sources
are genuine retrieval hits from that index. No LLM provider, API key or
network access is required — the already-downloaded local embedding
models in the app data dir are reused as-is.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"  # must precede any PySide6 import

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # allow running without an editable install

OUT_PATH = (
    Path(sys.argv[1]).expanduser()
    if len(sys.argv) > 1
    else REPO_ROOT / "screenshots" / "app.png"
)

QUESTION = (
    "When should I choose a microservices architecture "
    "over a layered monolith?"
)

ANSWER_MD = """\
**Short answer:** reach for **microservices** only when independent
deployability and selective scaling genuinely outweigh the operational
cost [1].

- A **layered monolith** ships as one deployable unit — simple to build,
  test and debug, but every change redeploys the whole system and scaling
  is all-or-nothing [1].
- **Microservices** give each bounded context its own service and
  datastore, so hot components scale alone and teams ship independently —
  paid for with network latency, distributed data and observability
  overhead [2].

Rule of thumb from lecture: *start layered; split along scaling or
team-boundary lines when the pain is real, not anticipated* [2].
"""

# Streamed in a few chunks — same render path as a real provider.
ANSWER_PARTS = [
    ANSWER_MD[: len(ANSWER_MD) // 3],
    ANSWER_MD[len(ANSWER_MD) // 3 : 2 * len(ANSWER_MD) // 3],
    ANSWER_MD[2 * len(ANSWER_MD) // 3 :],
]

_PDF_PAGES = [
    [
        (
            "title",
            "SWE 316 — Software Architecture\n"
            "Lecture 5: Architectural Styles & Design Decisions",
        ),
        (
            "body",
            "Instructor: Dr. A. Khalil — Fall 2025, week 5 handout.\n\n"
            "An architectural style is a named, reusable organisation of "
            "components, connectors and constraints. Choosing one is a "
            "trade-off between deployability, scalability, consistency and "
            "the cognitive load on the team — there is no free lunch, only "
            "different bills that arrive at different times.",
        ),
        ("h2", "5.1  The layered (n-tier) style"),
        (
            "body",
            "The system is divided into horizontal layers — presentation, "
            "application, domain and persistence — where each layer only "
            "calls the layer directly beneath it. The whole application "
            "compiles and deploys as a single unit: a layered monolith.\n\n"
            "Strengths: simple mental model, easy local debugging, cheap "
            "to test end-to-end, and a single deployment pipeline. "
            "Weaknesses: a one-line change redeploys everything, layers "
            "tend to leak abstractions over time, and scaling is "
            "all-or-nothing — you replicate the entire monolith even if "
            "only the reporting module is hot.",
        ),
    ],
    [
        ("h2", "5.2  Microservices"),
        (
            "body",
            "The application is decomposed into small services aligned "
            "with bounded contexts — payments, search, enrolment — each "
            "with its own datastore and its own release cadence. Services "
            "communicate over the network via APIs or messaging.\n\n"
            "Strengths: independent deployability (a team ships payments "
            "without touching search), selective scaling of hot "
            "components, and polyglot persistence. Weaknesses: everything "
            "a monolith hides becomes explicit — network latency and "
            "partial failure, eventual consistency across datastores, "
            "distributed tracing, schema versioning and an operational "
            "surface that demands mature DevOps.",
        ),
        ("h2", "5.3  Event-driven (publish–subscribe)"),
        (
            "body",
            "Components never call each other directly; they emit events "
            "to a broker and react to events they subscribe to. This "
            "decouples producers from consumers in time and identity, "
            "which is excellent for integrations and asynchronous "
            "workflows, but control flow becomes implicit — you read the "
            "event topology, not a call graph, and debugging means "
            "reconstructing causality from message logs.",
        ),
    ],
    [
        ("h2", "5.4  Pipe-and-filter"),
        (
            "body",
            "Data flows through a sequence of independent transformations "
            "— filters — connected by pipes. Compilers and batch "
            "ETL jobs are the textbook cases. Each stage is easy to "
            "reason about and to test in isolation, but the style fits "
            "stream-shaped work and little else; interactive systems "
            "fight the model.",
        ),
        ("h2", "5.5  Choosing a style"),
        (
            "body",
            "Lecture guidance: default to a layered monolith for small "
            "teams and early products — it is the cheapest architecture "
            "that works. Adopt microservices when at least one of these "
            "is true: a component's scaling profile diverges sharply "
            "from the rest, teams need independent release cadences, or "
            "availability requirements isolate failure domains. "
            "Event-driven styles compose well with either and are added "
            "for integration and asynchrony, not adopted wholesale. "
            "Pipe-and-filter is chosen when the problem is already a "
            "data pipeline.",
        ),
    ],
]

_MD_OS = """\
# Operating Systems — Study Notes (CS 340)

## Week 4: CPU Scheduling

### First-Come, First-Served (FCFS)
Jobs run in arrival order to completion. Trivially fair and starvation-free,
but the convoy effect wrecks average waiting time: one 100 ms job ahead of
ten 1 ms jobs makes everyone wait.

### Shortest Job First (SJF)
Run the job with the smallest next CPU burst. Provably optimal for average
waiting time, but requires predicting burst lengths — in practice done by
exponential averaging of previous bursts — and long jobs can starve.

### Round Robin (RR)
FCFS plus preemption: each job gets a time quantum q, then goes to the back
of the ready queue. Quantum too large → degenerates to FCFS; too small →
context-switch overhead dominates. Rule of thumb: q should be larger than
~80% of bursts. Great response time, poor turnaround for long jobs.

### Priority Scheduling
Each process carries a priority; the highest runs first. SJF is priority
scheduling where priority = predicted burst. Starvation is the hazard —
fix it with aging: gradually raise the priority of processes that wait.

## Week 5: Deadlocks

Four necessary conditions — mutual exclusion, hold and wait, no preemption,
circular wait. Break any one to prevent deadlock; alternatively avoid it
with the banker's algorithm (keep the system in a safe state) or detect
cycles in the resource-allocation graph and recover by killing processes.

## Week 6: Memory Management

Paging removes external fragmentation by fixing frame sizes; segmentation
matches the programmer's view but reintroduces fragmentation. TLB misses
and page faults dominate paged-system performance — the effective access
time formula EAT = (1 − p)·ma + p·(fault cost) shows why the page-fault
rate p must stay tiny.
"""

_MD_DB = """\
# Databases — Normalization Revision (CS 341)

## Functional Dependencies

X → Y means the value of X determines the value of Y. A candidate key is a
minimal set of attributes that functionally determines all attributes of
the relation. Closure sets X+ tell you what a key actually determines.

## The Normal Forms

### 1NF — atomic values
Every attribute holds a single value: no lists, no repeating groups, no
composite cells. A "students" column containing "Ali, Sara, Noor" violates
1NF outright.

### 2NF — no partial dependency
Applies to tables with composite keys: every non-key attribute must depend
on the whole key, not part of it. If (student_id, course_id) is the key and
student_name depends only on student_id, split it out.

### 3NF — no transitive dependency
Non-key attributes must not depend on other non-key attributes. If
course_id → instructor and instructor → instructor_office, then
instructor_office is transitively dependent on the key — move it to an
instructors table.

### BCNF — the stricter cousin
For every non-trivial dependency X → Y, X must be a superkey. Catches the
rare anomalies 3NF tolerates when a table has overlapping candidate keys.

## Why Bother — Update Anomalies

Denormalized tables suffer insertion anomalies (can't record a fact without
unrelated data), deletion anomalies (deleting one fact loses another) and
update anomalies (one logical change needs many row edits — miss one and
the data is inconsistent).
"""

_TXT_NET = """\
Computer Networks — final exam review sheet
============================================

OSI model, bottom to top: Physical, Data Link, Network, Transport,
Session, Presentation, Application. Remember which layer owns what:
routing is layer 3 (IP), reliable delivery is layer 4 (TCP), framing and
MAC are layer 2.

TCP vs UDP
----------
TCP: connection-oriented, reliable byte stream — three-way handshake
(SYN, SYN-ACK, ACK), sequence numbers, retransmission, flow control via
sliding window, congestion control (slow start, AIMD). Use it for web,
mail, files — anything that can't lose bytes.

UDP: connectionless datagrams, no guarantee of delivery, order or
duplication — but tiny header (8 bytes) and no handshake latency. Use it
for DNS lookups, VoIP, game state, anything where late = useless.

Subnetting quick reference: /24 = 256 addresses (254 usable), /25 = 128,
/26 = 64, /27 = 32, /28 = 16, /30 = 4 (point-to-point links).
"""


def _write_pdf(path: Path, pages: list[list[tuple[str, str]]]) -> None:
    """Render one multi-page PDF with real headings (larger/bold fonts so
    pymupdf4llm emits markdown headings) — exercises the real PDF loader."""
    try:
        import pymupdf
    except ImportError:  # older PyMuPDF exposes only the `fitz` alias
        import fitz as pymupdf

    styles = {"title": (19.0, "hebo"), "h2": (14.0, "hebo"), "body": (11.0, "helv")}
    margin, page_w, page_h = 56.0, 595.0, 842.0  # A4
    doc = pymupdf.open()
    for blocks in pages:
        page = doc.new_page(width=page_w, height=page_h)
        cursor = margin
        for style, text in blocks:
            size, font = styles[style]
            rect = pymupdf.Rect(margin, cursor, page_w - margin, page_h - margin)
            spare = page.insert_textbox(
                rect, text, fontsize=size, fontname=font
            )
            if spare < 0:
                raise RuntimeError(f"PDF text overflow in {path.name}")
            cursor += rect.height - spare + size * 0.7
    doc.save(str(path))
    doc.close()


def _write_samples(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    pdf = root / "SWE316-Architecture-L05.pdf"
    _write_pdf(pdf, _PDF_PAGES)
    os_notes = root / "operating-systems-notes.md"
    os_notes.write_text(_MD_OS, encoding="utf-8")
    db_notes = root / "db-normalization-notes.md"
    db_notes.write_text(_MD_DB, encoding="utf-8")
    net_notes = root / "networking-exam-review.txt"
    net_notes.write_text(_TXT_NET, encoding="utf-8")
    return [pdf, db_notes, net_notes, os_notes]


def _pump(app, ms: int = 300) -> None:
    """Let Qt's loop run for `ms` so layout, painting and scrolling settle."""
    deadline = time.monotonic() + ms / 1000.0
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from studybuddy.core import embeddings, rag
    from studybuddy.core.config import Settings
    from studybuddy.core.index import index_paths
    from studybuddy.core.store import Store
    from studybuddy.ui.main_window import _STYLESHEET, MainWindow

    # Keep the real model cache (~already-downloaded MiniLM + reranker in
    # the user data dir) even if data_dir() is ever redirected — pinning
    # the resolved path means fastembed finds the models, nothing
    # re-downloads, while the SQLite store still lives in the temp dir.
    real_model_cache = embeddings._cache_dir()
    embeddings._cache_dir = lambda: real_model_cache

    app = QApplication([sys.argv[0]])
    app.setApplicationName("StudyBuddy")
    app.setOrganizationName("MahmoudHaytham")
    app.setStyleSheet(_STYLESHEET)

    with tempfile.TemporaryDirectory(prefix="studybuddy-shot-") as tmp:
        tmp = Path(tmp)
        docs = _write_samples(tmp / "library")
        store = Store(tmp / "studybuddy.db")
        settings = Settings(
            document_paths=[str(p) for p in docs],
            top_k=3,  # keep the Sources list to ~3 entries in the shot
            # Reranking only reorders hits; skipping it means the script
            # needs just the small embedder cache, not the reranker model.
            rerank=False,
        )
        settings.save = lambda *a, **k: None  # never touch the real config

        # Index synchronously up front (deterministic), then let the
        # window's own IndexWorker run — every file hashes unchanged, so
        # it settles straight into the final populated status.
        result = index_paths(
            store,
            docs,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if result.failed:
            raise SystemExit(f"sample indexing failed: {result.failed}")

        window = MainWindow(settings, store)
        window.show()
        for worker in list(window._workers):
            worker.wait(120_000)
        _pump(app)

        # Genuine retrieval from the temp index — the same call the
        # ChatWorker makes, minus the provider.
        hits = rag.retrieve(store, settings, QUESTION)
        if not hits:
            raise SystemExit("retrieval returned no sources")

        chat = window.chat
        chat.add_user_message(QUESTION)
        chat.start_answer()
        for part in ANSWER_PARTS:
            chat.append_token(part)
            app.processEvents()
        chat.finish_answer(hits)
        _pump(app, 500)

        image = window.grab()
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not image.save(str(OUT_PATH)):
            raise SystemExit(f"could not save screenshot to {OUT_PATH}")
        size = OUT_PATH.stat().st_size
        if size < 20_000:
            raise SystemExit(
                f"screenshot at {OUT_PATH} looks blank ({size} bytes)"
            )
        print(
            f"saved {OUT_PATH} "
            f"({image.width()}x{image.height()}, {size} bytes)"
        )
        window.close()
        _pump(app, 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
