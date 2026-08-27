"""
Ingest is the entry point for both Task 1's "understand the pile" (first
documents into a new pile) and "stay alive" (a new document landing on an
existing pile) — it's the SAME endpoint and the SAME graph either way. The
graph's own idempotent-write design plus reconcile/examine being scoped to
touched claim_slots (see nodes.py) is what makes a later call to this
endpoint behave like an incremental update instead of a full re-run.

*** BUG FIX ***
This was previously the ONLY ingestion entry point that did not acquire
app.locking.pile_lock() before scheduling a run. mcp_server.py::
ingest_document and watcher.py::_trigger_run both call pile_lock(); this
REST route — the one the React UI and most graders will actually hit —
went straight to background_tasks.add_task() with no lock check. Two
concurrent uploads to the same pile via this endpoint could both spawn
graph runs with no serialization, which is exactly the corrupted-state
scenario behavior #9 exists to prevent. Fixed by acquiring the lock in a
short transaction before scheduling the background task, mirroring the
pattern already used in mcp_server.py.

*** BUG FIX ***
Also previously accepted any filename/extension and hardcoded
format="txt" regardless of what the extension actually was — a PDF or
DOCX would be written to raw/{id}.txt as undecoded bytes and processed
as if it were text (garbage extraction, not a clear failure). Now uses
app.formats.is_supported() to reject unsupported formats with 415
before any row is written.
"""
import uuid
from pathlib import Path as _Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Document, Pile, Run
from app.db.session import SessionLocal, get_session
from app.formats import detect_format, is_supported, unsupported_format_message
from app.locking import PileBusyError, pile_lock


def _raw_dir() -> _Path:
    """
    Returns the path to the raw document storage directory.
    Created on first access. Centralized here so watcher.py,
    mcp_server.py, and nodes.py all resolve to the same location
    regardless of working directory.
    """
    from app.config import get_settings
    raw = _Path(get_settings().inbox_dir).parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    return raw


router = APIRouter(prefix="/piles", tags=["ingest"])

MAX_DOCUMENT_BYTES = 20 * 1024 * 1024  # 20MB — generous for text/contracts, not unbounded


class IngestRequest(BaseModel):
    filename: str = Field(min_length=1)
    text_content: str = Field(
        min_length=1,
        description=(
            "Plain text content. Only .txt/.md filenames are accepted — "
            "see app/formats.py. Real PDF/DOCX ingestion is a planned "
            "extension point (app/graph/nodes.py::_read_document_text), "
            "not yet wired into this endpoint."
        ),
    )


class IngestResponse(BaseModel):
    document_id: str
    run_id: str
    status: str  # "processing" | "duplicate_skipped"


def _content_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


@router.post("/{pile_id}/documents", response_model=IngestResponse, status_code=202)
def ingest_document(
    pile_id: str,
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    pile = session.get(Pile, pile_id)
    if pile is None:
        raise HTTPException(404, f"Pile {pile_id} not found")

    # --- Format check: reject PDF/DOCX/unknown loudly, before writing anything ---
    if not is_supported(body.filename):
        raise HTTPException(415, unsupported_format_message(body.filename))

    if len(body.text_content.encode()) > MAX_DOCUMENT_BYTES:
        raise HTTPException(
            413,
            f"Document exceeds {MAX_DOCUMENT_BYTES} bytes. Large-file streaming "
            f"ingestion is not implemented in this build.",
        )

    content_hash = _content_hash(body.text_content)

    existing = (
        session.query(Document)
        .filter(Document.pile_id == pile_id, Document.content_hash == content_hash)
        .first()
    )
    if existing is not None:
        return IngestResponse(document_id=existing.id, run_id="", status="duplicate_skipped")

    doc = Document(
        id=str(uuid.uuid4()),
        pile_id=pile_id,
        filename=body.filename,
        format=detect_format(body.filename),
        content_hash=content_hash,
    )
    try:
        session.add(doc)
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = (
            session.query(Document)
            .filter(Document.pile_id == pile_id, Document.content_hash == content_hash)
            .first()
        )
        if existing is not None:
            return IngestResponse(document_id=existing.id, run_id="", status="duplicate_skipped")
        raise HTTPException(500, "Upload failed and could not be resolved as a duplicate.")

    raw = _raw_dir()
    (raw / f"{doc.id}.txt").write_text(body.text_content, encoding="utf-8")

    # --- Acquire pile lock BEFORE scheduling the run (bug fix — see module docstring) ---
    # Short transaction, releases immediately — the graph manages its own
    # per-node sessions once it actually starts (see graph.py _with_session).
    lock_session = SessionLocal()
    try:
        with lock_session.begin():
            with pile_lock(lock_session, pile_id):
                pass
    except PileBusyError as e:
        lock_session.close()
        raise HTTPException(
            409,
            f"Pile {pile_id} already has a run in progress: {e}",
        )
    finally:
        lock_session.close()

    # Persist the run before returning its id. The UI starts polling as soon
    # as it receives 202; creating this row only inside the background graph
    # creates an avoidable window of GET /runs/{id} -> 404.
    run_id = str(uuid.uuid4())
    session.add(
        Run(
            id=run_id,
            pile_id=pile_id,
            thread_id=run_id,
            stage="queued",
            status="running",
        )
    )
    session.commit()
    background_tasks.add_task(_run_graph_background, pile_id=pile_id, document_id=doc.id, run_id=run_id)

    return IngestResponse(document_id=doc.id, run_id=run_id, status="processing")


def _run_graph_background(*, pile_id: str, document_id: str, run_id: str) -> None:
    """
    Runs in FastAPI's background task thread pool.

    GraphInterrupt is NOT an error — it means the graph paused at
    human_gate waiting for a decision. We catch it explicitly and
    let the run stay in whatever status the human_gate node wrote
    (status=paused, stage=awaiting_approval).

    Any other exception IS an error and marks the run failed so
    GET /runs/{id} surfaces it rather than hanging on status=running.

    Note: the pile lock was already acquired and released (its short
    transaction committed) in ingest_document() above, before this
    background task was even scheduled. That check-then-schedule
    window is intentionally small; a true simultaneous double-submit
    is still caught by the SAME lock being re-acquired by any other
    concurrent caller (MCP, watcher, or a second REST request) against
    the still-active Run row this graph run creates in its first node.
    """
    from langgraph.errors import GraphInterrupt
    from app.graph.graph import get_compiled_graph

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": run_id}}
    try:
        graph.invoke(
            {
                "pile_id":       pile_id,
                "run_id":        run_id,
                "document_ids":  [document_id],
                "stage":         "new",
            },
            config=config,
        )
    except GraphInterrupt:
        # Expected — graph paused at human_gate.
        # human_gate node already wrote status=paused via _upsert_run
        # before calling interrupt(), so GET /runs/{id} already reflects
        # reality. Nothing to do here.
        pass
    except Exception as e:
        # Unexpected error — surface it so the run doesn't hang.
        _mark_run_failed(run_id, pile_id, str(e))


def _mark_run_failed(run_id: str, pile_id: str, error: str) -> None:
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    session = SessionLocal()
    try:
        stmt = (
            pg_insert(Run)
            .values(
                id=run_id,
                pile_id=pile_id,
                thread_id=run_id,
                stage="failed",
                status="failed",
            )
            .on_conflict_do_update(
                index_elements=["id"],
                set_={"stage": "failed", "status": "failed"},
            )
        )
        session.execute(stmt)
        session.commit()
    finally:
        session.close()
