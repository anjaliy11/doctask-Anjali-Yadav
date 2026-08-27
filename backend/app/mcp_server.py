"""
MCP server — behavior #4: "a machine can drive it."

Every operation the REST API exposes is also reachable as an MCP tool,
including the human-gate approve/reject operations. A machine driver
calls the same code paths a human would click through in the UI.

Design notes:
  - Each tool opens its own DB session and closes it in a finally block.
    MCP tools run in the FastMCP thread pool; sessions must never be
    shared across tool calls.
  - ingest_document runs the graph SYNCHRONOUSLY. This is correct for
    a machine driver: the caller blocks until the run reaches done,
    failed, or human_gate, then gets back a real terminal status rather
    than a run_id it has to poll. The REST endpoint uses a background
    task because a browser can't block; a machine driver has no such
    constraint.
  - approve_finding / reject_finding re-use the findings._decide()
    function directly (not the FastAPI route wrapper) to avoid
    constructing fake Request objects.

*** BUG FIX ***
ingest_document previously took filename + text_content and wrote
straight to raw/{id}.txt with no format check at all — the same gap
that existed in the REST route and the watcher before their fixes.
A caller could hand this tool a PDF's binary content under a
"invoice.pdf" filename (or even just an honest UTF-8 text_content
with a misleading .pdf filename) and it would be persisted and handed
to classify_and_chunk() as if it were guaranteed-readable text. Now
uses the same app.formats.is_supported() check as ingest.py and
watcher.py, returning a structured {"error": "unsupported_format", ...}
result instead of a fake success. This is the third and last of the
three ingestion entry points (REST, watcher, MCP) to get this check —
all three now agree.
"""
import logging
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from app.db.models import Pile, Run
from app.db.session import SessionLocal
from app.formats import is_supported, unsupported_format_message

logger = logging.getLogger(__name__)

mcp = FastMCP("doctask-pile-agent")


# ---------------------------------------------------------------------------
# Pile management
# ---------------------------------------------------------------------------

@mcp.tool()
def create_pile(name: str, rules: list[dict] | None = None) -> dict:
    """
    Create a new document pile with an optional rule checklist.

    rules format: [{"rule_id": "...", "description": "..."}, ...]
    Returns: {"id": "...", "name": "...", "rules": [...]}
    """
    from app.routes.piles import CreatePileRequest, RuleIn

    session = SessionLocal()
    try:
        body = CreatePileRequest(
            name=name,
            rules=[RuleIn(**r) for r in (rules or [])],
        )
        pile = Pile(
            name=body.name,
            rules_json={"rules": [r.model_dump() for r in body.rules]},
        )
        session.add(pile)
        session.commit()
        session.refresh(pile)
        return {
            "id": pile.id,
            "name": pile.name,
            "rules": body.rules and [r.model_dump() for r in body.rules] or [],
        }
    finally:
        session.close()


@mcp.tool()
def list_piles() -> list[dict]:
    """List all piles with their rule checklists."""
    session = SessionLocal()
    try:
        piles = session.query(Pile).order_by(Pile.created_at.desc()).all()
        return [
            {
                "id": p.id,
                "name": p.name,
                "rules": (p.rules_json or {}).get("rules", []),
                "created_at": p.created_at.isoformat(),
            }
            for p in piles
        ]
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

@mcp.tool()
def ingest_document(pile_id: str, filename: str, text_content: str) -> dict:
    """
    Add a document to a pile and run the full graph pipeline.

    Blocks until the run reaches: done | failed | paused (human_gate).
    Returns run_id and final status so the caller knows immediately
    whether human approval is needed before proceeding.

    Only .txt/.md filenames are accepted — see app/formats.py. A
    filename with an unsupported extension (e.g. .pdf, .docx) returns
    {"error": "unsupported_format", ...} before anything is written.

    Returns:
      {
        "document_id": "...",
        "run_id": "...",
        "status": "done" | "paused" | "failed" | "duplicate_skipped",
        "pending_interrupt": {...} | null,
        "error": "..." | null
      }
    """
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from app.db.models import Document
    from app.graph.graph import get_compiled_graph
    from app.graph.state import PileState
    from app.locking import PileBusyError, pile_lock
    from app.routes.ingest import MAX_DOCUMENT_BYTES, _content_hash

    # --- Format check FIRST, before anything is written (bug fix) ---
    if not is_supported(filename):
        return {
            "error": "unsupported_format",
            "cause": unsupported_format_message(filename),
        }

    if len(text_content.encode()) > MAX_DOCUMENT_BYTES:
        return {
            "error": "document_too_large",
            "cause": (
                f"Content exceeds {MAX_DOCUMENT_BYTES} bytes. "
                "Split the document or use the file-upload path."
            ),
        }

    # --- Validate pile exists ---
    session = SessionLocal()
    try:
        pile = session.get(Pile, pile_id)
        if pile is None:
            return {"error": "pile_not_found", "pile_id": pile_id}

        # --- Deduplication ---
        content_hash = _content_hash(text_content)
        existing = session.scalar(
            select(Document).where(
                Document.pile_id == pile_id,
                Document.content_hash == content_hash,
            )
        )
        if existing is not None:
            return {
                "document_id": existing.id,
                "run_id": "",
                "status": "duplicate_skipped",
                "pending_interrupt": None,
                "error": None,
            }

        # --- Persist document row ---
        doc_id = str(uuid.uuid4())
        doc = Document(
            id=doc_id,
            pile_id=pile_id,
            filename=filename,
            format=Path(filename).suffix.lstrip(".") or "txt",
            content_hash=content_hash,
        )
        session.add(doc)
        session.commit()

    finally:
        session.close()

    # --- Write raw file ---
    from app.config import get_settings
    raw_dir = Path(get_settings().inbox_dir).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{doc_id}.txt").write_text(text_content, encoding="utf-8")

    # --- Acquire pile lock (short transaction, releases immediately) ---
    lock_session = SessionLocal()
    try:
        with lock_session.begin():
            with pile_lock(lock_session, pile_id):
                pass
    except PileBusyError as e:
        return {
            "document_id": doc_id,
            "run_id": "",
            "status": "pile_busy",
            "error": str(e),
        }
    finally:
        lock_session.close()

    # --- Run graph synchronously ---
    # run_id == thread_id by contract (see _upsert_run in nodes.py)
    run_id = str(uuid.uuid4())
    state: PileState = {
        "pile_id": pile_id,
        "run_id": run_id,
        "document_ids": [doc_id],
        "stage": "new",
    }
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": run_id}}

    run_error: str | None = None
    try:
        graph.invoke(state, config)
    except Exception as e:
        run_error = str(e)
        logger.exception(
            "MCP ingest_document: graph failed for pile=%s doc=%s",
            pile_id, doc_id,
        )
        # Mark run failed so get_run_status reflects reality
        _mark_run_failed(run_id, pile_id)

    # --- Determine final status from checkpoint ---
    pending_interrupt = None
    final_status = "failed" if run_error else "done"

    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.tasks:
            for task in snapshot.tasks:
                if task.interrupts:
                    payload = task.interrupts[0].value
                    pending_interrupt = {
                        "item_type":   payload.get("item_type"),
                        "item_id":     payload.get("item_id"),
                        "rule_id":     payload.get("rule_id"),
                        "description": payload.get("description"),
                        "severity":    payload.get("severity"),
                    }
                    final_status = "paused"
                    break
    except Exception:
        logger.warning(
            "MCP ingest_document: could not read checkpoint state for run=%s",
            run_id,
        )

    return {
        "document_id": doc_id,
        "run_id": run_id,
        "status": final_status,
        "pending_interrupt": pending_interrupt,
        "error": run_error,
    }


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

@mcp.tool()
def get_run_status(run_id: str) -> dict:
    """
    Check a run's current stage, status, and what finding it is
    paused on (if any).

    Returns:
      {
        "run_id": "...",
        "pile_id": "...",
        "stage": "...",
        "status": "running" | "paused" | "done" | "failed",
        "pending_interrupt": {...} | null
      }
    """
    from app.routes.runs import _get_pending_interrupt

    session = SessionLocal()
    try:
        run = session.get(Run, run_id)
        if run is None:
            return {"error": "run_not_found", "run_id": run_id}

        pending = None
        if run.status in ("running", "paused"):
            try:
                pending_model = _get_pending_interrupt(run_id)
                if pending_model:
                    pending = pending_model.model_dump()
            except Exception:
                pass

        return {
            "run_id": run.id,
            "pile_id": run.pile_id,
            "stage": run.stage,
            "status": run.status,
            "pending_interrupt": pending,
        }
    finally:
        session.close()


@mcp.tool()
def resume_run(run_id: str) -> dict:
    """
    Re-attach to a run left mid-flight after a server restart.
    Safe to call on a finished run — returns current status as a no-op.

    This is behavior #2: kill the process, restart, call this tool,
    and the graph continues from its last checkpoint.
    """
    from app.graph.graph import get_compiled_graph

    session = SessionLocal()
    try:
        run = session.get(Run, run_id)
        if run is None:
            return {"error": "run_not_found", "run_id": run_id}
    finally:
        session.close()

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": run_id}}

    try:
        snapshot = graph.get_state(config)
    except Exception:
        return get_run_status(run_id)

    if not snapshot or not snapshot.next:
        return get_run_status(run_id)

    try:
        graph.invoke(None, config=config)
    except Exception as e:
        logger.exception("MCP resume_run: graph error for run=%s", run_id)
        return {"error": str(e), "run_id": run_id}

    return get_run_status(run_id)


# ---------------------------------------------------------------------------
# Findings — human gate operations
# ---------------------------------------------------------------------------

@mcp.tool()
def list_pending_findings(pile_id: str) -> list[dict]:
    """
    List all findings for a pile.
    Includes status: pending | approved | rejected.
    Pending findings require a human decision before the run can proceed.
    """
    from sqlalchemy import select
    from app.db.models import Finding

    session = SessionLocal()
    try:
        findings = session.scalars(
            select(Finding)
            .where(Finding.pile_id == pile_id)
            .order_by(Finding.created_at)
        ).all()
        return [
            {
                "id": f.id,
                "rule_id": f.rule_id,
                "description": f.description,
                "severity": f.severity,
                "status": f.status,
                "claim_slot": f.claim_slot,
                "created_at": f.created_at.isoformat(),
            }
            for f in findings
        ]
    finally:
        session.close()


@mcp.tool()
def approve_finding(finding_id: str) -> dict:
    """
    Approve a finding. If the graph is currently paused on this
    finding, it resumes automatically. If it is paused on a different
    finding, this decision is pre-recorded and applied when the graph
    reaches it.

    Returns the finding's new status and current run state.
    """
    return _decide_finding(finding_id, "approved")


@mcp.tool()
def reject_finding(finding_id: str) -> dict:
    """
    Reject a finding. Same resumption semantics as approve_finding.
    Rejected contradiction findings cause the deliverable to show
    the conflict as dismissed.
    """
    return _decide_finding(finding_id, "rejected")


# ---------------------------------------------------------------------------
# Cost reporting
# ---------------------------------------------------------------------------

@mcp.tool()
def get_run_cost(run_id: str) -> dict:
    """
    Token, latency, and cost breakdown for a run, by stage and provider.

    Returns:
      {
        "run_id": "...",
        "by_stage": [{"stage": "...", "provider": "...", "calls": N,
                      "tokens_in": N, "tokens_out": N,
                      "total_latency_ms": N, "cost_usd": 0.0}, ...],
        "total_cost_usd": 0.0,
        "total_tokens": N
      }
    """
    from sqlalchemy import func, select
    from app.db.models import RunCost

    session = SessionLocal()
    try:
        rows = session.execute(
            select(
                RunCost.stage,
                RunCost.provider,
                func.count(RunCost.id),
                func.sum(RunCost.tokens_in),
                func.sum(RunCost.tokens_out),
                func.sum(RunCost.latency_ms),
                func.sum(RunCost.cost_usd),
            )
            .where(RunCost.run_id == run_id)
            .group_by(RunCost.stage, RunCost.provider)
        ).all()

        by_stage = [
            {
                "stage":            r[0],
                "provider":         r[1],
                "calls":            r[2],
                "tokens_in":        r[3] or 0,
                "tokens_out":       r[4] or 0,
                "total_latency_ms": r[5] or 0,
                "cost_usd":         float(r[6] or 0),
            }
            for r in rows
        ]
        return {
            "run_id":         run_id,
            "by_stage":       by_stage,
            "total_cost_usd": sum(s["cost_usd"] for s in by_stage),
            "total_tokens":   sum(
                s["tokens_in"] + s["tokens_out"] for s in by_stage
            ),
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

@mcp.tool()
def whoami() -> dict:
    """
    Confirms which pile-agent instance this MCP connection is talking to.
    Useful as a first call to verify connectivity and credentials.
    """
    session = SessionLocal()
    try:
        pile_count = session.query(Pile).count()
        run_count  = session.query(Run).count()
        return {
            "service":      "doctask-pile-agent",
            "version":      "0.1.0",
            "piles_known":  pile_count,
            "runs_known":   run_count,
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _decide_finding(finding_id: str, decision: str) -> dict:
    """
    Shared implementation for approve/reject.
    Calls findings._decide() directly — the same function the REST
    route calls — without constructing a fake FastAPI Request.
    """
    from app.routes.findings import _decide

    session = SessionLocal()
    try:
        result = _decide(finding_id, decision, session)
        return {
            "finding_id": result.finding_id,
            "decision":   result.decision,
            "note":       result.note,
            "run_status": result.run_status.model_dump(),
        }
    except Exception as e:
        # Surface structured errors rather than crashing the MCP tool
        return {
            "error":      type(e).__name__,
            "cause":      str(e),
            "finding_id": finding_id,
            "decision":   decision,
        }
    finally:
        session.close()


def _mark_run_failed(run_id: str, pile_id: str) -> None:
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
