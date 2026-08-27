from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import Run
from app.db.session import get_session

router = APIRouter(prefix="/runs", tags=["runs"])


class PendingInterrupt(BaseModel):
    item_type: str
    item_id: str
    description: str
    # *** BUG FIX ***
    # This field was missing. mcp_server.py::get_run_status calls
    # pending_model.model_dump() on this model and returns it directly
    # to MCP callers, but the raw interrupt payload written by
    # nodes.py::human_gate always includes rule_id — so every MCP
    # caller silently lost rule_id even though it was present at the
    # source. mcp_server.py::ingest_document (a different code path)
    # reads rule_id straight off the raw payload dict and DID include
    # it, so the two "what are we paused on" responses were
    # inconsistent with each other for no reason. Adding it here fixes
    # both call sites at once since they share this model.
    rule_id: str | None = None
    severity: str | None = None


class RunStatus(BaseModel):
    run_id: str
    pile_id: str
    stage: str
    status: str
    pending_interrupt: PendingInterrupt | None = None


def _get_pending_interrupt(run_id: str) -> PendingInterrupt | None:
    """
    Read-only inspection of LangGraph checkpoint state.
    Returns the interrupt payload if the run is currently paused,
    None otherwise. Never resumes the graph.
    """
    from app.graph.graph import get_compiled_graph

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": run_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception:
        # Checkpoint doesn't exist yet (run just created but graph
        # hasn't started) — not an error, just not paused.
        return None

    if not snapshot or not snapshot.tasks:
        return None

    for task in snapshot.tasks:
        if task.interrupts:
            payload = task.interrupts[0].value
            return PendingInterrupt(
                item_type=payload["item_type"],
                item_id=payload["item_id"],
                description=payload["description"],
                rule_id=payload.get("rule_id"),
                severity=payload.get("severity"),
            )
    return None


@router.get("/{run_id}", response_model=RunStatus)
def get_run(run_id: str, session: Session = Depends(get_session)):
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")

    pending = None
    if run.status in ("running", "paused"):
        try:
            pending = _get_pending_interrupt(run_id)
        except Exception:
            pending = None

    return RunStatus(
        run_id=run.id,
        pile_id=run.pile_id,
        stage=run.stage,
        status=run.status,
        pending_interrupt=pending,
    )


@router.post("/{run_id}/resume", response_model=RunStatus)
def resume_run(run_id: str, session: Session = Depends(get_session)):
    """
    Re-attach to a run left mid-flight after a server restart.
    Safe to call on a finished run — no-op in that case.

    This handles behavior #2: kill the process, restart, call this
    endpoint with the run_id, and the graph continues from its last
    checkpoint without re-running completed nodes.
    """
    from app.graph.graph import get_compiled_graph

    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": run_id}}

    try:
        snapshot = graph.get_state(config)
    except Exception:
        return get_run(run_id, session)

    if not snapshot or not snapshot.next:
        # Nothing to resume — already at END or no checkpoint yet
        return get_run(run_id, session)

    # Resume from last checkpoint — passes None as input because
    # LangGraph reads state from the checkpoint, not from this call.
    graph.invoke(None, config=config)
    return get_run(run_id, session)
