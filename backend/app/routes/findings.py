from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Finding, Run
from app.db.session import get_session
from app.routes.runs import RunStatus, _get_pending_interrupt

router = APIRouter(prefix="/findings", tags=["findings"])


class DecisionResponse(BaseModel):
    finding_id: str
    decision: str
    run_status: RunStatus


def _decide(finding_id: str, decision: str, session: Session) -> DecisionResponse:
    from app.graph.graph import get_compiled_graph
    from langgraph.types import Command

    finding = session.get(Finding, finding_id)
    if finding is None:
        raise HTTPException(404, f"Finding {finding_id} not found")
    if finding.status != "pending":
        raise HTTPException(
            409, f"Finding {finding_id} was already {finding.status}; decisions are one-time."
        )

    active_run = session.scalar(
        select(Run)
        .where(Run.pile_id == finding.pile_id, Run.status.in_(["running", "paused"]))
        .order_by(Run.started_at.desc())
    )
    if active_run is None:
        raise HTTPException(
            409, f"No active run is currently paused on pile {finding.pile_id} for this finding."
        )

    pending = _get_pending_interrupt(active_run.id)
    if pending is None:
        raise HTTPException(
            409, f"Run {active_run.id} has no pending interrupt right now — try polling "
            f"GET /runs/{active_run.id} and retry once it's actually paused."
        )
    if pending.item_id != finding_id:
        raise HTTPException(
            409,
            f"Run {active_run.id} is currently waiting on a decision for finding "
            f"{pending.item_id}, not {finding_id}. Findings must be decided in the "
            f"order the run presents them — refresh and approve/reject {pending.item_id} first.",
        )

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": active_run.id}}
    graph.invoke(Command(resume=decision), config=config)

    session.expire_all()
    run = session.get(Run, active_run.id)
    status = RunStatus(run_id=run.id, pile_id=run.pile_id, stage=run.stage, status=run.status)
    return DecisionResponse(finding_id=finding_id, decision=decision, run_status=status)


@router.post("/{finding_id}/approve", response_model=DecisionResponse)
def approve_finding(finding_id: str, session: Session = Depends(get_session)):
    return _decide(finding_id, "approved", session)


@router.post("/{finding_id}/reject", response_model=DecisionResponse)
def reject_finding(finding_id: str, session: Session = Depends(get_session)):
    return _decide(finding_id, "rejected", session)


@router.get("/pile/{pile_id}")
def list_findings(pile_id: str, session: Session = Depends(get_session)):
    findings = session.scalars(
        select(Finding).where(Finding.pile_id == pile_id).order_by(Finding.created_at)
    ).all()
    return [
        {
            "id": f.id,
            "rule_id": f.rule_id,
            "description": f.description,
            "severity": f.severity,
            "status": f.status,
            "claim_slot": f.claim_slot,
        }
        for f in findings
    ]
