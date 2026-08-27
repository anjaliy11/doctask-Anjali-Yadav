from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import RunCost
from app.db.session import get_session

router = APIRouter(prefix="/runs", tags=["cost"])


class StageCost(BaseModel):
    stage: str
    provider: str
    calls: int
    tokens_in: int
    tokens_out: int
    total_latency_ms: int
    cost_usd: float


class RunCostReport(BaseModel):
    run_id: str
    by_stage: list[StageCost]
    total_cost_usd: float
    total_tokens: int


@router.get("/{run_id}/cost", response_model=RunCostReport)
def get_run_cost(run_id: str, session: Session = Depends(get_session)):
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
        StageCost(
            stage=r[0], provider=r[1], calls=r[2],
            tokens_in=r[3] or 0, tokens_out=r[4] or 0,
            total_latency_ms=r[5] or 0, cost_usd=float(r[6] or 0),
        )
        for r in rows
    ]
    return RunCostReport(
        run_id=run_id,
        by_stage=by_stage,
        total_cost_usd=sum(s.cost_usd for s in by_stage),
        total_tokens=sum(s.tokens_in + s.tokens_out for s in by_stage),
    )
