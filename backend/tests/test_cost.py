"""
Behavior #10: It knows what it cost.

Tests that:
  - Every LLM call logs a RunCost row with stage, provider, tokens, latency.
  - The cost endpoint aggregates correctly by stage.
  - Zero-cost runs (mocked LLM) report 0.0, not None or an error.
  - Two runs don't bleed costs into each other.
"""
import uuid
from unittest.mock import patch, MagicMock

import pytest
from sqlalchemy import select

from app.db.models import RunCost
from app.graph.llm import call_structured, _log_cost
from app.graph.schemas import ExtractionResult
from tests.conftest import make_pile, make_run


def test_log_cost_writes_row(session):
    """_log_cost() must write exactly one RunCost row."""
    pile = make_pile(session)
    run  = make_run(session, pile)

    _log_cost(
        session=session,
        run_id=run.id,
        stage="extract",
        provider="gemini",
        tokens_in=100,
        tokens_out=50,
        latency_ms=342,
    )
    session.flush()

    rows = session.scalars(
        select(RunCost).where(RunCost.run_id == run.id)
    ).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.stage        == "extract"
    assert row.provider     == "gemini"
    assert row.tokens_in    == 100
    assert row.tokens_out   == 50
    assert row.latency_ms   == 342
    assert row.cost_usd     == 0.0


def test_cost_endpoint_aggregates_by_stage(session):
    """
    The cost endpoint must sum tokens and latency per stage correctly.
    Two calls to the same stage must be aggregated into one row.
    """
    pile = make_pile(session)
    run  = make_run(session, pile)
    session.commit()

    _log_cost(session=session, run_id=run.id, stage="extract",
              provider="gemini", tokens_in=100, tokens_out=50, latency_ms=300)
    _log_cost(session=session, run_id=run.id, stage="extract",
              provider="gemini", tokens_in=80,  tokens_out=40, latency_ms=250)
    _log_cost(session=session, run_id=run.id, stage="reconcile",
              provider="groq",   tokens_in=200, tokens_out=100, latency_ms=500)
    session.commit()

    from app.routes.cost import get_run_cost
    report = get_run_cost(run.id, session)

    stages = {s.stage: s for s in report.by_stage}

    assert "extract"   in stages, "extract stage must appear in cost report"
    assert "reconcile" in stages, "reconcile stage must appear in cost report"

    extract = stages["extract"]
    assert extract.tokens_in    == 180,  "tokens_in for extract should be summed"
    assert extract.tokens_out   == 90,   "tokens_out for extract should be summed"
    assert extract.total_latency_ms == 550, "latency for extract should be summed"
    assert extract.calls        == 2,    "call count for extract should be 2"

    assert report.total_tokens == 180 + 90 + 200 + 100


def test_costs_isolated_between_runs(session):
    """
    RunCost rows for run A must not appear in run B's cost report.
    """
    pile  = make_pile(session)
    run_a = make_run(session, pile)
    run_b = make_run(session, pile, status="done")
    session.commit()

    _log_cost(session=session, run_id=run_a.id, stage="extract",
              provider="gemini", tokens_in=500, tokens_out=200, latency_ms=1000)
    session.commit()

    from app.routes.cost import get_run_cost
    report_b = get_run_cost(run_b.id, session)

    assert report_b.total_tokens == 0, (
        "Run B must not see Run A's cost rows"
    )
    assert report_b.total_cost_usd == 0.0


def test_zero_cost_run_reports_cleanly(session):
    """
    A run with no LLM calls reports 0.0 total cost — not None, not an error.
    """
    pile = make_pile(session)
    run  = make_run(session, pile, status="done")
    session.commit()

    from app.routes.cost import get_run_cost
    report = get_run_cost(run.id, session)

    assert report.total_cost_usd == 0.0
    assert report.total_tokens   == 0
    assert report.by_stage       == []
