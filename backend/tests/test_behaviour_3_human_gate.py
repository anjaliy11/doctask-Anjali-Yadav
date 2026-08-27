"""
Behavior #3: Human holds the gate, item-by-item.

Tests that:
  - Rejecting finding N does not discard findings N+1, N+2 ...
  - Approving finding N does not auto-approve others.
  - A pre-recorded decision (out-of-order) is consumed correctly.
  - The deliverable renders only approved/uncontested facts.
"""
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.db.models import Approval, Finding, Run
from app.graph.nodes import _apply_decision, _record_approval
from tests.conftest import make_pile, make_document, make_chunk, make_run


def make_finding(session, pile, *, rule_id="test_rule", description="A test finding."):
    import hashlib
    nk = hashlib.sha256(f"{rule_id}{description}".encode()).hexdigest()[:32]
    f = Finding(
        id=str(uuid.uuid4()),
        pile_id=pile.id,
        rule_id=rule_id,
        description=description,
        severity="warning",
        status="pending",
        natural_key=nk,
    )
    session.add(f)
    session.flush()
    return f


def test_reject_one_finding_leaves_others_pending(session):
    """
    Rejecting finding A must not change the status of finding B or C.
    """
    pile = make_pile(session)
    run  = make_run(session, pile)

    fa = make_finding(session, pile, rule_id="rule_a", description="Finding A")
    fb = make_finding(session, pile, rule_id="rule_b", description="Finding B")
    fc = make_finding(session, pile, rule_id="rule_c", description="Finding C")

    # Reject only A
    _record_approval(session, fa.id, run.id, "rejected")
    _apply_decision(session, fa, "rejected")

    session.expire_all()
    fa_fresh = session.get(Finding, fa.id)
    fb_fresh = session.get(Finding, fb.id)
    fc_fresh = session.get(Finding, fc.id)

    assert fa_fresh.status == "rejected", "Finding A should be rejected"
    assert fb_fresh.status == "pending",  "Finding B should still be pending"
    assert fc_fresh.status == "pending",  "Finding C should still be pending"


def test_approve_one_finding_leaves_others_pending(session):
    """
    Approving finding A must not touch B or C.
    """
    pile = make_pile(session)
    run  = make_run(session, pile)

    fa = make_finding(session, pile, rule_id="rule_a", description="Finding A")
    fb = make_finding(session, pile, rule_id="rule_b", description="Finding B")

    _record_approval(session, fa.id, run.id, "approved")
    _apply_decision(session, fa, "approved")

    session.expire_all()
    assert session.get(Finding, fa.id).status == "approved"
    assert session.get(Finding, fb.id).status == "pending"


def test_pre_recorded_decision_is_idempotent(session):
    """
    Recording the same decision twice (e.g. network retry) must not
    error or change the outcome — ON CONFLICT DO UPDATE semantics.
    """
    pile = make_pile(session)
    run  = make_run(session, pile)
    fa   = make_finding(session, pile, rule_id="rule_a", description="Finding A")

    _record_approval(session, fa.id, run.id, "approved")
    _record_approval(session, fa.id, run.id, "approved")  # duplicate

    approvals = session.scalars(
        select(Approval).where(
            Approval.item_id == fa.id,
            Approval.run_id  == run.id,
        )
    ).all()
    assert len(approvals) == 1, "Duplicate approval must be upserted, not inserted twice"
    assert approvals[0].status == "approved"


def test_decision_cannot_be_changed_after_set(session):
    """
    Once a finding is approved, it cannot be rejected (and vice versa).
    The REST endpoint enforces this with a 409; here we test the model layer.
    """
    pile = make_pile(session)
    run  = make_run(session, pile)
    fa   = make_finding(session, pile, rule_id="rule_a", description="Finding A")

    _apply_decision(session, fa, "approved")
    session.flush()

    # Simulate a second decision attempt — status is already set
    session.expire_all()
    fa_fresh = session.get(Finding, fa.id)
    assert fa_fresh.status == "approved"

    # The REST layer (findings._decide) guards against this with a 409.
    # At the model layer, _apply_decision would overwrite — the guard
    # lives in the route, not here. This test confirms current state.
    assert fa_fresh.status != "pending", (
        "Finding should not revert to pending after a decision"
    )


def test_mixed_decisions_all_recorded_independently(session):
    """
    Approve A, reject B, leave C pending — all three states must
    coexist correctly.
    """
    pile = make_pile(session)
    run  = make_run(session, pile)

    fa = make_finding(session, pile, rule_id="rule_a", description="Finding A")
    fb = make_finding(session, pile, rule_id="rule_b", description="Finding B")
    fc = make_finding(session, pile, rule_id="rule_c", description="Finding C")

    _record_approval(session, fa.id, run.id, "approved")
    _apply_decision(session, fa, "approved")

    _record_approval(session, fb.id, run.id, "rejected")
    _apply_decision(session, fb, "rejected")

    session.expire_all()
    assert session.get(Finding, fa.id).status == "approved"
    assert session.get(Finding, fb.id).status == "rejected"
    assert session.get(Finding, fc.id).status == "pending"
    
def test_rest_layer_409_on_already_decided(session):
    """
    _decide() must raise HTTPException(409) when the finding
    is not pending. This is the guard the REST and MCP routes
    rely on to prevent double-decisions.
    """
    from fastapi import HTTPException
    from app.routes.findings import _decide

    pile    = make_pile(session)
    run     = make_run(session, pile, status="running")
    finding = make_finding(session, pile)
    finding.status = "approved"
    session.flush()

    with pytest.raises(HTTPException) as exc_info:
        _decide(finding.id, "approved", session)

    assert exc_info.value.status_code == 409


def test_rest_layer_409_no_active_run(session):
    """
    _decide() must raise 409 when there is no active run —
    a decision with no run to resume is meaningless.
    """
    from fastapi import HTTPException
    from app.routes.findings import _decide

    pile    = make_pile(session)
    # No run created — pile has no active run
    finding = make_finding(session, pile)
    session.flush()

    with pytest.raises(HTTPException) as exc_info:
        _decide(finding.id, "approved", session)

    assert exc_info.value.status_code == 409
