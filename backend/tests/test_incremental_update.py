"""
Incremental update: an update costs like an update.

Tests that:
  - Adding a new document to a pile only recomputes sections whose
    source facts were touched by the new document.
  - Untouched sections have byte-identical content_hash before and after.
  - A new document that contradicts an existing fact surfaces a
    contradiction finding, never silently resolves it.
"""
import hashlib
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.db.models import DeliverableSection, Finding, SectionSource
from app.graph.nodes import compose_deliverable
from app.graph.schemas import ContradictionResult, Contradiction
from tests.conftest import (
    make_chunk,
    make_document,
    make_fact,
    make_pile,
    make_run,
    make_state,
)


def test_untouched_sections_are_byte_identical(session):
    """
    After adding a new document that only touches slot B,
    the section for slot A must have an identical content_hash.
    """
    pile = make_pile(session)
    run1 = make_run(session, pile)

    # Doc 1: has facts in slot A and slot B
    doc1   = make_document(session, pile, text="Slot A and Slot B content.")
    chunk1 = make_chunk(session, doc1, text="Net-30 payment terms.")
    chunk2 = make_chunk(
        session, doc1,
        text="Termination requires 90 days notice.",
        char_start=100,
    )

    make_fact(
        session, pile, chunk1,
        claim="Payment terms are Net-30.",
        claim_slot="payment_terms",
        natural_key="fact_payment_001",
    )
    make_fact(
        session, pile, chunk2,
        claim="Termination notice is 90 days.",
        claim_slot="termination_notice",
        natural_key="fact_termination_001",
    )
    session.commit()

    # First compose — builds both sections
    state1 = make_state(pile, run1, doc1)
    compose_deliverable(state1, session)

    payment_section_before = session.scalar(
        select(DeliverableSection).where(
            DeliverableSection.pile_id == pile.id,
            DeliverableSection.title   == "payment_terms",
        )
    )
    termination_section_before = session.scalar(
        select(DeliverableSection).where(
            DeliverableSection.pile_id == pile.id,
            DeliverableSection.title   == "termination_notice",
        )
    )
    assert payment_section_before,     "payment_terms section must exist after first compose"
    assert termination_section_before, "termination_notice section must exist after first compose"

    hash_payment_before     = payment_section_before.content_hash
    hash_termination_before = termination_section_before.content_hash

    # Doc 2: only touches payment_terms (new Net-60 fact)
    run2  = make_run(session, pile)
    doc2  = make_document(session, pile, text="Net-60 payment terms per amendment.")
    chunk3 = make_chunk(
        session, doc2,
        text="Net-60 payment terms per amendment.",
        char_start=0,
    )
    make_fact(
        session, pile, chunk3,
        claim="Payment terms are Net-60 per Amendment No. 1.",
        claim_slot="payment_terms",
        natural_key="fact_payment_002",
    )
    session.commit()

    # Second compose — only payment_terms should change
    state2 = make_state(pile, run2, doc2)
    compose_deliverable(state2, session)

    session.expire_all()
    payment_section_after = session.scalar(
        select(DeliverableSection).where(
            DeliverableSection.pile_id == pile.id,
            DeliverableSection.title   == "payment_terms",
        )
    )
    termination_section_after = session.scalar(
        select(DeliverableSection).where(
            DeliverableSection.pile_id == pile.id,
            DeliverableSection.title   == "termination_notice",
        )
    )

    assert payment_section_after.content_hash != hash_payment_before, (
        "payment_terms section must be recomputed when new payment fact arrives"
    )
    assert termination_section_after.content_hash == hash_termination_before, (
        "termination_notice section must be byte-identical — it was not touched"
    )


def test_contradiction_surfaced_not_silently_resolved(session):
    """
    When a new document introduces a fact that contradicts an existing
    fact in the same slot, a contradiction Finding must be created.
    The deliverable must show the conflict, not silently pick one value.
    """
    from app.graph.nodes import reconcile

    pile = make_pile(session)
    run  = make_run(session, pile)

    # Existing fact: Net-30
    doc1   = make_document(session, pile, text="Net-30 payment terms.")
    chunk1 = make_chunk(session, doc1, text="Net-30 payment terms.")
    make_fact(
        session, pile, chunk1,
        claim="Payment terms are Net-30.",
        claim_slot="payment_terms",
        natural_key="fact_net30",
    )

    # New fact: Net-60 (contradiction)
    doc2   = make_document(session, pile, text="Net-60 per Amendment No. 1.")
    chunk2 = make_chunk(session, doc2, text="Net-60 per Amendment No. 1.")
    fact2  = make_fact(
        session, pile, chunk2,
        claim="Payment terms are Net-60 per Amendment No. 1.",
        claim_slot="payment_terms",
        natural_key="fact_net60",
    )
    session.commit()

    contradiction_result = ContradictionResult(
        contradictions=[
            Contradiction(
                claim_slot="payment_terms",
                fact_id_a="fact_net30",
                fact_id_b="fact_net60",
                explanation=(
                    "Base contract states Net-30; Amendment No. 1 states Net-60. "
                    "These payment terms directly contradict each other."
                ),
            )
        ]
    )

    state = make_state(pile, run, doc2)
    with patch(
        "app.graph.nodes.call_structured",
        return_value=contradiction_result,
    ):
        reconcile(state, session)

    findings = session.scalars(
        select(Finding).where(
            Finding.pile_id  == pile.id,
            Finding.rule_id  == "contradiction",
            Finding.claim_slot == "payment_terms",
        )
    ).all()

    assert findings, (
        "A contradiction finding must be created when Net-30 conflicts with Net-60"
    )
    assert all(f.status == "pending" for f in findings), (
        "Contradiction findings must start as pending — never auto-resolved"
    )
    assert all(f.severity == "critical" for f in findings), (
        "Contradiction findings must be severity=critical"
    )