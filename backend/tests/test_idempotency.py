"""
Idempotency: re-running the same input produces identical state.

Tests that:
  - Ingesting the same document twice creates exactly one Document row.
  - Running extract_facts twice on the same chunk creates exactly one Fact.
  - Running compose_deliverable twice produces byte-identical sections.
  - Running classify_and_chunk twice creates exactly one Chunk per span.
"""
import hashlib
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from app.db.models import Chunk, DeliverableSection, Document, Fact
from app.graph.nodes import (
    classify_and_chunk,
    compose_deliverable,
    extract_facts,
)
from app.graph.schemas import ExtractionResult, ExtractedFact
from tests.conftest import (
    make_chunk,
    make_document,
    make_fact,
    make_pile,
    make_run,
    make_state,
)


def test_duplicate_document_ingest_rejected(session, raw_dir):
    """
    Uploading the same content twice to the same pile must produce
    exactly one Document row (content_hash uniqueness).
    """
    from app.routes.ingest import _content_hash
    from app.db.models import Document

    pile = make_pile(session)
    text = "This is a sample vendor contract."
    h    = _content_hash(text)

    doc1 = Document(
        id=str(uuid.uuid4()),
        pile_id=pile.id,
        filename="contract.txt",
        format="txt",
        content_hash=h,
    )
    session.add(doc1)
    session.flush()

    # Second insert with same content_hash must fail the unique constraint
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        doc2 = Document(
            id=str(uuid.uuid4()),
            pile_id=pile.id,
            filename="contract_copy.txt",
            format="txt",
            content_hash=h,
        )
        session.add(doc2)
        session.flush()


def test_classify_and_chunk_is_idempotent(session, mock_embedder, raw_dir):
    """
    Running classify_and_chunk twice on the same document produces
    exactly the same number of chunks — no duplicates.
    """
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    pile = make_pile(session)
    doc  = make_document(session, pile, text=text)
    run  = make_run(session, pile)
    session.commit()

    (raw_dir.parent / "raw" / f"{doc.id}.txt").write_text(text, encoding="utf-8")

    state = make_state(pile, run, doc)

    classify_and_chunk(state, session)
    count_after_first = session.query(Chunk).filter(
        Chunk.document_id == doc.id
    ).count()

    classify_and_chunk(state, session)
    count_after_second = session.query(Chunk).filter(
        Chunk.document_id == doc.id
    ).count()

    assert count_after_first == count_after_second, (
        f"classify_and_chunk is not idempotent: "
        f"{count_after_first} chunks after first run, "
        f"{count_after_second} after second"
    )


def test_extract_facts_is_idempotent(session, mock_embedder, raw_dir):
    """
    Running extract_facts twice on the same chunk produces exactly
    one Fact row per claim — the natural_key unique constraint enforces this.
    """
    source = "Payment is due within 30 days of receipt."
    pile   = make_pile(session)
    doc    = make_document(session, pile, text=source)
    run    = make_run(session, pile)
    chunk  = make_chunk(session, doc, text=source)
    session.commit()

    (raw_dir.parent / "raw" / f"{doc.id}.txt").write_text(source, encoding="utf-8")

    extraction = ExtractionResult(
        facts=[
            ExtractedFact(
                claim="Payment is due within 30 days.",
                claim_slot="payment_terms",
                evidence_quote="Payment is due within 30 days",
                confidence=0.95,
            )
        ]
    )

    state = make_state(pile, run, doc)

    with patch("app.graph.nodes.call_structured", return_value=extraction):
        extract_facts(state, session)
        count_after_first = session.query(Fact).filter(
            Fact.pile_id == pile.id
        ).count()

        extract_facts(state, session)
        count_after_second = session.query(Fact).filter(
            Fact.pile_id == pile.id
        ).count()

    assert count_after_first == count_after_second, (
        f"extract_facts is not idempotent: "
        f"{count_after_first} facts after first run, "
        f"{count_after_second} after second"
    )


def test_compose_deliverable_is_byte_identical_on_rerun(session):
    """
    Running compose_deliverable twice with identical inputs produces
    sections with the same content_hash both times.
    """
    pile  = make_pile(session)
    doc   = make_document(session, pile)
    run   = make_run(session, pile)
    chunk = make_chunk(session, doc, text="Net-30 payment terms.")
    make_fact(
        session, pile, chunk,
        claim="Payment terms are Net-30.",
        claim_slot="payment_terms",
        natural_key="idempotent_fact_001",
    )
    session.commit()

    state = make_state(pile, run, doc)

    compose_deliverable(state, session)
    section_after_first = session.scalar(
        select(DeliverableSection).where(
            DeliverableSection.pile_id == pile.id,
            DeliverableSection.title   == "payment_terms",
        )
    )
    hash_first = section_after_first.content_hash

    compose_deliverable(state, session)
    session.expire_all()
    section_after_second = session.scalar(
        select(DeliverableSection).where(
            DeliverableSection.pile_id == pile.id,
            DeliverableSection.title   == "payment_terms",
        )
    )

    assert section_after_second.content_hash == hash_first, (
        "compose_deliverable must produce byte-identical output on identical input"
    )
