"""
SQLAlchemy models. A few constraints here are load-bearing, not decorative:

- facts.natural_key is UNIQUE per pile → re-running extract() after a crash
  (LangGraph re-executes node code before the last interrupt on resume) is
  safe: the second insert of the same fact is a no-op via ON CONFLICT.
- section_sources is the dependency graph that makes incremental updates
  possible: a new document only invalidates sections that actually cite it.
- runs.status + a partial unique index on (pile_id) where status='running'
  is the second half of the concurrency story — the advisory lock (see
  app/locking.py) handles the "same instant" race, this catches the
  "overlapping but not exactly simultaneous" case.
"""
import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Pile(Base):
    __tablename__ = "piles"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    rules_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    pile_id: Mapped[str] = mapped_column(ForeignKey("piles.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    doc_type: Mapped[str] = mapped_column(String, default="unclassified")
    format: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        # Uploading byte-identical content twice must not double-process it.
        UniqueConstraint("pile_id", "content_hash", name="uq_document_pile_hash"),
    )


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list] = mapped_column(Vector(384), nullable=True)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("document_id", "char_start", "char_end", name="uq_chunk_span"),
    )


class Fact(Base):
    __tablename__ = "facts"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    pile_id: Mapped[str] = mapped_column(ForeignKey("piles.id"), nullable=False)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"), nullable=False)
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    claim_slot: Mapped[str] = mapped_column(String, nullable=False)  # e.g. "payment_terms"
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    grounded: Mapped[bool] = mapped_column(Boolean, default=True)
    suspicious: Mapped[bool] = mapped_column(Boolean, default=False)
    natural_key: Mapped[str] = mapped_column(String, nullable=False)
    extracted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("pile_id", "natural_key", name="uq_fact_natural_key"),
    )


class DeliverableSection(Base):
    __tablename__ = "deliverable_sections"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    pile_id: Mapped[str] = mapped_column(ForeignKey("piles.id"), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    rendered_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class SectionSource(Base):
    __tablename__ = "section_sources"
    section_id: Mapped[str] = mapped_column(ForeignKey("deliverable_sections.id"), primary_key=True)
    fact_id: Mapped[str] = mapped_column(ForeignKey("facts.id"), primary_key=True)


class Finding(Base):
    __tablename__ = "findings"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    pile_id: Mapped[str] = mapped_column(ForeignKey("piles.id"), nullable=False)
    rule_id: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    source_chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"), nullable=True)
    claim_slot: Mapped[str] = mapped_column(String, nullable=True)  # set for rule_id="contradiction"
    severity: Mapped[str] = mapped_column(String, default="info")  # info | warning | critical
    status: Mapped[str] = mapped_column(String, default="pending")  # pending | approved | rejected
    natural_key: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("pile_id", "natural_key", name="uq_finding_natural_key"),
    )


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    pile_id: Mapped[str] = mapped_column(ForeignKey("piles.id"), nullable=False)
    thread_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    stage: Mapped[str] = mapped_column(String, default="queued")
    status: Mapped[str] = mapped_column(String, default="running")  # running | paused | done | failed
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    # NOTE: "only one running run per pile" is NOT enforced by a DB constraint
    # here — a partial unique index on status='running' is the textbook way,
    # but that requires a raw Alembic migration (SQLAlchemy's declarative
    # postgresql_where doesn't reliably bind a mapped_column at class-body
    # eval time across versions, and shipping code that MIGHT silently no-op
    # is worse than not shipping it). It's enforced instead in
    # app/locking.py::pile_lock(), inside the same advisory-locked
    # transaction — see that file for why this is still race-safe.


class RunCost(Base):
    __tablename__ = "run_costs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0 on free tiers, kept for when that changes
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Approval(Base):
    __tablename__ = "approvals"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    item_type: Mapped[str] = mapped_column(String, nullable=False)  # "finding"
    item_id: Mapped[str] = mapped_column(String, nullable=False)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending | approved | rejected
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("item_type", "item_id", "run_id", name="uq_approval_item_per_run"),
    )


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    pile_id: Mapped[str] = mapped_column(ForeignKey("piles.id"), nullable=False)
    event: Mapped[str] = mapped_column(String, nullable=False)
    source_document_id: Mapped[str] = mapped_column(String, nullable=True)
    diff: Mapped[dict] = mapped_column(JSON, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
