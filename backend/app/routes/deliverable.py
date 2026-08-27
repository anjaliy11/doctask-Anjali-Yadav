"""
Deliverable routes — read the rendered output of a pile run.

The deliverable is NOT generated here; it is composed by the
compose_deliverable node in the graph and stored in deliverable_sections.

These endpoints are purely read-only: they expose what the graph produced.

Two endpoints:
  GET /piles/{pile_id}/deliverable        — structured JSON (for the UI)
  GET /piles/{pile_id}/deliverable/export — flat markdown (for download)
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import DeliverableSection, Finding, Pile
from app.db.session import get_session


router = APIRouter(prefix="/piles", tags=["deliverable"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class SectionOut(BaseModel):
    id: str
    title: str
    rendered_text: str
    content_hash: str
    updated_at: datetime


class DeliverableOut(BaseModel):
    pile_id: str
    pile_name: str
    sections: list[SectionOut]
    has_unresolved_conflicts: bool
    open_findings_count: int


# ---------------------------------------------------------------------------
# GET /piles/{pile_id}/deliverable
# ---------------------------------------------------------------------------

@router.get("/{pile_id}/deliverable", response_model=DeliverableOut)
def get_deliverable(
    pile_id: str,
    session: Session = Depends(get_session),
):
    """
    Return the structured deliverable for a pile.

    Important behavior:
      - Unknown pile -> 404
      - Existing pile with no composed sections yet -> 200 with sections=[]
      - Existing pile with sections -> 200 with those sections

    This endpoint is intentionally read-only. It never generates the
    deliverable; compose_deliverable in the graph is responsible for that.
    """

    pile = session.get(Pile, pile_id)

    if pile is None:
        raise HTTPException(
            status_code=404,
            detail=f"Pile {pile_id} not found",
        )

    sections = session.scalars(
        select(DeliverableSection)
        .where(DeliverableSection.pile_id == pile_id)
        .order_by(DeliverableSection.title)
    ).all()

    # A contradiction remains unresolved unless it was explicitly rejected.
    # "approved" means the human confirmed that the contradiction is real,
    # so it should still remain visible.
    unresolved_conflicts_count = session.scalar(
        select(func.count())
        .select_from(Finding)
        .where(
            Finding.pile_id == pile_id,
            Finding.rule_id == "contradiction",
            Finding.status != "rejected",
        )
    ) or 0

    # Pending findings are still awaiting a decision.
    open_findings_count = session.scalar(
        select(func.count())
        .select_from(Finding)
        .where(
            Finding.pile_id == pile_id,
            Finding.status == "pending",
        )
    ) or 0

    return DeliverableOut(
        pile_id=pile.id,
        pile_name=pile.name,
        sections=[
            SectionOut(
                id=section.id,
                title=section.title,
                rendered_text=section.rendered_text,
                content_hash=section.content_hash,
                updated_at=section.updated_at,
            )
            for section in sections
        ],
        has_unresolved_conflicts=unresolved_conflicts_count > 0,
        open_findings_count=open_findings_count,
    )


# ---------------------------------------------------------------------------
# GET /piles/{pile_id}/deliverable/export
# ---------------------------------------------------------------------------

@router.get(
    "/{pile_id}/deliverable/export",
    response_class=PlainTextResponse,
)
def export_deliverable(
    pile_id: str,
    session: Session = Depends(get_session),
):
    """
    Return the deliverable as a single Markdown document.

    Behavior:
      - Unknown pile -> 404
      - Existing pile but no deliverable sections -> 404
      - Existing composed deliverable -> Markdown attachment
    """

    pile = session.get(Pile, pile_id)

    if pile is None:
        raise HTTPException(
            status_code=404,
            detail=f"Pile {pile_id} not found",
        )

    sections = session.scalars(
        select(DeliverableSection)
        .where(DeliverableSection.pile_id == pile_id)
        .order_by(DeliverableSection.title)
    ).all()

    # Unlike the JSON endpoint, export requires an actual composed
    # deliverable to exist.
    if not sections:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No deliverable exists yet for pile {pile_id}. "
                "Run a document ingest first."
            ),
        )

    lines = [
        f"# Document Pile Report: {pile.name}",
        "",
        f"*Generated from pile `{pile.id}`*",
        "",
        "---",
        "",
    ]

    for section in sections:
        lines.append(section.rendered_text)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Append findings that are still awaiting human review.
    open_findings = session.scalars(
        select(Finding)
        .where(
            Finding.pile_id == pile_id,
            Finding.status == "pending",
        )
        .order_by(Finding.id)
    ).all()

    if open_findings:
        lines.extend([
            "## Open Findings (Pending Review)",
            "",
        ])

        for finding in open_findings:
            lines.append(
                f"- **[{finding.severity.upper()}]** "
                f"`{finding.rule_id}`: {finding.description}"
            )

        lines.append("")

    return PlainTextResponse(
        content="\n".join(lines),
        headers={
            "Content-Disposition": (
                f'attachment; filename="deliverable-{pile.id[:8]}.md"'
            )
        },
    )