from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import Pile
from app.db.session import get_session

router = APIRouter(prefix="/piles", tags=["piles"])


class RuleIn(BaseModel):
    rule_id: str
    description: str


class CreatePileRequest(BaseModel):
    name: str = Field(min_length=1)
    rules: list[RuleIn] = Field(default_factory=list)


class PileOut(BaseModel):
    id: str
    name: str
    rules: list[RuleIn]


@router.post("", response_model=PileOut, status_code=201)
def create_pile(
    body: CreatePileRequest,
    session: Session = Depends(get_session),
):
    pile = Pile(
        name=body.name,
        rules_json={
            "rules": [r.model_dump() for r in body.rules]
        },
    )

    session.add(pile)
    session.commit()
    session.refresh(pile)

    return PileOut(
        id=pile.id,
        name=pile.name,
        rules=body.rules,
    )


# GET /piles
@router.get("", response_model=list[PileOut])
def list_piles(session: Session = Depends(get_session)):
    piles = (
        session.query(Pile)
        .order_by(Pile.created_at.desc())
        .all()
    )

    return [
        PileOut(
            id=pile.id,
            name=pile.name,
            rules=[
                RuleIn(**rule)
                for rule in (pile.rules_json or {}).get("rules", [])
            ],
        )
        for pile in piles
    ]


# GET /piles/{pile_id}
@router.get("/{pile_id}", response_model=PileOut)
def get_pile(
    pile_id: str,
    session: Session = Depends(get_session),
):
    from fastapi import HTTPException

    pile = session.get(Pile, pile_id)

    if pile is None:
        raise HTTPException(404, f"Pile {pile_id} not found")

    rules = [
        RuleIn(**r)
        for r in (pile.rules_json or {}).get("rules", [])
    ]

    return PileOut(
        id=pile.id,
        name=pile.name,
        rules=rules,
    )