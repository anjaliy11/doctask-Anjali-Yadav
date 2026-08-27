"""
Pile-scoped concurrency control.

The advisory lock serializes concurrent attempts for the same pile.
After acquiring that lock, the current transaction checks whether another
run is already active.

A run is considered active when its status is "running".

The check happens after the advisory lock is acquired so two simultaneous
callers cannot both observe the pile as free and both proceed.
"""

from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models import Run


class PileBusyError(Exception):
    """Raised when a run is already in progress on this pile."""


@contextmanager
def pile_lock(
    session: Session,
    pile_id: str,
    *,
    allow_concurrent_run: bool = False,
):
    """
    Acquire the transaction-scoped PostgreSQL advisory lock for a pile.

    For normal callers, a currently running run makes the pile busy and
    raises PileBusyError.

    `allow_concurrent_run=True` bypasses the active-run check but still
    serializes access through the advisory lock.
    """

    # Serialize all lock attempts for this pile within PostgreSQL.
    session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:pid))"),
        {"pid": pile_id},
    )

    if not allow_concurrent_run:
        active = (
            session.query(Run)
            .filter(
                Run.pile_id == pile_id,
                Run.status == "running",
            )
            .first()
        )

        if active is not None:
            raise PileBusyError(
                f"Pile {pile_id} already has run {active.id} in progress "
                f"(stage={active.stage}). Wait for it to finish or pause "
                f"before starting another run."
            )

    try:
        yield
    finally:
        # pg_advisory_xact_lock is transaction-scoped.
        # The caller's transaction commit/rollback releases it.
        pass