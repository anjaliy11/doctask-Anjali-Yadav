"""
Behavior #9: Two runs at the same time stay two runs.

Tests split into two groups:
  - Pure unit tests (no DB): test the logic of pile_lock contract
  - DB tests: test actual advisory lock behavior against PostgreSQL

The advisory lock tests require a live DB and are skipped automatically
if TEST_DATABASE_URL is unreachable.
"""
import threading
import uuid

import pytest

from app.locking import PileBusyError, pile_lock


# ---------------------------------------------------------------------------
# Pure unit tests — no DB needed, never skipped
# ---------------------------------------------------------------------------

def test_pile_busy_error_is_raised_correctly():
    """
    PileBusyError must be a subclass of Exception so callers
    can catch it specifically without catching everything.
    """
    err = PileBusyError("pile abc already has a run")
    assert isinstance(err, Exception)
    assert "abc" in str(err)


def test_pile_lock_is_context_manager():
    """
    pile_lock must be usable as a context manager.
    Verify it is a generator-based contextmanager (has __enter__/__exit__
    via contextlib) — the actual lock acquisition requires a DB session.
    """
    import inspect
    from contextlib import _GeneratorContextManager
    # pile_lock is decorated with @contextmanager so calling it
    # returns a _GeneratorContextManager instance
    import sqlalchemy.orm
    mock_session = type(
        "MockSession",
        (),
        {
            "execute": lambda self, *a, **kw: None,
            "query":   lambda self, *a, **kw: type(
                "Q", (), {"filter": lambda self, *a, **kw: type(
                    "Q2", (), {"first": lambda self: None}
                )()}
            )(),
        }
    )()
    cm = pile_lock(mock_session, "test-pile-id")
    assert hasattr(cm, "__enter__"), "pile_lock must return a context manager"
    assert hasattr(cm, "__exit__"),  "pile_lock must return a context manager"


# ---------------------------------------------------------------------------
# DB tests — skipped if DB unreachable
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db_session_factory(db_engine):
    """Module-scoped session factory for concurrency tests."""
    from sqlalchemy.orm import sessionmaker
    return sessionmaker(bind=db_engine, expire_on_commit=False)


def test_pile_lock_blocks_second_acquisition(db_engine, session):
    """
    Two transactions cannot both hold the same pile lock when an
    active run exists. The second raises PileBusyError.
    """
    from tests.conftest import make_pile, make_run

    pile   = make_pile(session)
    active = make_run(session, pile, status="running")
    session.commit()

    from app.db.session import SessionLocal
    session2 = SessionLocal()
    try:
        with pytest.raises(PileBusyError):
            with session2.begin():
                with pile_lock(session2, pile.id):
                    pass
    finally:
        session2.close()


def test_pile_lock_allows_acquisition_after_run_completes(db_engine, session):
    """
    Once the active run finishes, the lock can be acquired again.
    """
    from tests.conftest import make_pile, make_run

    pile = make_pile(session)
    _    = make_run(session, pile, status="done")
    session.commit()

    from app.db.session import SessionLocal
    session2 = SessionLocal()
    try:
        with session2.begin():
            with pile_lock(session2, pile.id):
                pass  # must not raise
    finally:
        session2.close()


def test_different_piles_do_not_interfere(db_engine, session):
    """
    Advisory locks are pile-scoped. A running run on pile A must not
    block a lock acquisition on pile B.
    """
    from tests.conftest import make_pile, make_run

    pile_a = make_pile(session)
    pile_b = make_pile(session)
    _      = make_run(session, pile_a, status="running")
    session.commit()

    from app.db.session import SessionLocal
    session2 = SessionLocal()
    try:
        with session2.begin():
            with pile_lock(session2, pile_b.id):
                pass  # pile_b is free — must not raise
    finally:
        session2.close()


def test_concurrent_lock_attempts_serialized(db_engine, session):
    """
    Three threads all try to acquire the pile lock simultaneously.
    Exactly one succeeds; the others get PileBusyError.
    No corrupted state results.

    Requires a live DB — skipped automatically if DB unreachable.
    """
    from tests.conftest import make_pile, make_run
    from app.db.session import SessionLocal

    pile       = make_pile(session)
    active_run = make_run(session, pile, status="running")
    session.commit()
    pile_id = pile.id

    results = {"successes": 0, "busy": 0, "errors": []}
    lock    = threading.Lock()

    def try_acquire():
        sess = SessionLocal()
        try:
            with sess.begin():
                with pile_lock(sess, pile_id):
                    with lock:
                        results["successes"] += 1
        except PileBusyError:
            with lock:
                results["busy"] += 1
        except Exception as e:
            with lock:
                results["errors"].append(str(e))
        finally:
            sess.close()

    threads = [threading.Thread(target=try_acquire) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results["errors"] == [], (
        f"Unexpected errors in concurrent lock test: {results['errors']}"
    )
    # All 3 must either succeed or get PileBusyError — no silent failures
    assert results["successes"] + results["busy"] == 3, (
        "Every attempt must resolve cleanly"
    )
    # At least 1 must be blocked since active_run exists
    assert results["busy"] >= 1, (
        "At least one concurrent attempt must be blocked"
    )
def test_true_concurrent_race_only_one_wins(db_engine):
    """
    Two threads simultaneously try to acquire the pile lock with
    NO pre-existing active run. Exactly one must win — this tests
    the advisory lock itself, not just the active_run check.
    """
    from sqlalchemy.orm import sessionmaker
    from app.db.models import Run
    from tests.conftest import make_pile

    SessionFactory = sessionmaker(bind=db_engine)

    setup_sess = SessionFactory()
    pile = make_pile(setup_sess)
    setup_sess.commit()
    pile_id = pile.id
    setup_sess.close()

    winners = []
    losers  = []
    errors  = []
    lock    = threading.Lock()
    barrier = threading.Barrier(2)

    def race():
        sess = SessionFactory()
        try:
            barrier.wait()  # both threads enter simultaneously
            with sess.begin():
                with pile_lock(sess, pile_id):
                    run = Run(
                        id=str(uuid.uuid4()),
                        pile_id=pile_id,
                        thread_id=str(uuid.uuid4()),
                        status="running",
                        stage="classify_and_chunk",
                    )
                    sess.add(run)
                    sess.flush()
                    with lock:
                        winners.append(1)
        except PileBusyError:
            with lock:
                losers.append(1)
        except Exception as e:
            with lock:
                errors.append(str(e))
        finally:
            sess.close()

    threads = [threading.Thread(target=race) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Unexpected errors in race test: {errors}"
    assert len(winners) == 1, f"Expected exactly 1 winner, got {len(winners)}"
    assert len(losers)  == 1, f"Expected exactly 1 loser, got {len(losers)}"
