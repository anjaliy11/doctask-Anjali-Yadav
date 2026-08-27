# backend/tests/conftest.py
"""
Test fixtures.

Design principles:
  - No live DB required for unit tests: all DB-touching tests use
    a real PostgreSQL instance spun up by pytest-docker or a local
    test DB (TEST_DATABASE_URL env var). If neither is available,
    DB tests are skipped — never silently pass with a mock.
  - No live LLM keys required: the LLM is monkeypatched at the
    call_structured() boundary, not at the HTTP level. This means
    we test real node logic with controlled outputs.
  - No live embedding model: _get_embedder() is monkeypatched to
    return a fixed-dimension zero vector. Real similarity search
    is not what these tests exercise.
  - Every test that writes to the DB uses a transaction that is
    rolled back in teardown — tests are fully isolated.
"""
import os
import uuid
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, Chunk, Document, Fact, Pile, Run
from app.graph.schemas import ExtractionResult
from app.graph.state import PileState


# ---------------------------------------------------------------------------
# Fake embedder — no model download during tests
# ---------------------------------------------------------------------------

class _FakeEmbedder:
    def encode(self, text):
        import numpy as np
        return np.zeros(384, dtype="float32")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every test gets this by default — no test can accidentally make a
    real LLM or embedding call. Tests that specifically need different
    canned behavior override call_structured locally with their own
    monkeypatch, which takes precedence within that test's scope."""
    monkeypatch.setattr("app.graph.nodes._get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(
        "app.graph.nodes.call_structured",
        lambda **kw: ExtractionResult(facts=[]),
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

TEST_DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/doctask_test",
)


def _db_is_reachable() -> bool:
    """
    Quick TCP-level check with a 2-second timeout.
    Avoids hanging the entire test suite when Postgres is not running.
    Does NOT use SQLAlchemy — that has its own 30-second timeout.
    """
    import socket
    from urllib.parse import urlparse
    try:
        u = urlparse(TEST_DB_URL)
        host = u.hostname or "localhost"
        port = u.port or 5432
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def db_engine():
    """
    Session-scoped engine. Creates the test DB schema once per
    pytest session, tears it down at the end.
    Skips all DB tests if the DB is unreachable — fast TCP check
    first so we never hang on a dead connection.
    """
    if not _db_is_reachable():
        pytest.skip(
            f"PostgreSQL not reachable at {TEST_DB_URL} — "
            f"start Postgres or run: docker compose up db -d"
        )

    try:
        engine = create_engine(
            TEST_DB_URL,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        Base.metadata.create_all(engine)
        yield engine
        Base.metadata.drop_all(engine)
        engine.dispose()
    except Exception as e:
        pytest.skip(f"Test DB setup failed ({TEST_DB_URL}): {e}")


@pytest.fixture()
def session(db_engine) -> Generator[Session, None, None]:
    """
    Function-scoped DB session.

    Do not wrap the entire test in one uncommitted outer transaction.

    Several integration tests intentionally create data with this fixture and
    then call session.commit(). Application code under test opens independent
    sessions through app.db.session.SessionLocal(), so committed fixture data
    must be visible to those separate database connections.

    Cleanup is handled explicitly after each test instead.
    """
    connection = db_engine.connect()
    TestSession = sessionmaker(bind=connection, expire_on_commit=False)
    sess = TestSession()

    try:
        yield sess
    finally:
        sess.close()

        # Remove data created by this test. CASCADE also clears rows that
        # depend on piles/documents/runs while preserving the database schema.
        cleanup_connection = db_engine.connect()
        try:
            with cleanup_connection.begin():
                cleanup_connection.execute(
                    text(
                        """
                        TRUNCATE TABLE
                            approvals,
                            run_costs,
                            runs,
                            findings,
                            section_sources,
                            deliverable_sections,
                            facts,
                            chunks,
                            documents,
                            piles
                        CASCADE
                        """
                    )
                )
        finally:
            cleanup_connection.close()

        connection.close()


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

def make_pile(session: Session, *, rules: list[dict] | None = None) -> Pile:
    pile = Pile(
        id=str(uuid.uuid4()),
        name="test-pile",
        rules_json={"rules": rules or []},
    )
    session.add(pile)
    session.flush()
    return pile


def make_document(session: Session, pile: Pile, *, text: str = "Sample text.") -> Document:
    import hashlib
    doc = Document(
        id=str(uuid.uuid4()),
        pile_id=pile.id,
        filename="test.txt",
        format="txt",
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    session.add(doc)
    session.flush()
    return doc


def make_chunk(
    session: Session,
    doc: Document,
    *,
    text: str = "Payment is due within 30 days.",
    char_start: int = 0,
) -> Chunk:
    chunk = Chunk(
        id=str(uuid.uuid4()),
        document_id=doc.id,
        text=text,
        embedding=[0.0] * 384,
        char_start=char_start,
        char_end=char_start + len(text),
    )
    session.add(chunk)
    session.flush()
    return chunk


def make_fact(
    session: Session,
    pile: Pile,
    chunk: Chunk,
    *,
    claim: str = "Payment is due within 30 days.",
    claim_slot: str = "payment_terms",
    grounded: bool = True,
    suspicious: bool = False,
    natural_key: str | None = None,
) -> Fact:
    import hashlib
    nk = natural_key or hashlib.sha256(
        f"{chunk.id}||{claim_slot}||{claim}".encode()
    ).hexdigest()[:32]
    fact = Fact(
        id=str(uuid.uuid4()),
        pile_id=pile.id,
        chunk_id=chunk.id,
        claim=claim,
        claim_slot=claim_slot,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        confidence=0.95,
        grounded=grounded,
        suspicious=suspicious,
        natural_key=nk,
    )
    session.add(fact)
    session.flush()
    return fact


def make_run(
    session: Session,
    pile: Pile,
    *,
    status: str = "running",
    stage: str = "classify_and_chunk",
) -> Run:
    run_id = str(uuid.uuid4())
    run = Run(
        id=run_id,
        pile_id=pile.id,
        thread_id=run_id,
        status=status,
        stage=stage,
    )
    session.add(run)
    session.flush()
    return run


def make_state(pile: Pile, run: Run, doc: Document) -> PileState:
    return {
        "pile_id": pile.id,
        "run_id": run.id,
        "document_ids": [doc.id],
        "stage": "new",
    }


# ---------------------------------------------------------------------------
# LLM mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm():
    """
    Patches call_structured() to return controlled outputs.
    Tests set mock_llm.return_value before calling the node under test.
    """
    with patch("app.graph.nodes.call_structured") as mock:
        yield mock


@pytest.fixture()
def mock_embedder():
    """
    Patches _get_embedder() so no model weights are downloaded.
    Returns a fixed 384-dim zero vector for any input.
    """
    mock_model = MagicMock()
    mock_model.encode.return_value = [0.0] * 384
    with patch("app.graph.nodes._get_embedder", return_value=mock_model):
        yield mock_model


# ---------------------------------------------------------------------------
# Raw file fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_dir(tmp_path, monkeypatch):
    """
    Points INBOX_DIR to a temp directory so raw file writes
    don't touch the real sample_data/ directory during tests.

    *** BUG FIX ***
    This fixture previously had no return/yield statement, so every
    test that requested `raw_dir` received None. Every test that then
    did `raw_dir.parent / "raw" / f"{doc.id}.txt"` — the pattern used
    throughout test_behaviour_5_no_bluffing.py, test_behaviour_8_
    injection.py, and test_idempotency.py — crashed with
    AttributeError: 'NoneType' object has no attribute 'parent'
    before ever reaching its assertions. test_behaviour_2_resume.py
    was unaffected only because it happens to reconstruct the raw
    path itself from get_settings().inbox_dir instead of using the
    fixture's return value.

    Fix: return `inbox`, not None. Callers do `raw_dir.parent / "raw"`
    to get back to the raw/ directory this fixture already created
    (raw_dir.parent == tmp_path, and tmp_path / "raw" == `raw` above)
    — so the fixture must hand back the INBOX path, matching what
    INBOX_DIR was actually set to.
    """
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()

    monkeypatch.setenv("INBOX_DIR", str(inbox))

    from app import config as cfg_module
    cfg_module.get_settings.cache_clear()

    return inbox
