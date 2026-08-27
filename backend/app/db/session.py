import os
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


def _database_url() -> str:
    """
    Use the test database when running under pytest.

    Application code creates fresh sessions through SessionLocal(), including
    graph nodes and concurrency checks. Those sessions must point at the same
    database used by the pytest fixtures.
    """
    if "pytest" in sys.modules:
        return os.getenv(
            "TEST_DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/doctask_test",
        )

    return get_settings().database_url


DATABASE_URL = _database_url()

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


def get_session() -> Session:
    """FastAPI dependency — one session per request, always closed."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()