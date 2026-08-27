"""
Creates all tables via SQLAlchemy metadata. Fine for this project's
timeline; swap for Alembic migrations before this ever needs a second
environment (staging/prod) where schema changes must be versioned.
Run: python init_db.py
"""
from sqlalchemy import text

from app.db.models import Base
from app.db.session import engine

if __name__ == "__main__":
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    print("Schema created.")