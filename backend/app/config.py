"""
Central configuration. Every value comes from the environment (.env in dev,
real env vars in any hosted deployment) — nothing here is a secret, and this
file is safe to commit.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/doctask"

    gemini_api_key: str = ""
    groq_api_key: str = ""

    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "llama-3.3-70b-versatile"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Every operation that costs money gets a hard ceiling so a bug can't
    # spin the free tier into the ground unattended.
    max_llm_calls_per_run: int = 60
    extraction_batch_size: int = 8

    inbox_dir: str = "./sample_data/inbox"

    def require_llm_keys(self) -> None:
        if not self.gemini_api_key and not self.groq_api_key:
            raise RuntimeError(
                "Neither GEMINI_API_KEY nor GROQ_API_KEY is set. At least one "
                "free-tier provider is required — see .env.example."
            )

    @property
    def psycopg_conninfo(self) -> str:
        """SQLAlchemy's database_url uses the `postgresql+psycopg://` dialect
        prefix; raw psycopg (which LangGraph's PostgresSaver uses directly,
        not through SQLAlchemy) chokes on the `+psycopg` part. Strip it."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://")


@lru_cache
def get_settings() -> Settings:
    return Settings()
