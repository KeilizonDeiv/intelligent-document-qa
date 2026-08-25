from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"
    anthropic_rewrite_model: str = "claude-haiku-4-5"

    upload_dir: Path = BACKEND_DIR / "uploads"
    vector_db_path: Path = BACKEND_DIR / "vector_db"
    max_file_size_mb: int = 10

    chunk_size: int = 1000
    chunk_overlap: int = 200

    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_multiplier: int = 3

    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    log_level: str = "INFO"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
