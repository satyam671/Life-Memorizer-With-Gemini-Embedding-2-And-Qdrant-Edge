"""Application settings, loaded from environment / .env and validated."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Quantization(str, Enum):
    """Supported on-device quantization strategies."""

    none = "none"
    scalar = "scalar"
    binary = "binary"


class Settings(BaseSettings):
    """Runtime configuration for Life Memorizer.

    All values are environment-driven (prefix ``LIFE_MEMORIZER_``) so the same
    code runs unchanged on a laptop or an edge device.
    """

    model_config = SettingsConfigDict(
        env_prefix="LIFE_MEMORIZER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Gemini / embeddings -------------------------------------------------
    # The API key uses its conventional name (no prefix) to match Google tools.
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    # Gemini Embedding 2 (multi-modal: text/image/audio/video/PDF).
    # MRL output up to 3072 dims; max input 8,192 tokens.
    embed_model: str = Field(default="gemini-embedding-2")
    embed_dim: int = Field(default=768, ge=64, le=3072)
    fake_embeddings: bool = Field(default=False)

    # --- Qdrant Edge ---------------------------------------------------------
    db_path: Path = Field(default=Path("./life_memorizer_db"))
    collection: str = Field(default="life_moments")
    quantization: Quantization = Field(default=Quantization.scalar)

    # --- Memory lifecycle ----------------------------------------------------
    ttl_days: int = Field(default=90, ge=0)
    # When True, `maybe_auto_prune()` prunes once per `auto_prune_interval_hours`.
    auto_prune: bool = Field(default=False)
    auto_prune_interval_hours: float = Field(default=24.0, gt=0)
    # When True, expired moments are summarized into a compact digest before
    # deletion (keeps storage roughly constant); when False they are dropped.
    summarize_on_prune: bool = Field(default=True)

    # --- Privacy -------------------------------------------------------------
    # When False, the media_file_path is NOT persisted in the DB, so not even a
    # reference to raw media leaves the edge container.
    store_media_path: bool = Field(default=True)

    # --- Local RAG (answer generation) ---------------------------------------
    # Backend used to turn recalled context into a spoken-style answer.
    #   gemini -> Gemma via google-genai (needs GEMINI_API_KEY)
    #   ollama -> fully local Gemma via an Ollama HTTP server
    rag_backend: str = Field(default="ollama")
    rag_model: str = Field(default="gemma2:2b")
    ollama_host: str = Field(default="http://localhost:11434")
    # Number of memories fed to the model as grounding context.
    rag_context_size: int = Field(default=5, ge=1, le=50)
    # Use a deterministic offline answer generator (no model needed).
    fake_rag: bool = Field(default=False)

    # --- Ingest --------------------------------------------------------------
    fps: float = Field(default=1.0, gt=0)

    @field_validator("db_path", mode="before")
    @classmethod
    def _expand_path(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value

    @property
    def gemini_output_dim(self) -> int:
        """Native Gemini embedding dimensionality before Matryoshka truncation."""
        return 3072

    def require_api_key(self) -> str:
        """Return the Gemini API key or raise a clear error."""
        if self.fake_embeddings:
            return ""
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Set it in .env, export it, or run with "
                "LIFE_MEMORIZER_FAKE_EMBEDDINGS=1 for the offline stub embedder."
            )
        return self.gemini_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
