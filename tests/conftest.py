"""Shared pytest fixtures — everything runs fully offline."""

from __future__ import annotations

import pytest

from life_memorizer.config import Quantization, Settings
from life_memorizer.embeddings import FakeEmbedder
from life_memorizer.store import MemoryStore


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        fake_embeddings=True,
        embed_dim=128,
        db_path=tmp_path / "db",
        collection="test_moments",
        quantization=Quantization.none,
        ttl_days=30,
        fps=1.0,
    )


@pytest.fixture()
def embedder(settings) -> FakeEmbedder:
    return FakeEmbedder(dim=settings.embed_dim)


@pytest.fixture()
def store(settings) -> MemoryStore:
    s = MemoryStore(settings)
    s.ensure_collection(recreate=True)
    return s
