"""Tests for Step 5: edge optimization (TTL + summarization) and privacy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from life_memorizer.config import Quantization, Settings
from life_memorizer.embeddings import FakeEmbedder
from life_memorizer.models import VECTOR_OCR_LOG, VECTOR_VIDEO_FRAME, Moment
from life_memorizer.store import MemoryStore, _extractive_summary


def _old_moment(embedder, text, location="Home", days_old=400):
    m = Moment(
        location_context=location,
        timestamp=datetime.now(timezone.utc) - timedelta(days=days_old),
        ocr_text=text,
    )
    m.vectors[VECTOR_OCR_LOG] = embedder.embed_text(text)
    m.vectors[VECTOR_VIDEO_FRAME] = embedder.embed_text(text)
    return m


def test_extractive_summary_dedupes_and_trims():
    out = _extractive_summary(["keys on table", "keys on table", "wallet on counter"])
    assert "keys on table" in out
    assert "wallet on counter" in out
    assert out.count("keys on table") == 1


def test_extractive_summary_respects_budget():
    out = _extractive_summary(["x" * 500], max_chars=50)
    assert len(out) <= 50


def test_summarize_expired_collapses_per_location(store, embedder):
    store.upsert_moments(
        [
            _old_moment(embedder, "keys on the hallway table", location="Home"),
            _old_moment(embedder, "wallet on the kitchen counter", location="Home"),
            _old_moment(embedder, "cafe menu board", location="Cafe"),
        ]
    )
    assert store.count() == 3
    removed = store.summarize_expired(ttl_days=90)
    # 3 originals -> 2 digests (Home, Cafe) => net 1 removed.
    assert removed == 1
    assert store.count() == 2


def test_summary_points_are_flagged(store, embedder):
    store.upsert_moments([_old_moment(embedder, "keys on table", location="Home")])
    store.summarize_expired(ttl_days=90)
    records = store._scroll_all()
    assert records
    assert all((r.payload or {}).get("is_summary") for r in records)


def test_summary_is_still_searchable(store, embedder, settings):
    from life_memorizer.recall import RecallEngine

    store.upsert_moments(
        [
            _old_moment(embedder, "a set of brass house keys on the hallway table", location="Home"),
            _old_moment(embedder, "a brown leather wallet on the counter", location="Home"),
        ]
    )
    store.summarize_expired(ttl_days=90)
    engine = RecallEngine(settings, embedder, store)
    hits = engine.recall("keys hallway table", limit=1)
    assert hits
    assert hits[0].moment.is_summary


def test_ttl_zero_disables_summarization(store, embedder):
    store.upsert_moments([_old_moment(embedder, "keys", location="Home")])
    assert store.summarize_expired(ttl_days=0) == 0
    assert store.count() == 1


def test_store_media_path_privacy_flag(tmp_path):
    settings = Settings(
        fake_embeddings=True,
        embed_dim=64,
        db_path=tmp_path / "db",
        collection="priv_moments",
        quantization=Quantization.none,
        store_media_path=False,
    )
    embedder = FakeEmbedder(dim=64)
    store = MemoryStore(settings)
    store.ensure_collection(recreate=True)
    m = Moment(location_context="Home", media_file_path="media_cache/home/secret.jpg")
    m.vectors[VECTOR_OCR_LOG] = embedder.embed_text("note")
    store.upsert_moments([m])
    records = store._scroll_all()
    assert records
    assert all((r.payload or {}).get("media_file_path") is None for r in records)


def test_auto_prune_cadence_guard(store, embedder, tmp_path):
    store.settings.auto_prune = True
    store.settings.ttl_days = 90
    store.settings.summarize_on_prune = False
    store.upsert_moments([_old_moment(embedder, "old note", location="Home")])
    marker = tmp_path / ".last_prune"
    first = store.maybe_auto_prune(state_path=marker)
    assert first == 1
    # Immediately calling again is within the interval -> skipped.
    store.upsert_moments([_old_moment(embedder, "another old note", location="Home")])
    second = store.maybe_auto_prune(state_path=marker)
    assert second == 0
