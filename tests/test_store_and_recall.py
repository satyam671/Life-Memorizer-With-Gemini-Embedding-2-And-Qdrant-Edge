"""End-to-end offline tests for store, ingest and recall.

These use the realistic mock dataset (life_memorizer.mock_data) so the assertions
mirror the actual tutorial recall scenarios.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from life_memorizer.ingest import Ingestor
from life_memorizer.mock_data import build_mock_moments, seed_store
from life_memorizer.models import VECTOR_OCR_LOG, Modality, Moment
from life_memorizer.recall import RecallEngine


def _moment_text(embedder, text, location="Home", ts=None):
    """Tiny helper for tests that only need a single text-backed moment."""
    moment = Moment(
        location_context=location,
        timestamp=ts or datetime.now(timezone.utc),
        ocr_text=text,
    )
    moment.vectors[VECTOR_OCR_LOG] = embedder.embed_text(text)
    return moment


def test_seed_loads_full_session(store, embedder):
    written = seed_store(embedder, store)
    assert written >= 7
    assert store.count() == written


def test_scenario_a_visual_keys(store, embedder, settings):
    """Scenario A: 'Where did I leave my keys?' -> the hallway-table keys frame."""
    seed_store(embedder, store)
    engine = RecallEngine(settings, embedder, store)
    hits = engine.visual_search(
        "a set of brass house keys lying on the wooden hallway table next to a "
        "blue ceramic bowl and a stack of unopened mail",
        limit=1,
    )
    assert hits
    assert "keys" in (hits[0].moment.media_file_path or "")
    assert hits[0].moment.location_context == "Home"


def test_scenario_b_audio_sarah(store, embedder, settings):
    """Scenario B: 'What did Sarah say to buy?' -> her oat-milk/eggs request."""
    seed_store(embedder, store)
    engine = RecallEngine(settings, embedder, store)
    hits = engine.audio_recall(
        "can you grab oat milk and a dozen eggs on the way back? we're out of "
        "both, and maybe some fresh basil if they have it",
        limit=1,
    )
    assert hits
    assert hits[0].moment.transcript is not None
    assert "oat milk" in hits[0].moment.transcript


def test_scenario_c_hybrid_cafe_menu(store, embedder, settings):
    """Scenario C: hybrid search filtered to Cafe surfaces the menu board."""
    seed_store(embedder, store)
    engine = RecallEngine(settings, embedder, store)
    hits = engine.recall(
        "MAPLE & CO\nFlat White  4.20\nCappuccino  4.00\nCold Brew  4.50\n"
        "Avocado Toast  8.50\nOat milk +0.60",
        hybrid=True,
        limit=3,
        location_context="Cafe",
    )
    assert hits
    assert all(h.moment.location_context == "Cafe" for h in hits)
    assert any("menu" in (h.moment.media_file_path or "") for h in hits)


def test_location_filter_restricts_results(store, embedder, settings):
    seed_store(embedder, store)
    engine = RecallEngine(settings, embedder, store)
    hits = engine.recall("coffee order", limit=10, location_context="Cafe")
    assert hits
    assert all(h.moment.location_context == "Cafe" for h in hits)


def test_prune_removes_old_moments(store, embedder):
    old = _moment_text(
        embedder,
        "LEVEL 3 - SECTION B - SPACE 142",
        location="Work",
        ts=datetime.now(timezone.utc) - timedelta(days=400),
    )
    fresh = _moment_text(embedder, "SPRINT 7 sprint plan", location="Work")
    store.upsert_moments([old, fresh])
    removed = store.prune_expired(ttl_days=90)
    assert removed == 1
    assert store.count() == 1


def test_manual_ingest_text_only(store, embedder, settings):
    ingestor = Ingestor(settings, embedder, store)
    moment = ingestor.ingest_manual(
        location="Work", text_note="LEVEL 3 - SECTION B - SPACE 142"
    )
    assert VECTOR_OCR_LOG in moment.vectors
    engine = RecallEngine(settings, embedder, store)
    hits = engine.recall("LEVEL 3 - SECTION B - SPACE 142", location_context="Work")
    assert hits and hits[0].moment.location_context == "Work"


def test_build_mock_moments_have_named_vectors(embedder):
    moments = build_mock_moments(embedder)
    assert moments
    # Every moment must carry at least one named vector.
    assert all(m.vectors for m in moments)


def test_mock_data_filtered_when_fake_embeddings_false(store, embedder, settings):
    seed_store(embedder, store)
    ingestor = Ingestor(settings, embedder, store)
    ingestor.ingest_manual(
        location="Home", text_note="My actual keys on the counter"
    )

    # When fake_embeddings=True (default in test config), mock moments are retrieved.
    engine = RecallEngine(settings, embedder, store)
    hits = engine.recall("keys", limit=10)
    assert any(h.moment.source_clip == "mock://afternoon_session" for h in hits)

    # When fake_embeddings=False, mock moments are filtered out.
    settings.fake_embeddings = False
    engine_real = RecallEngine(settings, embedder, store)
    hits_real = engine_real.recall("keys", limit=10)
    assert all(h.moment.source_clip != "mock://afternoon_session" for h in hits_real)

