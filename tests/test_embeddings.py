"""Tests for the embedding layer."""

from __future__ import annotations

import math

import numpy as np

from life_memorizer.embeddings import FakeEmbedder, matryoshka_truncate


def test_fake_embedder_is_deterministic():
    e = FakeEmbedder(dim=64)
    a = e.embed_text("where did I leave my keys?")
    b = e.embed_text("where did I leave my keys?")
    assert a == b
    assert len(a) == 64


def test_fake_embedder_differs_for_different_text():
    e = FakeEmbedder(dim=64)
    assert e.embed_text("keys") != e.embed_text("wallet")


def test_fake_embedder_vectors_are_normalized():
    e = FakeEmbedder(dim=128)
    vec = np.asarray(e.embed_text("normalize me"))
    assert math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=1e-5)


def test_matryoshka_truncate_reduces_dim_and_normalizes():
    raw = np.arange(3072, dtype=np.float32) + 1.0
    out = matryoshka_truncate(raw, 768)
    assert out.shape[0] == 768
    assert math.isclose(float(np.linalg.norm(out)), 1.0, rel_tol=1e-5)


def test_matryoshka_truncate_pads_when_too_short():
    raw = np.ones(100, dtype=np.float32)
    out = matryoshka_truncate(raw, 256)
    assert out.shape[0] == 256


def _cosine(a, b):
    av, bv = np.asarray(a), np.asarray(b)
    return float(np.dot(av, bv))  # vectors are L2-normalized


def test_shared_words_increase_similarity():
    e = FakeEmbedder(dim=512)
    keys_scene = e.embed_text("brass house keys on the hallway table")
    keys_query = e.embed_text("where did I leave my keys?")
    menu = e.embed_text("flat white cappuccino cold brew avocado toast")
    # The query overlaps the keys scene ('keys') but not the cafe menu.
    assert _cosine(keys_query, keys_scene) > _cosine(keys_query, menu)


def test_query_ranks_correct_scene_first():
    e = FakeEmbedder(dim=512)
    scenes = {
        "keys": "a set of brass house keys lying on the wooden hallway table",
        "wallet": "a brown leather wallet left on the kitchen counter",
        "menu": "the chalkboard menu above the espresso bar at the cafe",
    }
    query = e.embed_text("where did I leave my keys?")
    ranked = sorted(
        scenes.items(),
        key=lambda kv: _cosine(query, e.embed_text(kv[1])),
        reverse=True,
    )
    assert ranked[0][0] == "keys"
