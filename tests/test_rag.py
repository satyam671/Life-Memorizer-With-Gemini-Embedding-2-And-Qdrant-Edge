"""Offline tests for the local RAG layer."""

from __future__ import annotations

from life_memorizer.mock_data import seed_store
from life_memorizer.models import Moment, RecallHit
from life_memorizer.rag import (
    LocalRAG,
    StubGenerator,
    build_prompt,
    build_context_block,
)


def _hit(text, location="Home", score=0.9):
    return RecallHit(
        moment=Moment(location_context=location, ocr_text=text),
        score=score,
        matched_vector="ocr_log",
    )


def test_context_block_is_numbered():
    block = build_context_block([_hit("keys on table"), _hit("wallet on counter")])
    assert block.startswith("1.")
    assert "2." in block


def test_prompt_includes_question_and_context():
    prompt = build_prompt("where are my keys?", [_hit("keys on the hallway table")])
    assert "where are my keys?" in prompt
    assert "keys on the hallway table" in prompt


def test_stub_generator_quotes_top_memory():
    prompt = build_prompt("where are my keys?", [_hit("keys on the hallway table")])
    answer = StubGenerator().generate("sys", prompt)
    assert "keys on the hallway table" in answer


def test_stub_generator_handles_no_context():
    prompt = build_prompt("where are my keys?", [])
    answer = StubGenerator().generate("sys", prompt)
    assert "don't have a memory" in answer.lower()


def test_local_rag_end_to_end_offline(store, embedder, settings):
    settings.fake_rag = True
    seed_store(embedder, store)
    rag = LocalRAG(settings, embedder, store, generator=StubGenerator())
    result = rag.ask("where did I leave my keys?", hybrid=True)
    assert result.sources
    assert result.answer
    assert result.answer != "I don't have a memory of that."


def test_local_rag_no_results_returns_safe_answer(store, embedder, settings):
    settings.fake_rag = True
    rag = LocalRAG(settings, embedder, store, generator=StubGenerator())
    # Empty store -> no memories -> safe fallback, no hallucination.
    result = rag.ask("what did I eat in 1990?")
    assert result.sources == []
    assert "don't have a memory" in result.answer.lower()
