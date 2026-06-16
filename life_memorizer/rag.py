"""Local RAG: answer questions using recalled memories.

This is the tutorial's expansion step. A natural-language question is embedded,
matched against the local Qdrant Edge memories (via :class:`RecallEngine`), and
the top moments are turned into a grounded prompt for an on-device LLM (Gemma).
The model answers *only* from the retrieved context, so it acts like a private,
offline cognitive assistant rather than a general chatbot.

Three generator backends are provided:

- :class:`OllamaGenerator`  - fully local Gemma via an Ollama HTTP server.
- :class:`GeminiGenerator`  - Gemma through the google-genai API.
- :class:`StubGenerator`    - deterministic, offline, extractive (no model).

The stub lets the whole `ask` flow run with zero setup (and powers the tests).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional, Protocol, runtime_checkable

from .config import Settings
from .embeddings import Embedder
from .models import Modality, RecallHit
from .recall import RecallEngine
from .store import MemoryStore

_SYSTEM_PROMPT = (
    "You are a private, on-device memory assistant. Answer the user's question "
    "using ONLY the recalled memories provided as context. Be concise and "
    "conversational, as if speaking aloud. If the context does not contain the "
    "answer, say you don't have a memory of that. Never invent details."
)

_NO_CONTEXT_ANSWER = "I don't have a memory of that."


@runtime_checkable
class Generator(Protocol):
    """Turns a grounded prompt into an answer string."""

    def generate(self, system: str, prompt: str) -> str: ...


def build_context_block(hits: list[RecallHit]) -> str:
    """Render recalled moments as a numbered context block for the prompt."""
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"{i}. {hit.as_context_line()}")
    return "\n".join(lines)


def build_prompt(question: str, hits: list[RecallHit]) -> str:
    """Assemble the grounded user prompt from the question and recalled context."""
    context = build_context_block(hits) if hits else "(no relevant memories found)"
    return (
        f"Recalled memories:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer using only the memories above."
    )


class StubGenerator:
    """Deterministic, offline generator (no LLM).

    Produces an extractive answer by quoting the most relevant recalled memory.
    Good enough to demo the end-to-end RAG flow without any model installed.
    """

    def generate(self, system: str, prompt: str) -> str:
        # The prompt embeds the context block; recover the first memory line.
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        memory_lines = [ln for ln in lines if ln[0:2].rstrip(".").isdigit()]
        if not memory_lines or "(no relevant memories found)" in prompt:
            return _NO_CONTEXT_ANSWER
        top = memory_lines[0]
        # Strip the leading "1. " and the bracketed metadata for a clean answer.
        body = top.split(")", 1)[-1].strip() if ")" in top else top
        body = body.lstrip("0123456789. ").strip()
        return f"Based on your memories: {body}" if body else _NO_CONTEXT_ANSWER


class OllamaGenerator:
    """Fully local Gemma via an Ollama server (http://localhost:11434)."""

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it running and is the "
                f"model '{self.model}' pulled? (`ollama pull {self.model}`). "
                "Or use LIFE_MEMORIZER_FAKE_RAG=1 for the offline stub."
            ) from exc
        return str(body.get("response", "")).strip() or _NO_CONTEXT_ANSWER


class GeminiGenerator:
    """Gemma through the google-genai API."""

    def __init__(self, settings: Settings) -> None:
        self.model = settings.rag_model
        api_key = settings.require_api_key()
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "google-genai is not installed. `pip install google-genai` or use "
                "LIFE_MEMORIZER_FAKE_RAG=1 / the ollama backend."
            ) from exc
        self._client = genai.Client(api_key=api_key)

    def generate(self, system: str, prompt: str) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return (response.text or "").strip() or _NO_CONTEXT_ANSWER


def build_generator(settings: Settings) -> Generator:
    """Factory: pick the generation backend from settings."""
    if settings.fake_rag:
        return StubGenerator()
    backend = settings.rag_backend.lower()
    if backend == "ollama":
        return OllamaGenerator(model=settings.rag_model, host=settings.ollama_host)
    if backend == "gemini":
        return GeminiGenerator(settings)
    raise ValueError(
        f"Unknown rag_backend {settings.rag_backend!r}. Use 'ollama', 'gemini', "
        "or set LIFE_MEMORIZER_FAKE_RAG=1."
    )


class RAGAnswer:
    """An answer plus the memories it was grounded on."""

    def __init__(self, answer: str, sources: list[RecallHit]) -> None:
        self.answer = answer
        self.sources = sources


class LocalRAG:
    """Retrieve-then-generate over local memories."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: MemoryStore,
        generator: Optional[Generator] = None,
    ) -> None:
        self.settings = settings
        self.engine = RecallEngine(settings, embedder, store)
        self.generator = generator or build_generator(settings)

    def ask(
        self,
        question: str,
        location_context: Optional[str] = None,
        hybrid: bool = True,
        limit: Optional[int] = None,
    ) -> RAGAnswer:
        """Answer ``question`` using recalled memories as grounding context."""
        k = limit or self.settings.rag_context_size
        hits = self.engine.recall(
            query=question,
            modality=Modality.text,
            limit=k,
            location_context=location_context,
            hybrid=hybrid,
        )
        if not hits:
            return RAGAnswer(_NO_CONTEXT_ANSWER, [])
        prompt = build_prompt(question, hits)
        answer = self.generator.generate(_SYSTEM_PROMPT, prompt)
        return RAGAnswer(answer, hits)
