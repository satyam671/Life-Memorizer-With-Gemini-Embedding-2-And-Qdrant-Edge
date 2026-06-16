"""Recall engine: query the local memory across modalities.

Because every modality shares Gemini's aligned space, a single text prompt can
be matched against the visual, audio or text named vectors interchangeably, and
the hybrid mode fuses them with configurable weights and payload filters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import Settings
from .embeddings import Embedder
from .models import (
    VECTOR_AMBIENT_AUDIO,
    VECTOR_OCR_LOG,
    VECTOR_VIDEO_FRAME,
    Modality,
    RecallHit,
)
from .store import MemoryStore

# Default fusion weights for hybrid search. Vision and text are favoured for
# typical "where / what did I read" queries; tune per deployment.
DEFAULT_HYBRID_WEIGHTS: dict[str, float] = {
    VECTOR_VIDEO_FRAME: 1.0,
    VECTOR_OCR_LOG: 1.0,
    VECTOR_AMBIENT_AUDIO: 0.6,
}


class RecallEngine:
    """Run visual / audio / text / hybrid recall over stored moments."""

    def __init__(self, settings: Settings, embedder: Embedder, store: MemoryStore) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store

    def recall(
        self,
        query: str,
        modality: Modality = Modality.text,
        target: Optional[Modality] = None,
        limit: int = 5,
        location_context: Optional[str] = None,
        hybrid: bool = False,
        weights: Optional[dict[str, float]] = None,
    ) -> list[RecallHit]:
        """Recall moments matching ``query``.

        Args:
            query: text prompt, or a path to an image/audio file when ``modality``
                is image/audio.
            modality: the modality of the *query* itself.
            target: which named vector space to search; defaults to a sensible
                mapping (text->ocr, image->video_frame, audio->ambient_audio).
            hybrid: if True, search all spaces and fuse with ``weights``.
        """
        query_vector = self._embed_query(query, modality)

        if hybrid:
            return self.store.hybrid_search(
                query_vector=query_vector,
                weights=weights or DEFAULT_HYBRID_WEIGHTS,
                limit=limit,
                location_context=location_context,
            )

        target_modality = target or modality
        vector_name = self.store.vector_name_for(target_modality)
        return self.store.search(
            vector_name=vector_name,
            query_vector=query_vector,
            limit=limit,
            location_context=location_context,
        )

    def visual_search(self, query: str, **kwargs) -> list[RecallHit]:
        """Scenario A: search the video_frame space (e.g. \"where are my keys?\")."""
        kwargs.setdefault("target", Modality.image)
        return self.recall(query, modality=Modality.text, **kwargs)

    def audio_recall(self, query: str, **kwargs) -> list[RecallHit]:
        """Scenario B: search the ambient_audio space (e.g. \"what did Sarah say?\")."""
        kwargs.setdefault("target", Modality.audio)
        return self.recall(query, modality=Modality.text, **kwargs)

    def _embed_query(self, query: str, modality: Modality) -> list[float]:
        if modality is Modality.text:
            return self.embedder.embed_text(query)
        if modality is Modality.image:
            return self.embedder.embed_image(Path(query))
        if modality is Modality.audio:
            return self.embedder.embed_audio(Path(query))
        raise ValueError(f"Unsupported query modality: {modality!r}")
