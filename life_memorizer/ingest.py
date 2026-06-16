"""Ingest pipeline: turn sensory feeds into stored Moments.

A 5-minute clip (or a live smart-glasses stream chunk) is decomposed into
sampled frames, audio chunks and OCR text. Each is embedded through Gemini 2.0
into its named vector and grouped into Moments which are batch-upserted into
Qdrant Edge.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .config import Settings
from .embeddings import Embedder
from .media import extract_audio_chunks, ocr_image, sample_frames
from .models import (
    VECTOR_AMBIENT_AUDIO,
    VECTOR_OCR_LOG,
    VECTOR_VIDEO_FRAME,
    Modality,
    Moment,
)
from .store import MemoryStore


class Ingestor:
    """Builds Moments from media and writes them to the store."""

    def __init__(self, settings: Settings, embedder: Embedder, store: MemoryStore) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store

    # --- high level ----------------------------------------------------------
    def ingest_video(
        self,
        video_path: str | Path,
        location: str = "Unknown",
        base_time: Optional[datetime] = None,
        keep_media: bool = True,
    ) -> list[Moment]:
        """Ingest a video clip into local memory and return the stored Moments."""
        video_path = Path(video_path)
        base_time = base_time or datetime.now(timezone.utc)
        work_dir = Path(tempfile.mkdtemp(prefix="life_memorizer_"))

        frames = sample_frames(video_path, work_dir / "frames", fps=self.settings.fps)
        audio_chunks = extract_audio_chunks(video_path, work_dir / "audio")

        moments = self._build_moments(
            frames=frames,
            audio_chunks=audio_chunks,
            location=location,
            base_time=base_time,
            source_clip=str(video_path),
            keep_media=keep_media,
        )
        self.store.upsert_moments(moments)
        return moments

    def ingest_manual(
        self,
        location: str = "Unknown",
        image_path: Optional[str | Path] = None,
        audio_path: Optional[str | Path] = None,
        text_note: Optional[str] = None,
        base_time: Optional[datetime] = None,
    ) -> Moment:
        """Ingest a single hand-assembled moment (image and/or audio and/or text).

        Useful for the tutorial's mock stream: a snapshot of a lost wallet, an
        audio clip of a conversation, and text from a restaurant menu.
        """
        moment = Moment(
            timestamp=base_time or datetime.now(timezone.utc),
            location_context=location,
        )
        if image_path is not None:
            moment.media_file_path = str(image_path)
            moment.vectors[VECTOR_VIDEO_FRAME] = self.embedder.embed_image(image_path)
            ocr = ocr_image(image_path)
            if ocr:
                moment.ocr_text = ocr
                moment.vectors[VECTOR_OCR_LOG] = self.embedder.embed_text(ocr)
        if audio_path is not None:
            moment.vectors[VECTOR_AMBIENT_AUDIO] = self.embedder.embed_audio(audio_path)
        if text_note:
            moment.ocr_text = (moment.ocr_text or text_note)
            moment.vectors[VECTOR_OCR_LOG] = self.embedder.embed_text(text_note)
        if not moment.vectors:
            raise ValueError("ingest_manual requires at least one of image/audio/text.")
        self.store.upsert_moments([moment])
        return moment

    # --- internals -----------------------------------------------------------
    def _build_moments(
        self,
        frames,
        audio_chunks,
        location: str,
        base_time: datetime,
        source_clip: str,
        keep_media: bool,
    ) -> list[Moment]:
        """Align frames and audio chunks by time bucket into Moments."""
        buckets: dict[int, Moment] = {}

        def bucket_for(second: float) -> Moment:
            key = int(second)
            if key not in buckets:
                buckets[key] = Moment(
                    timestamp=base_time + timedelta(seconds=key),
                    location_context=location,
                    source_clip=source_clip,
                )
            return buckets[key]

        for frame in frames:
            moment = bucket_for(frame.second)
            if keep_media:
                moment.media_file_path = str(frame.path)
            moment.vectors[VECTOR_VIDEO_FRAME] = self.embedder.embed_image(frame.path)
            ocr = ocr_image(frame.path)
            if ocr:
                moment.ocr_text = ocr
                moment.vectors[VECTOR_OCR_LOG] = self.embedder.embed_text(ocr)

        for chunk in audio_chunks:
            moment = bucket_for(chunk.second)
            moment.vectors[VECTOR_AMBIENT_AUDIO] = self.embedder.embed_audio(chunk.path)

        return [m for m in buckets.values() if m.vectors]
