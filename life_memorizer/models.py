"""Typed data models shared across the pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# Named vector keys used inside a Qdrant point. Each maps a modality to the
# semantic role it plays in a "moment in time".
VECTOR_VIDEO_FRAME = "video_frame"
VECTOR_AMBIENT_AUDIO = "ambient_audio"
VECTOR_OCR_LOG = "ocr_log"

NAMED_VECTORS = (VECTOR_VIDEO_FRAME, VECTOR_AMBIENT_AUDIO, VECTOR_OCR_LOG)


class Modality(str, Enum):
    """Input modality understood by the embedder."""

    image = "image"
    audio = "audio"
    text = "text"


# Which named vector a modality maps to when stored / queried.
MODALITY_TO_VECTOR: dict[Modality, str] = {
    Modality.image: VECTOR_VIDEO_FRAME,
    Modality.audio: VECTOR_AMBIENT_AUDIO,
    Modality.text: VECTOR_OCR_LOG,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    """Generate a Qdrant-compatible point id (UUID string)."""
    return str(uuid.uuid4())


class Moment(BaseModel):
    """A single indexed "moment in time".

    A moment bundles up to three aligned named vectors (vision / audio / text)
    plus light metadata. Raw media is *not* stored in the DB — only an optional
    path reference, keeping the vector store small and privacy-preserving.
    """

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=_utcnow)
    location_context: str = "Unknown"
    media_file_path: Optional[str] = None
    source_clip: Optional[str] = None
    transcript: Optional[str] = None
    ocr_text: Optional[str] = None

    # Marks a moment created by summarizing several older moments.
    is_summary: bool = False
    summary_count: int = 0

    # Named vectors present on this moment (key -> embedding).
    vectors: dict[str, list[float]] = Field(default_factory=dict)

    def payload(self, store_media_path: bool = True) -> dict[str, Any]:
        """Metadata persisted alongside the vectors in Qdrant.

        When ``store_media_path`` is False the media reference is omitted for
        stronger privacy (no pointer to raw media is persisted).
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "timestamp_epoch": int(self.timestamp.timestamp()),
            "location_context": self.location_context,
            "media_file_path": self.media_file_path if store_media_path else None,
            "source_clip": self.source_clip,
            "transcript": self.transcript,
            "ocr_text": self.ocr_text,
            "is_summary": self.is_summary,
            "summary_count": self.summary_count,
        }

    @classmethod
    def from_payload(cls, point_id: str, payload: dict[str, Any]) -> "Moment":
        """Rebuild a Moment from a Qdrant payload (vectors omitted)."""
        ts_raw = payload.get("timestamp")
        ts = datetime.fromisoformat(ts_raw) if ts_raw else _utcnow()
        return cls(
            id=str(point_id),
            timestamp=ts,
            location_context=payload.get("location_context", "Unknown"),
            media_file_path=payload.get("media_file_path"),
            source_clip=payload.get("source_clip"),
            transcript=payload.get("transcript"),
            ocr_text=payload.get("ocr_text"),
            is_summary=bool(payload.get("is_summary", False)),
            summary_count=int(payload.get("summary_count", 0) or 0),
        )


class RecallHit(BaseModel):
    """A single ranked search result."""

    moment: Moment
    score: float
    matched_vector: str

    def as_context_line(self) -> str:
        """Compact human/LLM-readable summary line for RAG prompting."""
        when = self.moment.timestamp.strftime("%Y-%m-%d %H:%M")
        note = self.moment.ocr_text or self.moment.transcript or (self.moment.media_file_path or "")
        note = (note[:120] + "…") if len(note) > 120 else note
        where = self.moment.location_context
        return f"[{when} @ {where}] ({self.matched_vector}, {self.score:.3f}) {note}".rstrip()
