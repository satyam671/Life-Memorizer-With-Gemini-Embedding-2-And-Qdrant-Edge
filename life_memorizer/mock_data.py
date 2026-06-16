"""Realistic mock memory dataset for the Life Memorizer tutorial.

This simulates a smart-glasses wearer's afternoon. The moments are deliberately
crafted so the three tutorial recall scenarios return *meaningful, correct*
results rather than coincidental matches:

  Scenario A (visual):  "Where did I leave my keys?"        -> the keys moment
  Scenario B (audio):   "What did Sarah say to buy?"        -> Sarah's request
  Scenario C (hybrid):  "the cafe menu" filtered to Cafe    -> the menu moment

Each entry carries the kind of OCR text / transcript a wearable would actually
capture, plus location_context and a relative timestamp, so payload filtering and
ranking behave like a real session. No external media is required: the moments
are embedded from their textual content via whichever embedder is supplied (the
offline FakeEmbedder works end-to-end with no API key).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from .embeddings import Embedder
from .models import (
    VECTOR_AMBIENT_AUDIO,
    VECTOR_OCR_LOG,
    VECTOR_VIDEO_FRAME,
    Moment,
)


@dataclass(frozen=True)
class MockMoment:
    """A scripted moment in the wearer's day.

    Attributes:
        minutes_ago: how long before "now" this happened (for realistic ordering).
        location: location_context tag (Home / Cafe / Work / Street ...).
        scene: what the wearer was looking at (drives the video_frame vector).
        speech: ambient speech transcript (drives the ambient_audio vector).
        ocr: text read in-scene, e.g. signage or a menu (drives the ocr_log vector).
        media_file_path: where the captured frame/clip would live on the device.
    """

    minutes_ago: int
    location: str
    scene: Optional[str] = None
    speech: Optional[str] = None
    ocr: Optional[str] = None
    media_file_path: Optional[str] = None
    tags: tuple[str, ...] = field(default_factory=tuple)


# A coherent, ~90-minute slice of a day with distinct, queryable events.
MOCK_MOMENTS: tuple[MockMoment, ...] = (
    MockMoment(
        minutes_ago=88,
        location="Home",
        scene="a set of brass house keys lying on the wooden hallway table next to a "
        "blue ceramic bowl and a stack of unopened mail",
        ocr="",
        media_file_path="media_cache/home/hallway_table_keys.jpg",
        tags=("keys", "home"),
    ),
    MockMoment(
        minutes_ago=86,
        location="Home",
        scene="a brown leather wallet left on the kitchen counter beside the toaster",
        speech="don't forget your wallet, it's on the counter by the toaster",
        media_file_path="media_cache/home/kitchen_counter_wallet.jpg",
        tags=("wallet", "home"),
    ),
    MockMoment(
        minutes_ago=72,
        location="Street",
        scene="a pedestrian crossing with a green walk signal and a red double-decker bus",
        ocr="WAIT  /  WALK",
        media_file_path="media_cache/street/crossing.jpg",
        tags=("commute", "street"),
    ),
    MockMoment(
        minutes_ago=64,
        location="Street",
        scene="Sarah walking beside me on the sidewalk holding a reusable shopping bag",
        speech="Sarah says: can you buy and grab oat milk and a dozen eggs on the way back? we're out of "
        "both, and maybe some fresh basil if they have it",
        media_file_path="media_cache/street/with_sarah.jpg",
        tags=("sarah", "shopping", "conversation"),
    ),
    MockMoment(
        minutes_ago=58,
        location="Cafe",
        scene="the chalkboard menu mounted above the espresso bar at Maple & Co cafe",
        ocr="MAPLE & CO\nFlat White  4.20\nCappuccino  4.00\nCold Brew  4.50\n"
        "Avocado Toast  8.50\nOat milk +0.60",
        media_file_path="media_cache/cafe/menu_board.jpg",
        tags=("menu", "cafe"),
    ),
    MockMoment(
        minutes_ago=55,
        location="Cafe",
        scene="my cafe order of a flat white coffee in a white cup on a marble table next to my phone",
        speech="I would like to order one flat white coffee and an avocado toast please, thanks at the cafe",
        media_file_path="media_cache/cafe/table_order.jpg",
        tags=("cafe", "order"),
    ),
    MockMoment(
        minutes_ago=30,
        location="Work",
        scene="car parked beside a parking garage pillar painted with the level marker",
        ocr="LEVEL 3 - SECTION B - SPACE 142",
        media_file_path="media_cache/work/parking_level3.jpg",
        tags=("parking", "work", "car"),
    ),
    MockMoment(
        minutes_ago=12,
        location="Work",
        scene="a whiteboard with the sprint plan and three sticky notes",
        speech="let's ship the edge quantization change before the demo on friday",
        ocr="SPRINT 7\n- Matryoshka downscale 3072->768\n- Scalar quantization\n- TTL pruning",
        media_file_path="media_cache/work/whiteboard_sprint.jpg",
        tags=("work", "planning"),
    ),
)


def build_mock_moments(
    embedder: Embedder,
    base_time: Optional[datetime] = None,
) -> list[Moment]:
    """Embed the scripted dataset into real :class:`Moment` objects.

    Each non-empty modality field is embedded into its named vector through the
    supplied embedder, so the result is ready to upsert into Qdrant Edge.
    """
    import uuid
    now = base_time or datetime.now(timezone.utc)
    moments: list[Moment] = []
    for idx, mock in enumerate(MOCK_MOMENTS):
        moment_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"mock-moment-{idx}"))
        moment = Moment(
            id=moment_id,
            timestamp=now - timedelta(minutes=mock.minutes_ago),
            location_context=mock.location,
            media_file_path=mock.media_file_path,
            source_clip="mock://afternoon_session",
        )
        if mock.scene:
            moment.vectors[VECTOR_VIDEO_FRAME] = embedder.embed_text(mock.scene)
        if mock.speech:
            moment.transcript = mock.speech
            moment.vectors[VECTOR_AMBIENT_AUDIO] = embedder.embed_text(mock.speech)
        if mock.ocr:
            moment.ocr_text = mock.ocr
            moment.vectors[VECTOR_OCR_LOG] = embedder.embed_text(mock.ocr)
        if moment.vectors:
            moments.append(moment)
    return moments


def seed_store(embedder: Embedder, store, base_time: Optional[datetime] = None) -> int:
    """Embed and upsert the mock dataset into ``store``. Returns moments written."""
    moments = build_mock_moments(embedder, base_time=base_time)
    return store.upsert_moments(moments)
