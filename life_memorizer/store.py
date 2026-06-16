"""Qdrant Edge storage layer.

Runs Qdrant fully embedded against a local, file-backed (memmap) path — no
server, no cloud. Stores each "moment" as a single point carrying up to three
named vectors (video_frame / ambient_audio / ocr_log) plus light metadata, with
on-device quantization and TTL-based pruning.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np
import qdrant_edge as qe

from .config import Quantization, Settings
from .models import (
    MODALITY_TO_VECTOR,
    NAMED_VECTORS,
    Modality,
    Moment,
    RecallHit,
)


class MemoryStore:
    """Thin, typed wrapper around an embedded Qdrant Edge instance."""

    def __init__(self, settings: Settings, client: Optional[qe.EdgeShard] = None) -> None:
        self.settings = settings
        self.collection = settings.collection
        self.dim = settings.embed_dim
        if client is not None:
            self.client = client
        else:
            settings.db_path.mkdir(parents=True, exist_ok=True)
            config_json = settings.db_path / "edge_config.json"
            if config_json.exists():
                self.client = qe.EdgeShard.load(str(settings.db_path))
            else:
                self.client = None

    # --- collection lifecycle ------------------------------------------------
    def _quantization_config(self):
        q = self.settings.quantization
        if q is Quantization.scalar:
            return qe.ScalarQuantizationConfig(
                type=qe.ScalarType.Int8,
                always_ram=True,
            )
        if q is Quantization.binary:
            return qe.BinaryQuantizationConfig(always_ram=True)
        return None

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection with named vectors + quantization if needed."""
        config_json = self.settings.db_path / "edge_config.json"

        if recreate:
            if self.client is not None:
                self.client.close()
                self.client = None
            import shutil
            for path in self.settings.db_path.iterdir():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)

        if self.client is not None:
            return

        if config_json.exists():
            self.client = qe.EdgeShard.load(str(self.settings.db_path))
            return

        vector_params = qe.EdgeVectorParams(
            size=self.dim,
            distance=qe.Distance.Cosine,
            on_disk=True,
        )
        vectors_config = {name: vector_params for name in NAMED_VECTORS}

        config = qe.EdgeConfig(
            vectors=vectors_config,
            on_disk_payload=True,
            quantization_config=self._quantization_config(),
        )
        self.client = qe.EdgeShard.create(str(self.settings.db_path), config)

        # Payload indexes speed up filtered hybrid queries on the edge.
        self.client.update(
            qe.UpdateOperation.create_field_index(
                "location_context",
                qe.PayloadSchemaType.Keyword,
            )
        )
        self.client.update(
            qe.UpdateOperation.create_field_index(
                "timestamp_epoch",
                qe.PayloadSchemaType.Integer,
            )
        )

    # --- writes --------------------------------------------------------------
    def upsert_moments(self, moments: Iterable[Moment]) -> int:
        """Batch-upsert moments. Returns the number written."""
        store_path = self.settings.store_media_path
        points: list[qe.Point] = []
        for moment in moments:
            if not moment.vectors:
                continue
            points.append(
                qe.Point(
                    id=moment.id,
                    vector=dict(moment.vectors),
                    payload=moment.payload(store_media_path=store_path),
                )
            )
        if not points:
            return 0
        self.client.update(qe.UpdateOperation.upsert_points(points))
        return len(points)

    def _build_query_filter(self, location_context: Optional[str]) -> Optional[qe.Filter]:
        must_conditions = []
        must_not_conditions = []

        if location_context:
            must_conditions.append(
                qe.FieldCondition(
                    key="location_context",
                    match=qe.MatchValue(location_context),
                )
            )

        if not self.settings.fake_embeddings:
            must_not_conditions.append(
                qe.FieldCondition(
                    key="source_clip",
                    match=qe.MatchValue("mock://afternoon_session"),
                )
            )

        if not must_conditions and not must_not_conditions:
            return None

        return qe.Filter(
            must=must_conditions or None,
            must_not=must_not_conditions or None,
        )

    # --- reads / search ------------------------------------------------------
    def search(
        self,
        vector_name: str,
        query_vector: list[float],
        limit: int = 5,
        location_context: Optional[str] = None,
    ) -> list[RecallHit]:
        """Search a single named vector space, optionally filtered by location."""
        flt = self._build_query_filter(location_context)
        results = self.client.query(
            qe.QueryRequest(
                query=qe.Query.Nearest(query_vector, using=vector_name),
                limit=limit,
                filter=flt,
                with_payload=True,
                with_vector=False,
            )
        )
        return [
            RecallHit(
                moment=Moment.from_payload(p.id, p.payload or {}),
                score=float(p.score),
                matched_vector=vector_name,
            )
            for p in results
        ]

    def hybrid_search(
        self,
        query_vector: list[float],
        weights: dict[str, float],
        limit: int = 5,
        location_context: Optional[str] = None,
    ) -> list[RecallHit]:
        """Search several named vectors and fuse results with weights.

        The same query vector is run against each requested named space (valid
        because all modalities share Gemini's aligned space). Per-point scores
        are combined as a weighted sum, keeping the best matched vector label.
        """
        fused: dict[str, RecallHit] = {}
        for vector_name, weight in weights.items():
            if weight <= 0:
                continue
            hits = self.search(
                vector_name=vector_name,
                query_vector=query_vector,
                limit=limit * 3,
                location_context=location_context,
            )
            for hit in hits:
                weighted = hit.score * weight
                existing = fused.get(hit.moment.id)
                if existing is None:
                    fused[hit.moment.id] = RecallHit(
                        moment=hit.moment,
                        score=weighted,
                        matched_vector=hit.matched_vector,
                    )
                else:
                    existing.score += weighted
                    if weighted > hit.score * weights.get(existing.matched_vector, 1.0):
                        existing.matched_vector = hit.matched_vector
        ranked = sorted(fused.values(), key=lambda h: h.score, reverse=True)
        return ranked[:limit]

    def vector_name_for(self, modality: Modality) -> str:
        return MODALITY_TO_VECTOR[modality]

    def _scroll_all(self, flt: Optional[qe.Filter] = None) -> list[qe.Record]:
        """Read every point (with vectors + payload) matching an optional filter."""
        records: list[qe.Record] = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                qe.ScrollRequest(
                    filter=flt,
                    with_payload=True,
                    with_vector=True,
                    limit=256,
                    offset=offset,
                )
            )
            records.extend(batch)
            if offset is None:
                break
        return records

    # --- maintenance ---------------------------------------------------------
    def count(self) -> int:
        return self.client.count(qe.CountRequest(exact=True))

    def prune_expired(self, ttl_days: Optional[int] = None) -> int:
        """Delete moments older than the TTL. Returns count requested for deletion.

        A TTL of 0 disables pruning entirely.
        """
        ttl = self.settings.ttl_days if ttl_days is None else ttl_days
        if ttl <= 0:
            return 0
        cutoff_epoch = int(time.time()) - ttl * 86400
        before = self.count()
        self.client.update(
            qe.UpdateOperation.delete_points_by_filter(
                qe.Filter(
                    must=[
                        qe.FieldCondition(
                            key="timestamp_epoch",
                            range=qe.RangeFloat(lt=cutoff_epoch),
                        )
                    ]
                )
            )
        )
        after = self.count()
        return max(before - after, 0)

    def _expired_filter(self, cutoff_epoch: int) -> qe.Filter:
        return qe.Filter(
            must=[
                qe.FieldCondition(
                    key="timestamp_epoch",
                    range=qe.RangeFloat(lt=cutoff_epoch),
                )
            ]
        )

    def summarize_expired(self, ttl_days: Optional[int] = None) -> int:
        """Collapse expired moments into compact digests, then delete originals.

        For each ``location_context`` among the expired moments, the named
        vectors are mean-pooled (and re-normalized) into one representative
        "memory digest" point, with the original OCR/transcript text trimmed and
        concatenated extractively. This keeps storage roughly constant over time
        instead of growing, while preserving a searchable gist of the past.

        Returns the net change in stored points removed (originals minus digests).
        A TTL of 0 disables summarization.
        """
        ttl = self.settings.ttl_days if ttl_days is None else ttl_days
        if ttl <= 0:
            return 0
        cutoff_epoch = int(time.time()) - ttl * 86400
        expired = self._scroll_all(self._expired_filter(cutoff_epoch))
        if not expired:
            return 0

        before = self.count()
        digests = self._build_digests(expired)

        # Delete originals first, then write digests (new ids), all within TTL.
        self.client.update(
            qe.UpdateOperation.delete_points_by_filter(self._expired_filter(cutoff_epoch))
        )
        if digests:
            self.upsert_moments(digests)
        after = self.count()
        return max(before - after, 0)

    def _build_digests(self, records: list[qe.Record]) -> list[Moment]:
        """Group expired records by location and mean-pool into digest Moments."""
        by_location: dict[str, list[qe.Record]] = {}
        for rec in records:
            loc = (rec.payload or {}).get("location_context", "Unknown")
            by_location.setdefault(loc, []).append(rec)

        digests: list[Moment] = []
        for location, group in by_location.items():
            pooled = self._mean_pool_vectors(group)
            if not pooled:
                continue
            notes: list[str] = []
            latest_epoch = 0
            for rec in group:
                payload = rec.payload or {}
                text = payload.get("ocr_text") or payload.get("transcript")
                if text:
                    notes.append(text.replace("\n", " ").strip())
                latest_epoch = max(latest_epoch, int(payload.get("timestamp_epoch", 0) or 0))
            digest_text = _extractive_summary(notes)
            ts = datetime.fromtimestamp(latest_epoch or time.time(), tz=timezone.utc)
            digests.append(
                Moment(
                    timestamp=ts,
                    location_context=location,
                    source_clip="summary://digest",
                    ocr_text=digest_text or None,
                    is_summary=True,
                    summary_count=len(group),
                    vectors=pooled,
                )
            )
        return digests

    @staticmethod
    def _mean_pool_vectors(records: list[qe.Record]) -> dict[str, list[float]]:
        """Average each named vector across records and L2-normalize the result."""
        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for rec in records:
            vectors = rec.vector or {}
            if not isinstance(vectors, dict):
                continue
            for name, vec in vectors.items():
                arr = np.asarray(vec, dtype=np.float32)
                sums[name] = sums.get(name, np.zeros_like(arr)) + arr
                counts[name] = counts.get(name, 0) + 1
        pooled: dict[str, list[float]] = {}
        for name, total in sums.items():
            mean = total / max(counts[name], 1)
            norm = float(np.linalg.norm(mean))
            if norm > 0:
                mean = mean / norm
            pooled[name] = mean.astype(np.float32).tolist()
        return pooled

    def maybe_auto_prune(self, state_path=None) -> int:
        """Run pruning at most once per configured interval (cadence guard).

        Records the last run timestamp in a small file next to the DB so repeated
        process starts don't prune on every invocation. Returns points removed
        (0 if skipped or disabled).
        """
        if not self.settings.auto_prune or self.settings.ttl_days <= 0:
            return 0
        from pathlib import Path

        marker = Path(state_path) if state_path else self.settings.db_path / ".last_prune"
        now = time.time()
        interval = self.settings.auto_prune_interval_hours * 3600
        try:
            last = float(marker.read_text().strip()) if marker.exists() else 0.0
        except (ValueError, OSError):
            last = 0.0
        if now - last < interval:
            return 0
        removed = (
            self.summarize_expired()
            if self.settings.summarize_on_prune
            else self.prune_expired()
        )
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(now))
        except OSError:
            pass
        return removed


def _extractive_summary(notes: list[str], max_chars: int = 280) -> str:
    """Build a compact, deterministic digest string from moment texts.

    Extractive only (no LLM): de-duplicates, keeps order, and trims to a budget.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for note in notes:
        key = note.lower()
        if note and key not in seen:
            seen.add(key)
            unique.append(note)
    joined = " | ".join(unique)
    if len(joined) > max_chars:
        joined = joined[: max_chars - 1].rstrip() + "\u2026"
    return joined


