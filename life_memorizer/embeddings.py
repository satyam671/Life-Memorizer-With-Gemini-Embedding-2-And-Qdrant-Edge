"""Multi-modal embedding layer.

Wraps Gemini Embedding 2.0 so that images, audio and text are all projected into
the *same* aligned vector space, then applies Matryoshka truncation (3072 -> N)
and L2 normalization so vectors are edge-friendly and cosine-ready.

A deterministic :class:`FakeEmbedder` is provided so the entire pipeline runs
offline with no API key (useful for tests, demos and CI).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .config import Settings
from .models import Modality


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0 or math.isnan(norm):
        return vec
    return vec / norm


def matryoshka_truncate(vector: np.ndarray, dim: int) -> np.ndarray:
    """Truncate a Gemini embedding to ``dim`` then re-normalize.

    Gemini's Matryoshka training means the leading ``dim`` components form a
    usable lower-dimensional embedding, so we can simply slice and renormalize.
    """
    if vector.shape[0] < dim:
        # Pad defensively; should not happen with real Gemini output.
        vector = np.pad(vector, (0, dim - vector.shape[0]))
    truncated = vector[:dim].astype(np.float32)
    return _l2_normalize(truncated)


@runtime_checkable
class Embedder(Protocol):
    """Common interface for all embedders."""

    dim: int

    def embed_text(self, text: str) -> list[float]: ...
    def embed_image(self, image_path: str | Path) -> list[float]: ...
    def embed_audio(self, audio_path: str | Path) -> list[float]: ...

    def embed(self, modality: Modality, payload: str | Path) -> list[float]: ...


class _BaseEmbedder:
    """Shared dispatch logic for concrete embedders."""

    dim: int

    def embed(self, modality: Modality, payload: str | Path) -> list[float]:
        if modality is Modality.text:
            return self.embed_text(str(payload))
        if modality is Modality.image:
            return self.embed_image(payload)
        if modality is Modality.audio:
            return self.embed_audio(payload)
        raise ValueError(f"Unsupported modality: {modality!r}")

    # Concrete subclasses implement these.
    def embed_text(self, text: str) -> list[float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed_image(self, image_path: str | Path) -> list[float]:  # pragma: no cover
        raise NotImplementedError

    def embed_audio(self, audio_path: str | Path) -> list[float]:  # pragma: no cover
        raise NotImplementedError


# Very small English stopword set; removing these sharpens token overlap so a
# question like "where did I leave my keys?" focuses on "keys".
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "and", "or",
        "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
        "with", "as", "i", "my", "me", "you", "your", "we", "our", "they",
        "did", "do", "does", "where", "what", "who", "when", "how", "why",
        "can", "could", "would", "should", "will", "shall", "may", "might",
        "have", "has", "had", "get", "got", "there", "here", "so", "if",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords removed."""
    cleaned = []
    word = []
    for ch in text.lower():
        if ch.isalnum():
            word.append(ch)
        else:
            if word:
                cleaned.append("".join(word))
                word = []
    if word:
        cleaned.append("".join(word))
    return [t for t in cleaned if t not in _STOPWORDS and len(t) > 1]


def _hash_index(token: str, dim: int) -> tuple[int, float]:
    """Map a token to a (dimension, sign) via stable hashing (feature hashing)."""
    digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 - not security
    idx = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return idx, sign


class FakeEmbedder(_BaseEmbedder):
    """Deterministic, offline, *semantic* embedder for tests / demos.

    Unlike a plain content hash (which scatters similar texts to unrelated
    vectors), this uses **feature hashing** over tokens: each token — plus its
    character trigrams for fuzzy overlap (keys/key) — is hashed into a vector
    dimension and accumulated. Texts that share words therefore land close under
    cosine similarity, so natural-language recall returns sensible results with
    no API key. It is fully deterministic and a drop-in for GeminiEmbedder.

    It is *not* a real semantic model: it captures lexical overlap, not deep
    meaning. For true multi-modal semantics use the real Gemini embedder.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def _bag_vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            # Fall back to a stable per-string vector so empty/edge inputs still work.
            return self._seeded_vector(text)
        for token in tokens:
            idx, sign = _hash_index(token, self.dim)
            vec[idx] += sign
            # Character trigrams give partial credit for related word forms.
            padded = f"#{token}#"
            for i in range(len(padded) - 2):
                tri = padded[i : i + 3]
                t_idx, t_sign = _hash_index(f"tri::{tri}", self.dim)
                vec[t_idx] += t_sign * 0.3
        normalized = _l2_normalize(vec)
        if float(np.linalg.norm(normalized)) == 0.0:
            return self._seeded_vector(text)
        return normalized.tolist()

    def _seeded_vector(self, key: str) -> list[float]:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        return _l2_normalize(vec).tolist()

    def embed_text(self, text: str) -> list[float]:
        return self._bag_vector(text)

    def embed_image(self, image_path: str | Path) -> list[float]:
        # Mock dataset paths don't exist on disk; their names describe the scene
        # (e.g. hallway_table_keys.jpg) so we embed the path's words. Real files
        # are embedded from a stable content hash instead.
        return self._bag_vector(_describe_media(image_path))

    def embed_audio(self, audio_path: str | Path) -> list[float]:
        return self._bag_vector(_describe_media(audio_path))


def _describe_media(path: str | Path) -> str:
    """Turn a media reference into text the bag embedder can use.

    - Real file on disk: stable content hash (no semantics, but consistent).
    - Non-existent mock path: the human-readable words in the path/filename.
    """
    p = Path(path)
    if p.exists() and p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return f"file {h.hexdigest()}"
    # Use the path stem (and parent folder) words: home/hallway_table_keys.jpg
    parts = [p.stem]
    if p.parent.name:
        parts.append(p.parent.name)
    return " ".join(parts).replace("_", " ").replace("-", " ")


class GeminiEmbedder(_BaseEmbedder):
    """Gemini Embedding 2.0 wrapper producing aligned multi-modal vectors.

    All modalities are routed through the same embedding model so the resulting
    vectors live in one shared space, then Matryoshka-truncated to ``dim``.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Gemini Embedding 2 emits 3072 dims; we Matryoshka-downscale to dim.
        self.dim = settings.embed_dim
        self.model = settings.embed_model
        api_key = settings.require_api_key()
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "google-genai is not installed. Run `pip install google-genai` or "
                "use the offline stub via LIFE_MEMORIZER_FAKE_EMBEDDINGS=1."
            ) from exc
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    # --- public API ----------------------------------------------------------
    def embed_text(self, text: str) -> list[float]:
        return self._embed_content(text)

    def embed_image(self, image_path: str | Path) -> list[float]:
        part = self._file_part(image_path, default_mime="image/jpeg")
        return self._embed_content(part)

    def embed_audio(self, audio_path: str | Path) -> list[float]:
        part = self._file_part(audio_path, default_mime="audio/wav")
        return self._embed_content(part)

    # --- internals -----------------------------------------------------------
    def _file_part(self, path: str | Path, default_mime: str):
        from google.genai import types

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Media file not found: {p}")
        mime = _guess_mime(p, default_mime)
        return types.Part.from_bytes(data=p.read_bytes(), mime_type=mime)

    def _embed_content(self, content) -> list[float]:
        from google.genai import types

        config = types.EmbedContentConfig(output_dimensionality=self.settings.gemini_output_dim)
        response = self._client.models.embed_content(
            model=self.model,
            contents=content,
            config=config,
        )
        values = response.embeddings[0].values
        vec = np.asarray(values, dtype=np.float32)
        return matryoshka_truncate(vec, self.dim).tolist()


def _guess_mime(path: Path, default: str) -> str:
    import mimetypes

    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or default


def build_embedder(settings: Settings) -> Embedder:
    """Factory: return the offline stub or the real Gemini embedder."""
    if settings.fake_embeddings:
        return FakeEmbedder(dim=settings.embed_dim)
    return GeminiEmbedder(settings)
