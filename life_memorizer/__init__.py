"""Life Memorizer: offline, privacy-first multi-modal cognitive assistant.

Indexes vision, audio and text into a unified semantic space (Gemini 2.0) and
stores everything locally in Qdrant Edge for instant, offline semantic recall.
"""

from .config import Settings, get_settings
from .models import Modality, Moment, RecallHit
from .embeddings import GeminiEmbedder, FakeEmbedder, build_embedder
from .store import MemoryStore
from .ingest import Ingestor
from .recall import RecallEngine
from .rag import LocalRAG, RAGAnswer, build_generator

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "get_settings",
    "Modality",
    "Moment",
    "RecallHit",
    "GeminiEmbedder",
    "FakeEmbedder",
    "build_embedder",
    "MemoryStore",
    "Ingestor",
    "RecallEngine",
    "LocalRAG",
    "RAGAnswer",
    "build_generator",
    "__version__",
]
