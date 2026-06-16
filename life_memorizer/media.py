"""Media helpers: frame sampling, audio extraction and OCR.

Every function degrades gracefully: if an optional dependency (OpenCV, ffmpeg,
Tesseract) is missing, it raises a clear, actionable error rather than crashing
obscurely. This keeps the core pipeline importable on minimal edge installs.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampledFrame:
    """A frame extracted from a video at a given offset."""

    path: Path
    second: float


@dataclass(frozen=True)
class AudioChunk:
    """An audio segment extracted from a video at a given offset."""

    path: Path
    second: float


def sample_frames(video_path: str | Path, out_dir: Path, fps: float = 1.0) -> list[SampledFrame]:
    """Sample frames from a video at ``fps`` frames per second.

    Returns saved JPEG frames. Requires OpenCV (install via ``.[media]``).
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "OpenCV is required for frame sampling. Install with `pip install "
            "opencv-python-headless` or `pip install -e \".[media]\"`."
        ) from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(source_fps / fps)), 1)
    frames: list[SampledFrame] = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                second = index / source_fps
                frame_path = out_dir / f"frame_{int(second * 1000):08d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frames.append(SampledFrame(path=frame_path, second=second))
            index += 1
    finally:
        cap.release()
    return frames


def extract_audio_chunks(
    video_path: str | Path,
    out_dir: Path,
    chunk_seconds: float = 5.0,
) -> list[AudioChunk]:
    """Extract fixed-length WAV chunks from a video's audio track via ffmpeg."""
    ffmpeg = _ffmpeg_binary()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = _probe_duration(video_path)
    chunks: list[AudioChunk] = []
    start = 0.0
    while start < duration:
        chunk_path = out_dir / f"audio_{int(start * 1000):08d}.wav"
        cmd = [
            ffmpeg,
            "-y",
            "-ss",
            str(start),
            "-t",
            str(chunk_seconds),
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(chunk_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and chunk_path.exists() and chunk_path.stat().st_size > 0:
            chunks.append(AudioChunk(path=chunk_path, second=start))
        start += chunk_seconds
    return chunks


def ocr_image(image_path: str | Path) -> str:
    """Run OCR on an image. Returns extracted text, or "" if OCR is unavailable.

    OCR is best-effort: if pytesseract / Tesseract is not installed we return an
    empty string so ingest can continue (the image is still embedded visually).
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    if shutil.which("tesseract") is None:
        return ""
    try:
        with Image.open(str(image_path)) as img:
            return pytesseract.image_to_string(img).strip()
    except Exception:  # pragma: no cover - OCR best effort
        return ""


def _ffmpeg_binary() -> str:
    """Locate an ffmpeg binary (system or the imageio-ffmpeg bundled one)."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "ffmpeg is required for audio extraction. Install ffmpeg or "
            "`pip install imageio-ffmpeg`."
        ) from exc


def _probe_duration(video_path: Path) -> float:
    """Best-effort video duration in seconds (falls back to 300s = 5 min)."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            if fps > 0 and frames > 0:
                return frames / fps
    except Exception:  # pragma: no cover
        pass
    return 300.0
