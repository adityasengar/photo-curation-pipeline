"""OpenCV-based heuristic filters: blur detection & exposure analysis."""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from config import PipelineConfig

log = logging.getLogger(__name__)
FFMPEG_BIN = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


@dataclass
class FilterResult:
    passed: bool
    laplacian_var: float
    black_clip_pct: float
    white_clip_pct: float
    rejection_reason: str = ""


def decode_image(raw_bytes: bytes) -> np.ndarray | None:
    """Decode raw file bytes into a BGR numpy array with EXIF orientation applied."""
    img = _decode_with_pillow_bytes(raw_bytes)
    if img is not None:
        return img

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("cv2.imdecode returned None — corrupt or unsupported format")
    return img


def decode_image_path(path: Path) -> np.ndarray | None:
    """Decode an image from a filesystem path, including EXIF orientation for common formats."""
    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        return _decode_heic_with_sips(path)

    if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        img = _decode_with_pillow_path(path)
        if img is not None:
            return img

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        log.warning("cv2.imread returned None for %s", path)
    return img


def _decode_with_pillow_bytes(raw_bytes: bytes) -> np.ndarray | None:
    try:
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            return _pil_to_bgr(ImageOps.exif_transpose(pil_img))
    except Exception:
        return None


def _decode_with_pillow_path(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as pil_img:
            return _pil_to_bgr(ImageOps.exif_transpose(pil_img))
    except Exception as exc:
        log.warning("Pillow decode failed for %s: %s", path, exc)
        return None


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _decode_heic_with_sips(path: Path) -> np.ndarray | None:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / f"{path.stem}.png"
            subprocess.run(
                [FFMPEG_BIN, "-y", "-i", str(path), "-frames:v", "1", str(temp_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            img = cv2.imread(str(temp_path), cv2.IMREAD_COLOR)
            if img is None:
                log.warning("HEIC conversion succeeded but OpenCV could not read %s", path)
            return img
    except (OSError, subprocess.CalledProcessError) as exc:
        log.warning("Failed to decode %s via ffmpeg: %s", path, exc)
        return None


def read_image_dimensions(path: Path, decoded_img: np.ndarray | None = None) -> tuple[int, int] | None:
    """Return original image dimensions; HEIC metadata is read via sips when needed."""
    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        return _read_heic_dimensions(path)
    if decoded_img is not None:
        height, width = decoded_img.shape[:2]
        return width, height

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    height, width = img.shape[:2]
    return width, height


def _read_heic_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        log.warning("Failed to read HEIC metadata for %s: %s", path, exc)
        return None

    width = None
    height = None
    for line in result.stdout.splitlines():
        if "pixelWidth:" in line:
            width = int(line.split(":", 1)[1].strip())
        elif "pixelHeight:" in line:
            height = int(line.split(":", 1)[1].strip())

    if width is None or height is None:
        return None
    return width, height


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian — higher means sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure_clipping(gray: np.ndarray) -> tuple[float, float]:
    """Return (black_clip_%, white_clip_%) based on histogram bin counts."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    black_pct = float(hist[:5].sum() / total * 100)
    white_pct = float(hist[251:].sum() / total * 100)
    return black_pct, white_pct


def apply_filters(image: np.ndarray, cfg: PipelineConfig) -> FilterResult:
    """Run blur + exposure checks.

    Only clearly blurred images are rejected outright. Moderately soft images stay
    eligible and are handled later by duplicate selection via the quality score.
    """
    # Work on a down-scaled copy to save compute if the image is huge.
    h, w = image.shape[:2]
    max_dim = 1500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    lap_var = _laplacian_variance(gray)
    black_pct, white_pct = _exposure_clipping(gray)

    # ── Decision ──────────────────────────────────────────────────────
    if lap_var < cfg.hard_blur_threshold:
        reason = f"blur (Laplacian var {lap_var:.1f} < {cfg.hard_blur_threshold})"
        log.info("SKIP (blur) Laplacian var=%.1f", lap_var)
        return FilterResult(False, lap_var, black_pct, white_pct, reason)

    if black_pct > cfg.black_clip_pct:
        reason = f"underexposed (black clip {black_pct:.1f}% > {cfg.black_clip_pct}%)"
        log.info("SKIP (exposure) black clip=%.1f%%", black_pct)
        return FilterResult(False, lap_var, black_pct, white_pct, reason)

    if white_pct > cfg.white_clip_pct:
        reason = f"overexposed (white clip {white_pct:.1f}% > {cfg.white_clip_pct}%)"
        log.info("SKIP (exposure) white clip=%.1f%%", white_pct)
        return FilterResult(False, lap_var, black_pct, white_pct, reason)

    return FilterResult(True, lap_var, black_pct, white_pct)
