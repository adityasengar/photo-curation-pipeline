"""Pipeline configuration with sensible defaults for A4/A3 print curation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    # ── Google Drive ──────────────────────────────────────────────────
    folder_id: str = ""
    local_root: Path | None = None
    credentials_path: Path = Path("credentials.json")
    token_path: Path = Path("token.json")
    scopes: tuple[str, ...] = ("https://www.googleapis.com/auth/drive.readonly",)

    # ── Metadata filter ───────────────────────────────────────────────
    min_megapixels: float = 8.0  # 8 MP ≈ minimum for sharp A4 @ 300 DPI
    medium_min_megapixels: float = 1.0  # keep smaller images in a separate review bucket

    # ── OpenCV heuristics ─────────────────────────────────────────────
    hard_blur_threshold: float = 25.0      # only reject clearly blurred images outright
    laplacian_threshold: float = 80.0       # variance of Laplacian; lower → blurry
    black_clip_pct: float = 15.0            # % of pixels in the darkest 5 bins
    white_clip_pct: float = 15.0            # % of pixels in the brightest 5 bins

    # ── CLIP aesthetic scoring ────────────────────────────────────────
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    positive_prompt: str = "A high quality, professional, beautiful, sharp photograph"
    negative_prompt: str = "A blurry, ugly, amateur, poorly framed snapshot"
    clip_score_threshold: float = 0.55      # positive similarity must exceed this

    # ── Output ────────────────────────────────────────────────────────
    output_dir: Path = Path("Curated_For_Print")
    csv_path: Path = Path("Curated_For_Print/curation_log.csv")
    local_output_dirname: str = "Curated_Best"
    local_highres_dirname: str = "Above_8MP"
    local_mediumres_dirname: str = "Between_1MP_and_8MP"

    # ── Resource management ───────────────────────────────────────────
    page_size: int = 100                    # Drive API page size
    max_download_bytes: int = 100 * 1024 * 1024  # skip files > 100 MB
    similarity_threshold: float = 0.97
    duplicate_color_threshold: float = 0.14
    duplicate_hash_threshold: int = 24
    require_sequence_proximity: bool = True
    sequence_group_window: int = 20
    scene_family_window: int = 4
    scene_family_similarity_threshold: float = 0.93
    scene_family_color_threshold: float = 0.28
    scene_family_hash_threshold: int = 20
    singleton_hard_blur_threshold: float = 50.0
    singleton_min_quality_score: float = 2.75
    singleton_group_penalty: float = 0.6
    enable_background_people_check: bool = False
    background_people_hard_reject: bool = False
    max_background_faces: int = 0
    dominant_face_ratio: float = 0.05
    background_face_ratio: float = 0.015
    background_face_relative_ratio: float = 0.45
    background_people_penalty: float = 0.25

    @property
    def min_pixels(self) -> int:
        return int(self.min_megapixels * 1_000_000)
