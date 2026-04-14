"""Main pipeline orchestrator — wires the three filter stages together."""

from __future__ import annotations

import csv
import gc
import logging
import math
import re
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from config import PipelineConfig
from drive_client import DriveClient, DriveImage
from image_filters import decode_image, decode_image_path, read_image_dimensions, apply_filters, FilterResult
from aesthetic_scorer import AestheticScorer
from scene_semantics import BackgroundPeopleChecker, SceneAnalysis

log = logging.getLogger(__name__)


@dataclass
class CurationRecord:
    file_id: str
    filename: str
    megapixels: float
    laplacian_var: float
    black_clip_pct: float
    white_clip_pct: float
    clip_score: float
    saved_path: str
    source_path: str = ""
    group_id: int = 0
    quality_score: float = 0.0


@dataclass
class PipelineStats:
    total_listed: int = 0
    passed_metadata: int = 0
    passed_opencv: int = 0
    passed_clip: int = 0
    failed_decode: int = 0
    records: list[CurationRecord] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"\n{'═' * 50}\n"
            f"  Pipeline complete\n"
            f"  Images considered : {self.total_listed}\n"
            f"  Passed resolution : {self.passed_metadata}\n"
            f"  Passed OpenCV     : {self.passed_opencv}\n"
            f"  Passed CLIP       : {self.passed_clip}\n"
            f"  Decode failures   : {self.failed_decode}\n"
            f"  Saved to disk     : {len(self.records)}\n"
            f"{'═' * 50}"
        )


@dataclass
class LocalImageCandidate:
    path: Path
    filename: str
    megapixels: float
    resolution_tier: str
    img_vector: list[float]
    color_thumb: list[float]
    perceptual_hash: list[int]
    filt: FilterResult
    scene: SceneAnalysis | None
    clip_score: float
    passed_clip: bool
    quality_score: float


@dataclass
class LocalImageDecision:
    filename: str
    source_path: str
    extension: str
    resolution_tier: str = ""
    megapixels: float = 0.0
    decode_ok: bool = False
    passed_resolution: bool = False
    passed_opencv: bool = False
    passed_clip: bool = False
    group_id: int = 0
    group_size: int = 0
    scene_family_id: int = 0
    scene_family_size: int = 0
    selected_filename: str = ""
    selected_saved_path: str = ""
    selected_tier: str = ""
    selected_in_group: bool = False
    status: str = "pending"
    rejection_reason: str = ""
    laplacian_var: float = 0.0
    black_clip_pct: float = 0.0
    white_clip_pct: float = 0.0
    clip_score: float = 0.0
    quality_score: float = 0.0
    singleton_group: bool = False
    singleton_penalty_applied: float = 0.0
    copied_raw_path: str = ""
    matching_raw_found: bool = False
    face_count: int = 0
    background_face_count: int = 0
    dominant_face_ratio: float = 0.0
    scene_flagged: bool = False
    scene_penalty: float = 0.0
    passed_scene_check: bool = True
    scene_rejection_reason: str = ""


class CurationPipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._drive = DriveClient(cfg) if not cfg.local_root else None
        self._scorer: AestheticScorer | None = None  # lazy-loaded
        self._background_checker: BackgroundPeopleChecker | None = None
        self._clip_unavailable = False

    def _ensure_scorer(self) -> AestheticScorer:
        if self._scorer is None:
            self._scorer = AestheticScorer(self._cfg)
        return self._scorer

    def _ensure_background_checker(self) -> BackgroundPeopleChecker | None:
        if not self._cfg.enable_background_people_check:
            return None
        if self._background_checker is None:
            self._background_checker = BackgroundPeopleChecker(self._cfg)
        return self._background_checker

    def run(self) -> PipelineStats:
        if self._cfg.local_root:
            return self._run_local()
        return self._run_drive()

    def _run_drive(self) -> PipelineStats:
        stats = PipelineStats()
        output_dir = Path(self._cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log.info("Scanning Drive folder %s …", self._cfg.folder_id)

        assert self._drive is not None
        for image_meta in self._drive.list_eligible_images():
            stats.total_listed += 1
            stats.passed_metadata += 1
            log.info(
                "[%d] Processing %s (%.1f MP) …",
                stats.total_listed, image_meta.name, image_meta.megapixels,
            )

            # ── Stage 2: stream + OpenCV ──────────────────────────────
            t0 = time.perf_counter()
            raw_bytes = self._drive.stream_file(image_meta.file_id)
            img = decode_image(raw_bytes)
            del raw_bytes  # free the byte buffer
            gc.collect()

            if img is None:
                stats.failed_decode += 1
                continue

            filt = apply_filters(img, self._cfg)
            if not filt.passed:
                log.info("  ✗ OpenCV: %s", filt.rejection_reason)
                del img
                gc.collect()
                continue
            stats.passed_opencv += 1

            # ── Stage 3: CLIP aesthetic scoring ───────────────────────
            scorer = self._ensure_scorer()
            passed_clip, clip_score = scorer.passes(img)
            del img
            gc.collect()

            if not passed_clip:
                log.info("  ✗ CLIP: %.3f", clip_score)
                continue
            stats.passed_clip += 1

            # ── Stage 4: download original & log ──────────────────────
            saved_path = self._drive.download_original(image_meta.file_id, output_dir)
            elapsed = time.perf_counter() - t0

            record = CurationRecord(
                file_id=image_meta.file_id,
                filename=image_meta.name,
                megapixels=image_meta.megapixels,
                laplacian_var=round(filt.laplacian_var, 2),
                black_clip_pct=round(filt.black_clip_pct, 2),
                white_clip_pct=round(filt.white_clip_pct, 2),
                clip_score=round(clip_score, 4),
                saved_path=str(saved_path),
            )
            stats.records.append(record)
            log.info(
                "  ✓ KEPT  lap=%.1f  clip=%.3f  (%.1fs)",
                filt.laplacian_var, clip_score, elapsed,
            )

        # ── Write CSV ─────────────────────────────────────────────────
        self._write_csv(stats.records)
        return stats

    def _run_local(self) -> PipelineStats:
        stats = PipelineStats()
        assert self._cfg.local_root is not None

        for folder in self._discover_local_folders(self._cfg.local_root):
            log.info("Scanning local folder %s", folder)
            decisions: dict[str, LocalImageDecision] = {}
            candidates = self._build_local_candidates(folder, stats, decisions)
            if not candidates:
                log.info("No decodable images found in %s", folder)
                self._write_local_csv(output_dir=folder / self._cfg.local_output_dirname, decisions=decisions)
                continue

            scene_families = self._group_scene_families(candidates)
            for scene_family_id, family in enumerate(scene_families, start=1):
                for candidate in family:
                    decision = decisions[str(candidate.path)]
                    decision.scene_family_id = scene_family_id
                    decision.scene_family_size = len(family)

            groups = self._group_similar_images(candidates)
            output_dir = folder / self._cfg.local_output_dirname
            output_dir.mkdir(parents=True, exist_ok=True)
            highres_dir = output_dir / self._cfg.local_highres_dirname
            mediumres_dir = output_dir / self._cfg.local_mediumres_dirname
            highres_dir.mkdir(parents=True, exist_ok=True)
            mediumres_dir.mkdir(parents=True, exist_ok=True)

            selected: list[LocalImageCandidate] = []
            for group_id, group in enumerate(groups, start=1):
                for candidate in group:
                    decision = decisions[str(candidate.path)]
                    decision.group_id = group_id
                    decision.group_size = len(group)
                    decision.singleton_group = len(group) == 1

                eligible = [
                    c
                    for c in group
                    if decisions[str(c.path)].status in {"pending", "eligible", "selected"} and c.filt.passed
                ]
                if not eligible:
                    for candidate in group:
                        decision = decisions[str(candidate.path)]
                        decision.status = "rejected"
                        if not decision.rejection_reason:
                            decision.rejection_reason = "no_eligible_image_in_group"
                    continue

                best = max(
                    eligible,
                    key=lambda c: (
                        c.passed_clip,
                        c.quality_score,
                        c.megapixels,
                    ),
                )

                if len(group) == 1:
                    singleton_failures: list[str] = []
                    if best.filt.laplacian_var < self._cfg.singleton_hard_blur_threshold:
                        singleton_failures.append(
                            f"singleton_blur ({best.filt.laplacian_var:.1f} < {self._cfg.singleton_hard_blur_threshold})"
                        )
                    if best.quality_score < self._cfg.singleton_min_quality_score:
                        singleton_failures.append(
                            f"singleton_quality ({best.quality_score:.3f} < {self._cfg.singleton_min_quality_score})"
                        )

                    if singleton_failures:
                        decision = decisions[str(best.path)]
                        decision.status = "rejected"
                        decision.rejection_reason = "; ".join(singleton_failures)
                        continue

                    best = replace(
                        best,
                        quality_score=best.quality_score - self._cfg.singleton_group_penalty,
                    )
                    decision = decisions[str(best.path)]
                    decision.quality_score = round(best.quality_score, 4)
                    decision.singleton_penalty_applied = round(self._cfg.singleton_group_penalty, 4)

                selected.append((best, group, group_id))

            # ── keep_ratio trim ───────────────────────────────────────────
            # Sort all winners by quality score descending and keep only the
            # top fraction. Images trimmed here are marked rejected so the
            # CSV reflects the decision.
            if self._cfg.keep_ratio < 1.0 and selected:
                keep_n = max(1, math.ceil(len(selected) * self._cfg.keep_ratio))
                selected_sorted = sorted(selected, key=lambda t: t[0].quality_score, reverse=True)
                kept = set(id(t[0]) for t in selected_sorted[:keep_n])
                cutoff_score = selected_sorted[keep_n - 1][0].quality_score
                trimmed = [t for t in selected if id(t[0]) not in kept]
                for c, _, _gid in trimmed:
                    d = decisions[str(c.path)]
                    d.status = "rejected"
                    d.rejection_reason = (
                        f"keep_ratio_trim (score={c.quality_score:.3f}, cutoff={cutoff_score:.3f})"
                    )
                selected = [t for t in selected if id(t[0]) in kept]
                log.info(
                    "  keep_ratio=%.2f: kept %d of %d winners (cutoff score=%.3f)",
                    self._cfg.keep_ratio, len(selected), len(selected) + len(trimmed), cutoff_score,
                )

            for best, group, group_id in selected:
                winner_dir = self._tier_output_dir(best.resolution_tier, highres_dir, mediumres_dir)
                destination = self._copy_local_winner(best.path, winner_dir)
                raw_copy_path = self._copy_matching_raw(best.path, winner_dir)
                winner_decision = decisions[str(best.path)]
                winner_decision.selected_filename = best.filename
                winner_decision.selected_saved_path = str(destination)
                winner_decision.selected_tier = best.resolution_tier
                winner_decision.selected_in_group = True
                winner_decision.status = "selected"
                if raw_copy_path is not None:
                    winner_decision.copied_raw_path = str(raw_copy_path)
                    winner_decision.matching_raw_found = True

                for candidate in group:
                    decision = decisions[str(candidate.path)]
                    decision.selected_filename = best.filename
                    decision.selected_saved_path = str(destination)
                    decision.selected_tier = best.resolution_tier
                    if candidate.path == best.path:
                        continue
                    if decision.status == "pending":
                        decision.status = "rejected"
                    if decision.rejection_reason:
                        decision.rejection_reason = f"{decision.rejection_reason}; replaced by {best.filename}"
                    else:
                        decision.rejection_reason = f"similar_to_selected:{best.filename}"

                stats.records.append(
                    CurationRecord(
                        file_id="local",
                        filename=best.filename,
                        megapixels=best.megapixels,
                        laplacian_var=round(best.filt.laplacian_var, 2),
                        black_clip_pct=round(best.filt.black_clip_pct, 2),
                        white_clip_pct=round(best.filt.white_clip_pct, 2),
                        clip_score=round(best.clip_score, 4),
                        saved_path=str(destination),
                        source_path=str(best.path),
                        group_id=group_id,
                        quality_score=round(best.quality_score, 4),
                    )
                )
                log.info(
                    "  ✓ Selected group %d winner: %s (score=%.3f)",
                    group_id,
                    best.filename,
                    best.quality_score,
                )

            self._write_local_csv(output_dir=output_dir, decisions=decisions)
            log.info(
                "Saved %d winners from %s into %s",
                len(selected),
                folder.name,
                output_dir,
            )

        return stats

    def _discover_local_folders(self, root: Path) -> list[Path]:
        folders: list[Path] = []
        if self._contains_supported_images(root):
            folders.append(root)

        for child in sorted(root.iterdir()):
            if (
                child.is_dir()
                and not self._is_generated_output_dir(child)
                and self._contains_supported_images(child)
            ):
                folders.append(child)
        return folders

    def _is_generated_output_dir(self, folder: Path) -> bool:
        name = folder.name
        if name == self._cfg.local_output_dirname:
            return True
        return any(name.startswith(prefix) for prefix in GENERATED_OUTPUT_PREFIXES)

    def _contains_supported_images(self, folder: Path) -> bool:
        return any(p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES for p in folder.iterdir())

    def _build_local_candidates(
        self,
        folder: Path,
        stats: PipelineStats,
        decisions: dict[str, LocalImageDecision],
    ) -> list[LocalImageCandidate]:
        candidates: list[LocalImageCandidate] = []
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue

            decision = LocalImageDecision(
                filename=path.name,
                source_path=str(path),
                extension=path.suffix.lower(),
            )
            decisions[str(path)] = decision
            stats.total_listed += 1
            img = decode_image_path(path)
            if img is None:
                stats.failed_decode += 1
                decision.status = "rejected"
                decision.rejection_reason = "decode_failed"
                continue
            decision.decode_ok = True

            dimensions = read_image_dimensions(path, img)
            if dimensions is None:
                stats.failed_decode += 1
                decision.status = "rejected"
                decision.rejection_reason = "metadata_unavailable"
                del img
                gc.collect()
                continue

            width, height = dimensions
            megapixels = width * height / 1e6
            decision.megapixels = round(megapixels, 2)
            resolution_tier = self._resolution_tier(megapixels)
            decision.resolution_tier = resolution_tier
            if resolution_tier == "below_1mp":
                log.info(
                    "SKIP (resolution) %s — %.1f MP < %.1f MP",
                    path.name,
                    megapixels,
                    self._cfg.medium_min_megapixels,
                )
                decision.status = "rejected"
                decision.rejection_reason = "resolution_below_1mp"
                del img
                gc.collect()
                continue

            stats.passed_metadata += 1
            decision.passed_resolution = True
            filt = apply_filters(img, self._cfg)
            decision.laplacian_var = round(filt.laplacian_var, 2)
            decision.black_clip_pct = round(filt.black_clip_pct, 2)
            decision.white_clip_pct = round(filt.white_clip_pct, 2)
            if filt.passed:
                stats.passed_opencv += 1
                decision.passed_opencv = True
            else:
                decision.status = "rejected"
                decision.rejection_reason = filt.rejection_reason

            scene_analysis: SceneAnalysis | None = None
            if filt.passed:
                checker = self._ensure_background_checker()
                if checker is not None:
                    scene_analysis = checker.analyze(img)
                    decision.face_count = scene_analysis.face_count
                    decision.background_face_count = scene_analysis.background_face_count
                    decision.dominant_face_ratio = round(scene_analysis.dominant_face_ratio, 4)
                    decision.scene_flagged = scene_analysis.flagged
                    decision.scene_penalty = round(scene_analysis.penalty, 4)
                    decision.passed_scene_check = scene_analysis.passed
                    decision.scene_rejection_reason = scene_analysis.rejection_reason
                    if not scene_analysis.passed:
                        decision.status = "rejected"
                        decision.rejection_reason = scene_analysis.rejection_reason

            clip_score = 0.0
            passed_clip = False
            if filt.passed and (scene_analysis is None or scene_analysis.passed) and not self._clip_unavailable:
                try:
                    scorer = self._ensure_scorer()
                    passed_clip, clip_score = scorer.passes(img)
                    if passed_clip:
                        stats.passed_clip += 1
                        decision.passed_clip = True
                except Exception as exc:
                    self._clip_unavailable = True
                    log.warning("CLIP scoring unavailable for %s: %s", path.name, exc)

            quality_score = self._quality_score(
                megapixels,
                filt,
                clip_score,
                passed_clip,
                scene_analysis.penalty if scene_analysis is not None else 0.0,
            )
            decision.clip_score = round(clip_score, 4)
            decision.quality_score = round(quality_score, 4)
            if decision.status == "pending":
                decision.status = "eligible"
            vector = self._image_vector(img)
            color_thumb = self._color_thumb(img)
            perceptual_hash = self._perceptual_hash(img)
            del img
            gc.collect()

            candidates.append(
                LocalImageCandidate(
                    path=path,
                    filename=path.name,
                    megapixels=round(megapixels, 2),
                    resolution_tier=resolution_tier,
                    img_vector=vector,
                    color_thumb=color_thumb,
                    perceptual_hash=perceptual_hash,
                    filt=filt,
                    scene=scene_analysis,
                    clip_score=clip_score,
                    passed_clip=passed_clip,
                    quality_score=quality_score,
                )
            )
        return candidates

    def _group_similar_images(self, candidates: list[LocalImageCandidate]) -> list[list[LocalImageCandidate]]:
        return self._group_candidates(
            candidates,
            sequence_window=self._cfg.sequence_group_window,
            similarity_threshold=self._cfg.similarity_threshold,
            color_threshold=self._cfg.duplicate_color_threshold,
            hash_threshold=self._cfg.duplicate_hash_threshold,
        )

    def _group_scene_families(self, candidates: list[LocalImageCandidate]) -> list[list[LocalImageCandidate]]:
        return self._group_candidates(
            candidates,
            sequence_window=self._cfg.scene_family_window,
            similarity_threshold=self._cfg.scene_family_similarity_threshold,
            color_threshold=self._cfg.scene_family_color_threshold,
            hash_threshold=self._cfg.scene_family_hash_threshold,
        )

    def _group_candidates(
        self,
        candidates: list[LocalImageCandidate],
        *,
        sequence_window: int,
        similarity_threshold: float,
        color_threshold: float,
        hash_threshold: int,
    ) -> list[list[LocalImageCandidate]]:
        parent = list(range(len(candidates)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if not self._sequence_compatible(candidates[i].path, candidates[j].path, window=sequence_window):
                    continue
                similarity = self._cosine_similarity(candidates[i].img_vector, candidates[j].img_vector)
                color_distance = self._mean_abs_distance(
                    candidates[i].color_thumb,
                    candidates[j].color_thumb,
                )
                hash_distance = self._hamming_distance(
                    candidates[i].perceptual_hash,
                    candidates[j].perceptual_hash,
                )
                if (
                    similarity >= similarity_threshold
                    and color_distance <= color_threshold
                    and hash_distance <= hash_threshold
                ):
                    union(i, j)

        groups: dict[int, list[LocalImageCandidate]] = {}
        for idx, candidate in enumerate(candidates):
            groups.setdefault(find(idx), []).append(candidate)
        return list(groups.values())

    def _quality_score(
        self,
        megapixels: float,
        filt: FilterResult,
        clip_score: float,
        passed_clip: bool,
        scene_penalty: float,
    ) -> float:
        sharpness_score = min(filt.laplacian_var / max(self._cfg.laplacian_threshold, 1.0), 3.0)
        exposure_penalty = abs(filt.black_clip_pct - self._cfg.black_clip_pct / 2) / max(self._cfg.black_clip_pct, 1.0)
        exposure_penalty += abs(filt.white_clip_pct - self._cfg.white_clip_pct / 2) / max(self._cfg.white_clip_pct, 1.0)
        resolution_score = min(megapixels / max(self._cfg.min_megapixels, 1.0), 2.0)
        clip_bonus = clip_score if passed_clip else 0.0
        pass_bonus = 1.0 if filt.passed else 0.0
        return pass_bonus + sharpness_score + resolution_score + clip_bonus - exposure_penalty - scene_penalty

    def _image_vector(self, img) -> list[float]:
        gray = img
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        arr = thumb.astype("float32").reshape(-1)
        norm = math.sqrt(float((arr * arr).sum()))
        if norm == 0:
            return arr.tolist()
        return (arr / norm).tolist()

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _color_thumb(self, img) -> list[float]:
        thumb = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        arr = thumb.astype("float32").reshape(-1) / 255.0
        return arr.tolist()

    def _mean_abs_distance(self, a: list[float], b: list[float]) -> float:
        if not a:
            return 0.0
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    def _perceptual_hash(self, img) -> list[int]:
        gray = img
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")
        dct = cv2.dct(thumb)[:8, :8]
        median = float(np.median(dct[1:, :].reshape(-1)))
        return (dct > median).astype("uint8").reshape(-1).tolist()

    def _hamming_distance(self, a: list[int], b: list[int]) -> int:
        return sum(int(x != y) for x, y in zip(a, b))

    def _sequence_compatible(self, a: Path, b: Path, *, window: int | None = None) -> bool:
        if not self._cfg.require_sequence_proximity:
            return True

        parsed_a = self._parse_sequence_key(a.stem)
        parsed_b = self._parse_sequence_key(b.stem)
        if parsed_a is None or parsed_b is None:
            return False

        prefix_a, index_a = parsed_a
        prefix_b, index_b = parsed_b
        active_window = self._cfg.sequence_group_window if window is None else window
        return prefix_a == prefix_b and abs(index_a - index_b) <= active_window

    def _parse_sequence_key(self, stem: str) -> tuple[str, int] | None:
        match = re.match(r"^(.*?)(\d+)$", stem)
        if not match:
            return None
        prefix, index = match.groups()
        return prefix, int(index)

    def _copy_local_winner(self, source_path: Path, output_dir: Path) -> Path:
        destination = output_dir / source_path.name
        shutil.copy2(source_path, destination)
        return destination

    def _copy_matching_raw(self, source_path: Path, output_dir: Path) -> Path | None:
        if source_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            return None

        raw_candidates = [
            source_path.with_suffix(".RAF"),
            source_path.with_suffix(".raf"),
        ]
        raw_source = next((candidate for candidate in raw_candidates if candidate.exists()), None)
        if raw_source is None:
            return None

        raw_destination = output_dir / raw_source.name
        shutil.copy2(raw_source, raw_destination)
        return raw_destination

    def _resolution_tier(self, megapixels: float) -> str:
        if megapixels >= self._cfg.min_megapixels:
            return "above_8mp"
        if megapixels >= self._cfg.medium_min_megapixels:
            return "between_1mp_and_8mp"
        return "below_1mp"

    def _tier_output_dir(self, tier: str, highres_dir: Path, mediumres_dir: Path) -> Path:
        if tier == "above_8mp":
            return highres_dir
        return mediumres_dir

    def _write_local_csv(self, output_dir: Path, decisions: dict[str, LocalImageDecision]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "selection_log.csv"
        fieldnames = [
            "filename", "source_path", "extension", "resolution_tier", "status", "rejection_reason",
            "group_id", "group_size", "scene_family_id", "scene_family_size",
            "selected_in_group", "selected_filename", "selected_tier", "selected_saved_path",
            "decode_ok", "passed_resolution", "passed_opencv", "passed_clip", "megapixels",
            "laplacian_var", "black_clip_pct", "white_clip_pct",
            "singleton_group", "singleton_penalty_applied",
            "face_count", "background_face_count", "dominant_face_ratio", "passed_scene_check",
            "scene_flagged", "scene_penalty", "scene_rejection_reason",
            "clip_score", "quality_score", "matching_raw_found", "copied_raw_path", "saved_path",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for decision in sorted(decisions.values(), key=lambda d: d.filename):
                writer.writerow(
                    {
                        "filename": decision.filename,
                        "source_path": decision.source_path,
                        "extension": decision.extension,
                        "resolution_tier": decision.resolution_tier,
                        "status": decision.status,
                        "rejection_reason": decision.rejection_reason,
                        "group_id": decision.group_id,
                        "group_size": decision.group_size,
                        "scene_family_id": decision.scene_family_id,
                        "scene_family_size": decision.scene_family_size,
                        "selected_in_group": decision.selected_in_group,
                        "selected_filename": decision.selected_filename,
                        "selected_tier": decision.selected_tier,
                        "selected_saved_path": decision.selected_saved_path,
                        "decode_ok": decision.decode_ok,
                        "passed_resolution": decision.passed_resolution,
                        "passed_opencv": decision.passed_opencv,
                        "passed_clip": decision.passed_clip,
                        "megapixels": decision.megapixels,
                        "laplacian_var": decision.laplacian_var,
                        "black_clip_pct": decision.black_clip_pct,
                        "white_clip_pct": decision.white_clip_pct,
                        "singleton_group": decision.singleton_group,
                        "singleton_penalty_applied": decision.singleton_penalty_applied,
                        "face_count": decision.face_count,
                        "background_face_count": decision.background_face_count,
                        "dominant_face_ratio": decision.dominant_face_ratio,
                        "passed_scene_check": decision.passed_scene_check,
                        "scene_flagged": decision.scene_flagged,
                        "scene_penalty": decision.scene_penalty,
                        "scene_rejection_reason": decision.scene_rejection_reason,
                        "clip_score": decision.clip_score,
                        "quality_score": decision.quality_score,
                        "matching_raw_found": decision.matching_raw_found,
                        "copied_raw_path": decision.copied_raw_path,
                        "saved_path": decision.selected_saved_path,
                    }
                )
        log.info("CSV written → %s (%d rows)", csv_path, len(decisions))

    def _write_csv(self, records: list[CurationRecord]) -> None:
        if not records:
            log.info("No images survived curation — CSV not written.")
            return

        csv_path = Path(self._cfg.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "file_id", "filename", "megapixels",
            "laplacian_var", "black_clip_pct", "white_clip_pct",
            "clip_score", "saved_path",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r.__dict__)
        log.info("CSV written → %s (%d rows)", csv_path, len(records))


SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif",
}

GENERATED_OUTPUT_PREFIXES = (
    "Curated_Best",
    "Gemma_",
    "Stage2_",
)
