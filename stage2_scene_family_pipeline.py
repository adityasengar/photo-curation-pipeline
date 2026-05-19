#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import re
import shutil
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from image_filters import decode_image_path

RETRYABLE_OLLAMA_HTTP_CODES = {429, 500, 503, 504}


@dataclass
class SelectedPhoto:
    filename: str
    stage1_path: str
    source_path: str
    scene_family_id: int
    scene_family_size: int
    strict_group_id: int
    strict_group_size: int
    resolution_tier: str
    quality_score: float
    laplacian_var: float


@dataclass
class PhotoScore:
    filename: str
    path: str
    printworthy_score: int
    keepable: bool
    printworthy: bool
    category: str
    reason: str
    worth_enhancing: bool = False
    image_type: str = ""
    strengths: list[str] = field(default_factory=list)
    enhancement_goal: str = ""
    safe_edits: list[str] = field(default_factory=list)
    conditional_edits: list[str] = field(default_factory=list)
    avoid_edits: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)


def model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "1", "yes", "y"}:
            return True
        if cleaned in {"false", "0", "no", "n"}:
            return False
    return default


def normalize_string_list(value: Any, max_items: int = 8) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        return []

    normalized: list[str] = []
    seen = set()
    for item in raw_items:
        cleaned = str(item).strip()
        if not cleaned or cleaned in seen:
            continue
        normalized.append(cleaned)
        seen.add(cleaned)
        if len(normalized) >= max_items:
            break
    return normalized


def dedupe_preserve_order(names: list[str]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        deduped.append(name)
        seen.add(name)
    return deduped


def build_print_improvement_prompt(
    *,
    filename: str,
    image_type: str,
    strengths: list[str],
    enhancement_goal: str,
    safe_edits: list[str],
    conditional_edits: list[str],
    avoid_edits: list[str],
    risk_flags: list[str],
) -> str:
    def joined(items: list[str], fallback: str) -> str:
        return "; ".join(items) if items else fallback

    goal = enhancement_goal or "Make this photo feel cleaner, more polished, and print-ready while preserving the exact scene."
    img_type = image_type or "travel photograph"
    return (
        "Enhance this existing travel photograph into a high-quality print-ready image while preserving the exact scene and composition.\n\n"
        "Filename:\n"
        f"{filename}\n\n"
        "Preserve exactly:\n"
        "- the same framing, crop, perspective, orientation, subject placement, architecture layout, people, clothing, facial identity, artwork content, text, and scene geometry\n"
        "- the same emotional feel and photographic authenticity\n\n"
        "Do not:\n"
        "- add, remove, move, or replace people, objects, buildings, signs, reflections, clouds, or background elements\n"
        "- change identity, facial expression, body pose, clothing, or gaze direction\n"
        "- repaint, restyle, cartoonize, or create an artificial HDR look\n"
        "- invent fine detail in areas that are badly blurred or fully blown out\n"
        "- alter artwork content if this is a museum or art photo\n\n"
        "Goal:\n"
        f"{goal}\n\n"
        "Image type:\n"
        f"{img_type}\n\n"
        "Why this photo is strong:\n"
        f"{joined(strengths, 'Keep the strongest existing composition and mood.')}\n\n"
        "Safe edits to apply:\n"
        f"{joined(safe_edits, 'Subtle exposure balancing, natural white balance cleanup, gentle local contrast, realistic noise reduction, and mild print sharpening.')}\n\n"
        "Conditional edits only if they remain faithful to the original photo:\n"
        f"{joined(conditional_edits, 'If a sky is blown white, allow only plausible natural sky recovery. If the subject is awkwardly cut, allow only a very small and believable border expansion. If the frame is unintentionally tilted, allow only mild perspective or horizon correction.')}\n\n"
        "Edits to avoid:\n"
        f"{joined(avoid_edits, 'Do not change pose, identity, scene layout, text, architecture, or artwork content.')}\n\n"
        "Risk flags to respect:\n"
        f"{joined(risk_flags, 'Preserve authenticity and avoid over-processing.')}\n\n"
        "The result should look like the same photograph, just more refined, natural, and print-ready."
    )


def image_to_base64(path: Path) -> str:
    """Return a model-safe base64 image payload.

    We normalize all images to JPEG bytes before sending to Ollama so that
    mixed album inputs (JPG/PNG/HEIC) do not fail with 'image: unknown format'.
    """
    decoded = decode_image_path(path)
    if decoded is not None:
        ok, encoded = cv2.imencode(".jpg", decoded, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            return base64.b64encode(encoded.tobytes()).decode("ascii")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call_ollama(model: str, prompt: str, image_paths: list[Path], timeout: int) -> tuple[str, float]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": "json",
        "images": [image_to_base64(path) for path in image_paths],
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    attempts = 3
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            duration = time.perf_counter() - start
            return raw.get("response", "").strip(), duration
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in RETRYABLE_OLLAMA_HTTP_CODES and attempt < attempts:
                time.sleep(1.5 * attempt)
                last_error = RuntimeError(
                    f"Retryable Ollama HTTP {exc.code} on attempt {attempt}/{attempts}: {body[:500]}"
                )
                continue
            raise RuntimeError(
                f"Ollama request failed with HTTP {exc.code}: {body[:2000]}"
            ) from exc
        except urllib.error.URLError as exc:
            if attempt < attempts:
                time.sleep(1.5 * attempt)
                last_error = exc
                continue
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Ollama request failed without a captured exception")


def extract_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(1))


def build_family_prompt(image_names: list[str], max_keep: int) -> str:
    labeled = ", ".join(image_names)
    ordered = "; ".join(f"image {index + 1} is {name}" for index, name in enumerate(image_names))
    return (
        "You are reviewing a scene family from a travel photo album. "
        "These images come from the same moment or visual scene and may include near-duplicates, "
        "zoom variants, reframed shots, or slight expression/exposure differences. "
        "Your job is to keep only the strongest images from this family. "
        "Keep at most "
        f"{max_keep} "
        "images. Only keep more than one if they are genuinely distinct in framing, subject emphasis, "
        "emotional moment, or storytelling value. "
        "Reject weak, awkward, blurry, redundant, or throwaway frames. "
        "Prefer human-meaningful choices over purely technical sharpness. "
        "Do not penalize a small subject in frame if the composition is clearly intentional. "
        "Do not penalize shallow depth of field if deliberate. "
        f"The candidate filenames are: {labeled}. {ordered}. "
        "Return strict JSON with keys "
        "'keep_filenames', 'primary_filename', and 'photos'. "
        "'keep_filenames' must be an array of up to "
        f"{max_keep} "
        "filenames ranked best to worst among the kept images. "
        "'primary_filename' must be the single strongest image in the family. "
        "'photos' must be an array of objects with keys "
        "'filename', 'keep', 'distinct', and 'reason'."
    )


def build_portfolio_batch_prompt(image_names: list[str], top_per_batch: int) -> str:
    labeled = ", ".join(image_names)
    ordered = "; ".join(f"image {index + 1} is {name}" for index, name in enumerate(image_names))
    return (
        "You are a ruthless but fair gallery photo editor reviewing family-reduced travel photos after scene-family consolidation. "
        "These images are already stronger survivors, so be selective but not arbitrary. "
        "A photo can be printworthy either because it is a warm family image or because it is beautiful in general "
        "(city, nature, architecture, atmosphere, or composition). "
        "Prefer images that feel memorable, distinct, and worth printing or framing. "
        "Assume most images are keepable memories, but only a minority are truly printworthy. "
        "Reject ordinary, repetitive, awkward, or low-energy images from the top tier. "
        "For each photo, also suggest only faithful print-improvement guidance. "
        "You may allow gentle sky recovery if an overexposed white sky could plausibly look natural, "
        "minor border expansion if a subject is awkwardly cut and a very small outpaint would help, "
        "and mild perspective or horizon cleanup if the frame feels accidentally tilted. "
        "Do not suggest changing identity, facial expression, body pose, scene layout, architecture content, artwork content, or text. "
        f"The candidate filenames are: {labeled}. {ordered}. "
        "Return strict JSON with keys 'top_batch_filenames' and 'photos'. "
        "'top_batch_filenames' must be an array of up to "
        f"{top_per_batch} "
        "filenames ranked best to worst. "
        "'photos' must be an array of objects with keys "
        "'filename', 'printworthy_score', 'keepable', 'printworthy', 'category', 'reason', "
        "'worth_enhancing', 'image_type', 'strengths', 'enhancement_goal', "
        "'safe_edits', 'conditional_edits', 'avoid_edits', and 'risk_flags'. "
        "'printworthy_score' must be an integer from 1 to 10."
    )


def build_portfolio_final_prompt(image_names: list[str], top_n: int) -> str:
    labeled = ", ".join(image_names)
    ordered = "; ".join(f"image {index + 1} is {name}" for index, name in enumerate(image_names))
    return (
        "You are selecting the final ranked shortlist of printworthy travel photos from already family-reduced survivors. "
        "A photo can rank highly either as a beautiful family image or as a striking scenic or architectural travel image. "
        "Be highly selective. Prefer images that feel strong enough to print or frame. "
        "Avoid ranking images highly if they are repetitive, ordinary, awkward, accidentally bad, or noticeably weaker than the best finalists. "
        "If two images feel too similar, prefer the stronger one. "
        "For each ranked image, also provide faithful print-improvement guidance. "
        "You may suggest plausible sky recovery for a blown white sky, slight crop repair or minimal border expansion if the subject is awkwardly cut, "
        "and mild perspective cleanup if the frame looks unintentionally tilted. "
        "Do not suggest changing identity, expression, body pose, text, artwork content, scene geometry, or replacing major elements. "
        f"The finalist filenames are: {labeled}. {ordered}. "
        "Return strict JSON with keys 'ranked_filenames', 'top_filename', and 'reasons'. "
        "'ranked_filenames' must be an array of up to "
        f"{top_n} "
        "filenames ranked best to worst. "
        "'top_filename' must be the best overall filename. "
        "'reasons' must be an array of objects with keys "
        "'filename', 'reason', 'worth_enhancing', 'image_type', 'strengths', 'enhancement_goal', "
        "'safe_edits', 'conditional_edits', 'avoid_edits', and 'risk_flags'."
    )


def read_stage1_selected(stage1_dir: Path) -> tuple[list[SelectedPhoto], Path]:
    csv_path = stage1_dir / "selection_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Stage 1 CSV not found: {csv_path}")

    selected: list[SelectedPhoto] = []
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "selected":
                continue
            stage1_path = Path(row["selected_saved_path"]) if row["selected_saved_path"] else None
            if stage1_path is None or not stage1_path.exists():
                continue
            selected.append(
                SelectedPhoto(
                    filename=row["filename"],
                    stage1_path=str(stage1_path),
                    source_path=row["source_path"],
                    scene_family_id=int(row.get("scene_family_id") or 0),
                    scene_family_size=int(row.get("scene_family_size") or 0),
                    strict_group_id=int(row.get("group_id") or 0),
                    strict_group_size=int(row.get("group_size") or 0),
                    resolution_tier=row["resolution_tier"],
                    quality_score=float(row.get("quality_score") or 0.0),
                    laplacian_var=float(row.get("laplacian_var") or 0.0),
                )
            )
    return selected, csv_path


def read_family_survivors(family_dir: Path) -> tuple[list[SelectedPhoto], Path]:
    csv_path = family_dir / "family_reduction.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Family reduction CSV not found: {csv_path}")

    survivors: list[SelectedPhoto] = []
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            kept_value = str(row.get("kept_after_family_reduction", "")).strip().lower()
            if kept_value not in {"true", "1", "yes"}:
                continue
            stage1_path = Path(row["stage1_path"]) if row.get("stage1_path") else None
            if stage1_path is None or not stage1_path.exists():
                continue
            survivors.append(
                SelectedPhoto(
                    filename=row["filename"],
                    stage1_path=str(stage1_path),
                    source_path=row["stage1_path"],
                    scene_family_id=int(row.get("scene_family_id") or 0),
                    scene_family_size=int(row.get("family_input_count") or 0),
                    strict_group_id=int(row.get("strict_group_id") or 0),
                    strict_group_size=int(row.get("strict_group_size") or 0),
                    resolution_tier="",
                    quality_score=0.0,
                    laplacian_var=0.0,
                )
            )
    return survivors, csv_path


def normalize_family_result(
    parsed: dict[str, Any],
    family: list[SelectedPhoto],
    max_keep: int,
) -> tuple[list[str], str | None, list[dict[str, Any]]]:
    valid_names = {photo.filename for photo in family}
    keep_filenames = [name for name in parsed.get("keep_filenames", []) if name in valid_names]
    primary_filename = parsed.get("primary_filename")
    if primary_filename not in valid_names:
        primary_filename = None

    photo_entries_by_name: dict[str, dict[str, Any]] = {}
    for item in parsed.get("photos", []):
        filename = item.get("filename")
        if filename not in valid_names:
            continue
        photo_entries_by_name[filename] = {
            "filename": filename,
            "keep": bool(item.get("keep")),
            "distinct": bool(item.get("distinct")),
            "reason": str(item.get("reason", "")),
        }

    if not keep_filenames:
        keep_from_flags = [item["filename"] for item in photo_entries_by_name.values() if item["keep"]]
        if keep_from_flags:
            keep_filenames = [name for name in keep_from_flags if name in valid_names]

    if primary_filename and primary_filename not in keep_filenames:
        keep_filenames = [primary_filename] + keep_filenames

    if not keep_filenames:
        fallback = sorted(
            family,
            key=lambda photo: (photo.quality_score, photo.laplacian_var, photo.filename),
            reverse=True,
        )
        keep_filenames = [fallback[0].filename]
        primary_filename = fallback[0].filename

    deduped: list[str] = []
    seen = set()
    for name in keep_filenames:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    keep_filenames = deduped[:max_keep]

    if primary_filename is None and keep_filenames:
        primary_filename = keep_filenames[0]

    for photo in family:
        entry = photo_entries_by_name.setdefault(
            photo.filename,
            {
                "filename": photo.filename,
                "keep": False,
                "distinct": False,
                "reason": "",
            },
        )
        entry["keep"] = photo.filename in keep_filenames

    photo_entries = [photo_entries_by_name[photo.filename] for photo in family if photo.filename in photo_entries_by_name]
    return keep_filenames, primary_filename, photo_entries


def normalize_batch_result(parsed: dict[str, Any], batch_paths: list[Path]) -> tuple[list[str], list[PhotoScore]]:
    valid_names = {path.name: str(path) for path in batch_paths}
    top_names = [name for name in parsed.get("top_batch_filenames", []) if name in valid_names]
    top_names = dedupe_preserve_order(top_names)

    photos: list[PhotoScore] = []
    for item in parsed.get("photos", []):
        filename = item.get("filename")
        if filename not in valid_names:
            continue
        try:
            score = int(item.get("printworthy_score", 0))
        except Exception:
            score = 0
        photos.append(
            PhotoScore(
                filename=filename,
                path=valid_names[filename],
                printworthy_score=max(0, min(10, score)),
                keepable=coerce_bool(item.get("keepable"), default=True),
                printworthy=coerce_bool(item.get("printworthy"), default=False),
                category=str(item.get("category", "")),
                reason=str(item.get("reason", "")),
                worth_enhancing=coerce_bool(item.get("worth_enhancing"), default=coerce_bool(item.get("printworthy"), default=False)),
                image_type=str(item.get("image_type", item.get("category", ""))).strip(),
                strengths=normalize_string_list(item.get("strengths")),
                enhancement_goal=str(item.get("enhancement_goal", "")).strip(),
                safe_edits=normalize_string_list(item.get("safe_edits")),
                conditional_edits=normalize_string_list(item.get("conditional_edits")),
                avoid_edits=normalize_string_list(item.get("avoid_edits")),
                risk_flags=normalize_string_list(item.get("risk_flags")),
            )
        )

    if not top_names and photos:
        ranked = sorted(photos, key=lambda item: item.printworthy_score, reverse=True)
        top_names = dedupe_preserve_order([item.filename for item in ranked])

    return top_names, photos


def copy_images(image_paths: list[Path], output_dir: Path, prefix_rank: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for index, path in enumerate(image_paths, start=1):
        destination_name = f"{index:02d}_{path.name}" if prefix_rank else path.name
        destination = output_dir / destination_name
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def run_family_reduction(
    *,
    selected: list[SelectedPhoto],
    model: str,
    timeout: int,
    family_divisor: int,
    only_family_ids: list[int] | None,
    max_families: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir = output_dir / "decisions"
    images_dir = output_dir / "images"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    families: dict[int, list[SelectedPhoto]] = defaultdict(list)
    for photo in selected:
        families[photo.scene_family_id].append(photo)

    kept: list[SelectedPhoto] = []
    decision_summaries: list[dict[str, Any]] = []
    family_csv_rows: list[dict[str, Any]] = []

    family_ids = sorted(families)
    if only_family_ids is not None:
        requested = set(only_family_ids)
        family_ids = [family_id for family_id in family_ids if family_id in requested]
    if max_families is not None:
        family_ids = family_ids[:max_families]

    for family_id in family_ids:
        family = sorted(families[family_id], key=lambda photo: photo.filename)
        max_keep = max(1, math.ceil(len(family) / family_divisor))
        keep_filenames: list[str]
        primary_filename: str | None
        photo_entries: list[dict[str, Any]]
        raw_response = ""
        elapsed = 0.0
        mode = "model"

        if len(family) <= max_keep:
            keep_filenames = [photo.filename for photo in family]
            primary_filename = keep_filenames[0]
            photo_entries = [
                {
                    "filename": photo.filename,
                    "keep": True,
                    "distinct": len(family) > 1,
                    "reason": "auto_keep_family_size_within_limit",
                }
                for photo in family
            ]
            mode = "auto_keep"
        elif len(family) == 1:
            keep_filenames = [family[0].filename]
            primary_filename = family[0].filename
            photo_entries = [
                {
                    "filename": family[0].filename,
                    "keep": True,
                    "distinct": True,
                    "reason": "auto_keep_singleton_family",
                }
            ]
            mode = "auto_singleton"
        else:
            prompt = build_family_prompt([photo.filename for photo in family], max_keep)
            raw_response, elapsed = call_ollama(
                model=model,
                prompt=prompt,
                image_paths=[Path(photo.stage1_path) for photo in family],
                timeout=timeout,
            )
            parsed = extract_json(raw_response)
            keep_filenames, primary_filename, photo_entries = normalize_family_result(parsed, family, max_keep)

        keep_lookup = {photo.filename: photo for photo in family}
        family_kept = [keep_lookup[name] for name in keep_filenames if name in keep_lookup]
        kept.extend(family_kept)

        decision_summary = {
            "scene_family_id": family_id,
            "input_count": len(family),
            "scene_family_size_values": sorted({photo.scene_family_size for photo in family}),
            "strict_group_ids": sorted({photo.strict_group_id for photo in family}),
            "max_keep": max_keep,
            "keep_filenames": keep_filenames,
            "primary_filename": primary_filename,
            "mode": mode,
            "elapsed_seconds": round(elapsed, 2),
            "photos": photo_entries,
            "raw_response": raw_response,
        }
        decision_summaries.append(decision_summary)
        (decisions_dir / f"scene_family_{family_id:04d}.json").write_text(json.dumps(decision_summary, indent=2))

        photo_entry_by_name = {item["filename"]: item for item in photo_entries}
        for photo in family:
            entry = photo_entry_by_name.get(photo.filename, {})
            family_csv_rows.append(
                {
                    "scene_family_id": family_id,
                    "filename": photo.filename,
                    "kept_after_family_reduction": photo.filename in keep_filenames,
                    "primary_in_family": photo.filename == primary_filename,
                    "family_input_count": len(family),
                    "family_keep_limit": max_keep,
                    "strict_group_id": photo.strict_group_id,
                    "strict_group_size": photo.strict_group_size,
                    "stage1_path": photo.stage1_path,
                    "reason": entry.get("reason", ""),
                }
            )

    kept = sorted({photo.filename: photo for photo in kept}.values(), key=lambda photo: photo.filename)
    copied = copy_images([Path(photo.stage1_path) for photo in kept], images_dir)

    with (output_dir / "family_reduction.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene_family_id",
                "filename",
                "kept_after_family_reduction",
                "primary_in_family",
                "family_input_count",
                "family_keep_limit",
                "strict_group_id",
                "strict_group_size",
                "stage1_path",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(family_csv_rows)

    summary = {
        "model": model,
        "input_selected_count": len(selected),
        "family_count": len(families),
        "processed_family_count": len(family_ids),
        "kept_count": len(kept),
        "family_divisor": family_divisor,
        "only_family_ids": only_family_ids,
        "max_families": max_families,
        "images_dir": str(images_dir),
        "copied_images": [str(path) for path in copied],
        "family_decisions": decision_summaries,
    }
    (output_dir / "family_reduction_summary.json").write_text(json.dumps(summary, indent=2))
    return {"kept": kept, "summary": summary}


def run_portfolio_rerank(
    *,
    survivors: list[SelectedPhoto],
    model: str,
    timeout: int,
    portfolio_divisor: int,
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "ranked"
    images_dir.mkdir(parents=True, exist_ok=True)

    survivor_paths = [Path(photo.stage1_path) for photo in survivors]
    if not survivor_paths:
        raise ValueError("No family survivors available for portfolio rerank")

    top_n = max(1, math.ceil(len(survivor_paths) / portfolio_divisor))
    top_per_batch = max(1, math.ceil(batch_size / portfolio_divisor))
    batches = [survivor_paths[i:i + batch_size] for i in range(0, len(survivor_paths), batch_size)]

    all_scores: dict[str, PhotoScore] = {}
    finalists: list[str] = []
    batch_reports: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(batches, start=1):
        prompt = build_portfolio_batch_prompt([path.name for path in batch], top_per_batch)
        raw_response, elapsed = call_ollama(model=model, prompt=prompt, image_paths=batch, timeout=timeout)
        parsed = extract_json(raw_response)
        top_names, scores = normalize_batch_result(parsed, batch)

        finalists.extend(top_names[:top_per_batch])
        for score in scores:
            all_scores[score.filename] = score

        batch_report = {
            "batch_index": batch_index,
            "batch_size": len(batch),
            "elapsed_seconds": round(elapsed, 2),
            "top_batch_filenames": top_names[:top_per_batch],
            "photos": [asdict(score) for score in scores],
            "raw_response": raw_response,
        }
        batch_reports.append(batch_report)
        (output_dir / f"portfolio_batch_{batch_index:02d}.json").write_text(json.dumps(batch_report, indent=2))

    deduped_finalists: list[str] = []
    seen = set()
    for name in finalists:
        if name not in seen:
            deduped_finalists.append(name)
            seen.add(name)

    if len(deduped_finalists) < top_n:
        remaining = sorted(all_scores.values(), key=lambda item: item.printworthy_score, reverse=True)
        for score in remaining:
            if score.filename not in seen:
                deduped_finalists.append(score.filename)
                seen.add(score.filename)
            if len(deduped_finalists) >= max(top_n, top_per_batch * len(batches)):
                break

    finalist_paths = [Path(all_scores[name].path) for name in deduped_finalists if name in all_scores]
    final_prompt = build_portfolio_final_prompt([path.name for path in finalist_paths], top_n)
    final_raw, final_elapsed = call_ollama(model=model, prompt=final_prompt, image_paths=finalist_paths, timeout=timeout)
    final_parsed = extract_json(final_raw)

    ranked_names = [name for name in final_parsed.get("ranked_filenames", []) if name in all_scores]
    ranked_names = dedupe_preserve_order(ranked_names)[:top_n]
    if not ranked_names:
        ranked_names = dedupe_preserve_order([path.name for path in finalist_paths[:top_n]])
    top_name = final_parsed.get("top_filename")
    if top_name not in all_scores and ranked_names:
        top_name = ranked_names[0]
    reasons = final_parsed.get("reasons", [])

    final_reason_by_name: dict[str, dict[str, Any]] = {}
    for item in reasons:
        filename = item.get("filename")
        if filename not in all_scores:
            continue
        final_reason_by_name[filename] = {
            "reason": str(item.get("reason", "")),
            "worth_enhancing": coerce_bool(item.get("worth_enhancing"), default=all_scores[filename].worth_enhancing),
            "image_type": str(item.get("image_type", all_scores[filename].image_type)).strip(),
            "strengths": normalize_string_list(item.get("strengths")) or all_scores[filename].strengths,
            "enhancement_goal": str(item.get("enhancement_goal", all_scores[filename].enhancement_goal)).strip(),
            "safe_edits": normalize_string_list(item.get("safe_edits")) or all_scores[filename].safe_edits,
            "conditional_edits": normalize_string_list(item.get("conditional_edits")) or all_scores[filename].conditional_edits,
            "avoid_edits": normalize_string_list(item.get("avoid_edits")) or all_scores[filename].avoid_edits,
            "risk_flags": normalize_string_list(item.get("risk_flags")) or all_scores[filename].risk_flags,
        }

    ranked_paths = [Path(all_scores[name].path) for name in ranked_names if name in all_scores]
    copied_ranked = copy_images(ranked_paths, images_dir, prefix_rank=True)

    summary = {
        "model": model,
        "input_survivor_count": len(survivors),
        "portfolio_divisor": portfolio_divisor,
        "target_top_n": top_n,
        "batch_size": batch_size,
        "top_per_batch": top_per_batch,
        "batch_count": len(batches),
        "finalist_count": len(finalist_paths),
        "final_elapsed_seconds": round(final_elapsed, 2),
        "top_filename": top_name,
        "ranked_filenames": ranked_names,
        "ranked_details": [
            {
                "filename": name,
                "path": all_scores[name].path,
                "printworthy_score": all_scores[name].printworthy_score,
                "category": all_scores[name].category,
                "batch_reason": all_scores[name].reason,
                "final_reason": final_reason_by_name.get(name, {}).get("reason", ""),
                "worth_enhancing": final_reason_by_name.get(name, {}).get("worth_enhancing", all_scores[name].worth_enhancing),
                "image_type": final_reason_by_name.get(name, {}).get("image_type", all_scores[name].image_type),
                "strengths": final_reason_by_name.get(name, {}).get("strengths", all_scores[name].strengths),
                "enhancement_goal": final_reason_by_name.get(name, {}).get("enhancement_goal", all_scores[name].enhancement_goal),
                "safe_edits": final_reason_by_name.get(name, {}).get("safe_edits", all_scores[name].safe_edits),
                "conditional_edits": final_reason_by_name.get(name, {}).get("conditional_edits", all_scores[name].conditional_edits),
                "avoid_edits": final_reason_by_name.get(name, {}).get("avoid_edits", all_scores[name].avoid_edits),
                "risk_flags": final_reason_by_name.get(name, {}).get("risk_flags", all_scores[name].risk_flags),
                "print_improvement_prompt": build_print_improvement_prompt(
                    filename=name,
                    image_type=final_reason_by_name.get(name, {}).get("image_type", all_scores[name].image_type),
                    strengths=final_reason_by_name.get(name, {}).get("strengths", all_scores[name].strengths),
                    enhancement_goal=final_reason_by_name.get(name, {}).get("enhancement_goal", all_scores[name].enhancement_goal),
                    safe_edits=final_reason_by_name.get(name, {}).get("safe_edits", all_scores[name].safe_edits),
                    conditional_edits=final_reason_by_name.get(name, {}).get("conditional_edits", all_scores[name].conditional_edits),
                    avoid_edits=final_reason_by_name.get(name, {}).get("avoid_edits", all_scores[name].avoid_edits),
                    risk_flags=final_reason_by_name.get(name, {}).get("risk_flags", all_scores[name].risk_flags),
                ),
            }
            for name in ranked_names
            if name in all_scores
        ],
        "batch_reports": batch_reports,
        "final_raw_response": final_raw,
        "copied_ranked_images": [str(path) for path in copied_ranked],
    }
    (output_dir / "portfolio_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", help="Path to a Stage 1 output folder containing selection_log.csv")
    parser.add_argument("--resume-family-dir", help="Path to an existing family-reduction output folder; skips family reduction and reruns only portfolio ranking")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--family-divisor", type=int, default=4)
    parser.add_argument("--portfolio-divisor", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--scene-family-ids", help="Comma-separated scene_family_id values to process")
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--family-output-dir")
    parser.add_argument("--portfolio-output-dir")
    args = parser.parse_args()

    if not args.stage1_dir and not args.resume_family_dir:
        raise ValueError("Provide either --stage1-dir or --resume-family-dir")
    if args.stage1_dir and args.resume_family_dir:
        raise ValueError("Use only one of --stage1-dir or --resume-family-dir")

    stage1_dir = None
    resume_family_dir = None
    if args.stage1_dir:
        stage1_dir = Path(args.stage1_dir).expanduser().resolve()
        if not stage1_dir.exists():
            raise FileNotFoundError(stage1_dir)
    if args.resume_family_dir:
        resume_family_dir = Path(args.resume_family_dir).expanduser().resolve()
        if not resume_family_dir.exists():
            raise FileNotFoundError(resume_family_dir)
    only_family_ids = None
    if args.scene_family_ids:
        only_family_ids = [
            int(part.strip())
            for part in args.scene_family_ids.split(",")
            if part.strip()
        ]

    base_dir = stage1_dir if stage1_dir is not None else resume_family_dir
    assert base_dir is not None
    album_root = base_dir.parent
    slug = model_slug(args.model)
    family_output_dir = None
    if stage1_dir is not None:
        family_output_dir = (
            Path(args.family_output_dir).expanduser().resolve()
            if args.family_output_dir
            else album_root / f"Stage2_FamilyReduced_{slug}"
        )
    portfolio_output_dir = (
        Path(args.portfolio_output_dir).expanduser().resolve()
        if args.portfolio_output_dir
        else album_root / f"Stage2_PortfolioTop_{slug}"
    )

    if stage1_dir is not None:
        selected, csv_path = read_stage1_selected(stage1_dir)
        family_result = run_family_reduction(
            selected=selected,
            model=args.model,
            timeout=args.timeout,
            family_divisor=args.family_divisor,
            only_family_ids=only_family_ids,
            max_families=args.max_families,
            output_dir=family_output_dir,
        )
        family_output_dir_str = str(family_output_dir)
    else:
        selected, csv_path = read_family_survivors(resume_family_dir)
        family_result = {"kept": selected, "summary": {"processed_family_count": len(selected)}}
        family_output_dir_str = str(resume_family_dir)

    portfolio_summary = run_portfolio_rerank(
        survivors=family_result["kept"],
        model=args.model,
        timeout=args.timeout,
        portfolio_divisor=args.portfolio_divisor,
        batch_size=args.batch_size,
        output_dir=portfolio_output_dir,
    )

    result = {
        "model": args.model,
        "stage1_dir": str(stage1_dir) if stage1_dir is not None else None,
        "resume_family_dir": str(resume_family_dir) if resume_family_dir is not None else None,
        "stage1_csv": str(csv_path),
        "stage1_selected_count": len(selected),
        "family_output_dir": family_output_dir_str,
        "family_survivor_count": len(family_result["kept"]),
        "only_family_ids": only_family_ids,
        "processed_family_count": family_result["summary"]["processed_family_count"],
        "portfolio_output_dir": str(portfolio_output_dir),
        "final_top_count": len(portfolio_summary["ranked_filenames"]),
        "top_filename": portfolio_summary["top_filename"],
        "ranked_filenames": portfolio_summary["ranked_filenames"],
        "timestamp": datetime.now().isoformat(),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
