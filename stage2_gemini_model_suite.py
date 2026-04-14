#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import time
from time import sleep
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from stage2_scene_family_pipeline import (
    PhotoScore,
    SelectedPhoto,
    build_family_prompt,
    build_print_improvement_prompt,
    build_portfolio_batch_prompt,
    build_portfolio_final_prompt,
    dedupe_preserve_order,
    coerce_bool,
    copy_images,
    extract_json,
    model_slug,
    normalize_string_list,
    normalize_batch_result,
    normalize_family_result,
    read_stage1_selected,
)


GEMINI_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
RETRYABLE_HTTP_CODES = {429, 500, 503, 504}


def _family_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "keep_filenames": {
                "type": "array",
                "items": {"type": "string"},
            },
            "primary_filename": {"type": "string"},
            "photos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "keep": {"type": "boolean"},
                        "distinct": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["filename", "keep", "distinct", "reason"],
                },
            },
        },
        "required": ["keep_filenames", "primary_filename", "photos"],
    }


def _portfolio_batch_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "top_batch_filenames": {
                "type": "array",
                "items": {"type": "string"},
            },
            "photos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "printworthy_score": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "keepable": {"type": "boolean"},
                        "printworthy": {"type": "boolean"},
                        "category": {"type": "string"},
                        "reason": {"type": "string"},
                        "worth_enhancing": {"type": "boolean"},
                        "image_type": {"type": "string"},
                        "strengths": {"type": "array", "items": {"type": "string"}},
                        "enhancement_goal": {"type": "string"},
                        "safe_edits": {"type": "array", "items": {"type": "string"}},
                        "conditional_edits": {"type": "array", "items": {"type": "string"}},
                        "avoid_edits": {"type": "array", "items": {"type": "string"}},
                        "risk_flags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "filename",
                        "printworthy_score",
                        "keepable",
                        "printworthy",
                        "category",
                        "reason",
                        "worth_enhancing",
                        "image_type",
                        "strengths",
                        "enhancement_goal",
                        "safe_edits",
                        "conditional_edits",
                        "avoid_edits",
                        "risk_flags",
                    ],
                },
            },
        },
        "required": ["top_batch_filenames", "photos"],
    }


def _portfolio_final_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "ranked_filenames": {
                "type": "array",
                "items": {"type": "string"},
            },
            "top_filename": {"type": "string"},
            "reasons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "reason": {"type": "string"},
                        "worth_enhancing": {"type": "boolean"},
                        "image_type": {"type": "string"},
                        "strengths": {"type": "array", "items": {"type": "string"}},
                        "enhancement_goal": {"type": "string"},
                        "safe_edits": {"type": "array", "items": {"type": "string"}},
                        "conditional_edits": {"type": "array", "items": {"type": "string"}},
                        "avoid_edits": {"type": "array", "items": {"type": "string"}},
                        "risk_flags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "filename",
                        "reason",
                        "worth_enhancing",
                        "image_type",
                        "strengths",
                        "enhancement_goal",
                        "safe_edits",
                        "conditional_edits",
                        "avoid_edits",
                        "risk_flags",
                    ],
                },
            },
        },
        "required": ["ranked_filenames", "top_filename", "reasons"],
    }


def _load_api_key(env_name: str) -> str:
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {env_name} before running the Gemini suite."
        )
    return api_key


def _resize_for_review(
    path: Path,
    *,
    review_max_side: int,
    review_jpeg_quality: int,
    review_max_bytes: int,
) -> tuple[str, bytes, dict[str, Any]]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        original_width, original_height = image.size

        side_candidates = []
        side = review_max_side
        while side >= 768:
            side_candidates.append(side)
            side = int(side * 0.85)

        if min(image.size) < 768:
            side_candidates.append(max(image.size))

        quality_candidates = []
        quality = review_jpeg_quality
        while quality >= 60:
            quality_candidates.append(quality)
            quality -= 6

        best_bytes = b""
        best_info: dict[str, Any] = {}

        for max_side in side_candidates:
            candidate = image.copy()
            candidate.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            width, height = candidate.size

            for quality_value in quality_candidates:
                buffer = io.BytesIO()
                candidate.save(
                    buffer,
                    format="JPEG",
                    quality=quality_value,
                    optimize=True,
                )
                data = buffer.getvalue()
                best_bytes = data
                best_info = {
                    "source_path": str(path),
                    "original_width": original_width,
                    "original_height": original_height,
                    "review_width": width,
                    "review_height": height,
                    "review_jpeg_quality": quality_value,
                    "review_bytes": len(data),
                }
                if len(data) <= review_max_bytes:
                    return "image/jpeg", data, best_info

        return "image/jpeg", best_bytes, best_info


def _extract_text_response(response_json: dict[str, Any]) -> str:
    candidates = response_json.get("candidates") or []
    if not candidates:
        raise RuntimeError(
            f"Gemini returned no candidates: {json.dumps(response_json, indent=2)[:1000]}"
        )

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [part.get("text", "") for part in parts if part.get("text")]
    if texts:
        return "".join(texts).strip()

    prompt_feedback = response_json.get("promptFeedback")
    finish_reason = candidates[0].get("finishReason")
    raise RuntimeError(
        "Gemini returned no text output. "
        f"finish_reason={finish_reason!r} prompt_feedback={prompt_feedback!r}"
    )


def _call_gemini_json(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_paths: list[Path],
    timeout: int,
    response_schema: dict[str, Any],
    review_max_side: int,
    review_jpeg_quality: int,
    review_max_bytes: int,
) -> tuple[str, dict[str, Any], float, dict[str, Any], list[dict[str, Any]]]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    review_images: list[dict[str, Any]] = []

    for path in image_paths:
        mime_type, review_bytes, review_info = _resize_for_review(
            path,
            review_max_side=review_max_side,
            review_jpeg_quality=review_jpeg_quality,
            review_max_bytes=review_max_bytes,
        )
        review_images.append(review_info)
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(review_bytes).decode("ascii"),
                }
            }
        )

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseJsonSchema": response_schema,
        },
    }

    def do_request(request_payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        attempts = 3
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                GEMINI_ENDPOINT_TEMPLATE.format(model=model),
                data=json.dumps(request_payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                method="POST",
            )
            request_start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    response_json = json.loads(response.read().decode("utf-8"))
                request_duration = time.perf_counter() - request_start
                return response_json, request_duration
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in RETRYABLE_HTTP_CODES and attempt < attempts:
                    sleep(1.5 * attempt)
                    last_error = RuntimeError(
                        f"Retryable Gemini API error HTTP {exc.code}: {body[:1000]}"
                    )
                    continue
                raise RuntimeError(
                    f"Gemini API request failed with HTTP {exc.code}: {body[:2000]}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < attempts:
                    sleep(1.5 * attempt)
                    last_error = exc
                    continue
                raise RuntimeError(f"Gemini API request failed: {exc}") from exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini API request failed without a captured exception")

    used_plain_json_fallback = False
    try:
        raw_json, duration = do_request(payload)
    except RuntimeError as exc:
        message = str(exc)
        lowered = message.lower()
        if "http 400" in lowered and "json mode is not enabled" in lowered:
            fallback_payload = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0},
            }
            raw_json, duration = do_request(fallback_payload)
            used_plain_json_fallback = True
        else:
            raise

    response_text = _extract_text_response(raw_json)
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        parsed = extract_json(response_text)

    response_meta = {
        "model": model,
        "model_version": raw_json.get("modelVersion"),
        "usage_metadata": raw_json.get("usageMetadata", {}),
        "prompt_feedback": raw_json.get("promptFeedback", {}),
        "candidate_finish_reason": (raw_json.get("candidates") or [{}])[0].get("finishReason"),
        "used_plain_json_fallback": used_plain_json_fallback,
    }
    return response_text, parsed, duration, response_meta, review_images


def run_family_reduction(
    *,
    selected: list[SelectedPhoto],
    api_key: str,
    model: str,
    timeout: int,
    family_divisor: int,
    only_family_ids: list[int] | None,
    max_families: int | None,
    output_dir: Path,
    review_max_side: int,
    review_jpeg_quality: int,
    review_max_bytes: int,
    dry_run: bool,
) -> dict[str, Any]:
    families: dict[int, list[SelectedPhoto]] = {}
    for photo in selected:
        families.setdefault(photo.scene_family_id, []).append(photo)

    family_ids = sorted(families)
    if only_family_ids is not None:
        requested = set(only_family_ids)
        family_ids = [family_id for family_id in family_ids if family_id in requested]
    if max_families is not None:
        family_ids = family_ids[:max_families]

    if dry_run:
        return {
            "kept": [],
            "summary": {
                "model": model,
                "dry_run": True,
                "input_selected_count": len(selected),
                "family_count": len(families),
                "processed_family_count": len(family_ids),
                "family_divisor": family_divisor,
                "only_family_ids": only_family_ids,
                "max_families": max_families,
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir = output_dir / "decisions"
    images_dir = output_dir / "images"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    kept: list[SelectedPhoto] = []
    decision_summaries: list[dict[str, Any]] = []
    family_csv_rows: list[dict[str, Any]] = []

    for family_id in family_ids:
        family = sorted(families[family_id], key=lambda photo: photo.filename)
        max_keep = max(1, math.ceil(len(family) / family_divisor))
        keep_filenames: list[str]
        primary_filename: str | None
        photo_entries: list[dict[str, Any]]
        raw_response = ""
        elapsed = 0.0
        mode = "model"
        response_meta: dict[str, Any] = {}
        review_images: list[dict[str, Any]] = []

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
            raw_response, parsed, elapsed, response_meta, review_images = _call_gemini_json(
                api_key=api_key,
                model=model,
                prompt=prompt,
                image_paths=[Path(photo.stage1_path) for photo in family],
                timeout=timeout,
                response_schema=_family_schema(),
                review_max_side=review_max_side,
                review_jpeg_quality=review_jpeg_quality,
                review_max_bytes=review_max_bytes,
            )
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
            "gemini_response_meta": response_meta,
            "review_images": review_images,
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
    api_key: str,
    model: str,
    timeout: int,
    portfolio_divisor: int,
    batch_size: int,
    output_dir: Path,
    review_max_side: int,
    review_jpeg_quality: int,
    review_max_bytes: int,
    dry_run: bool,
) -> dict[str, Any]:
    survivor_paths = [Path(photo.stage1_path) for photo in survivors]
    if not survivor_paths:
        raise ValueError("No family survivors available for portfolio rerank")

    top_n = max(1, math.ceil(len(survivor_paths) / portfolio_divisor))
    top_per_batch = max(1, math.ceil(batch_size / portfolio_divisor))
    batches = [survivor_paths[i:i + batch_size] for i in range(0, len(survivor_paths), batch_size)]

    if dry_run:
        return {
            "model": model,
            "dry_run": True,
            "input_survivor_count": len(survivors),
            "portfolio_divisor": portfolio_divisor,
            "target_top_n": top_n,
            "batch_size": batch_size,
            "top_per_batch": top_per_batch,
            "batch_count": len(batches),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "ranked"
    images_dir.mkdir(parents=True, exist_ok=True)

    all_scores: dict[str, PhotoScore] = {}
    finalists: list[str] = []
    batch_reports: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(batches, start=1):
        prompt = build_portfolio_batch_prompt([path.name for path in batch], top_per_batch)
        raw_response, parsed, elapsed, response_meta, review_images = _call_gemini_json(
            api_key=api_key,
            model=model,
            prompt=prompt,
            image_paths=batch,
            timeout=timeout,
            response_schema=_portfolio_batch_schema(),
            review_max_side=review_max_side,
            review_jpeg_quality=review_jpeg_quality,
            review_max_bytes=review_max_bytes,
        )
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
            "gemini_response_meta": response_meta,
            "review_images": review_images,
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
    final_raw, final_parsed, final_elapsed, final_meta, final_review_images = _call_gemini_json(
        api_key=api_key,
        model=model,
        prompt=final_prompt,
        image_paths=finalist_paths,
        timeout=timeout,
        response_schema=_portfolio_final_schema(),
        review_max_side=review_max_side,
        review_jpeg_quality=review_jpeg_quality,
        review_max_bytes=review_max_bytes,
    )

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
        "final_gemini_response_meta": final_meta,
        "final_review_images": final_review_images,
        "copied_ranked_images": [str(path) for path in copied_ranked],
    }
    (output_dir / "portfolio_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1-dir",
        required=True,
        help="Path to a Stage 1 output folder containing selection_log.csv",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="Gemini model to run. Pass multiple times for a comparison suite.",
    )
    parser.add_argument("--family-divisor", type=int, default=4)
    parser.add_argument("--portfolio-divisor", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--scene-family-ids", help="Comma-separated scene_family_id values to process")
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--suite-output-dir")
    parser.add_argument(
        "--output-root-dir",
        help="Optional root directory for model-specific family and portfolio outputs. Defaults to the album parent folder.",
    )
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--review-max-side", type=int, default=1600)
    parser.add_argument("--review-jpeg-quality", type=int, default=84)
    parser.add_argument("--review-max-bytes", type=int, default=900000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and planned output structure without calling Gemini.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    stage1_dir = Path(args.stage1_dir).expanduser().resolve()
    if not stage1_dir.exists():
        raise FileNotFoundError(stage1_dir)

    only_family_ids = None
    if args.scene_family_ids:
        only_family_ids = [
            int(part.strip())
            for part in args.scene_family_ids.split(",")
            if part.strip()
        ]

    selected, csv_path = read_stage1_selected(stage1_dir)
    album_root = stage1_dir.parent
    output_root_dir = (
        Path(args.output_root_dir).expanduser().resolve()
        if args.output_root_dir
        else album_root
    )
    suite_output_dir = (
        Path(args.suite_output_dir).expanduser().resolve()
        if args.suite_output_dir
        else output_root_dir / "Stage2_GeminiModelSuite"
    )

    api_key = ""
    if not args.dry_run:
        api_key = _load_api_key(args.api_key_env)

    suite_runs: list[dict[str, object]] = []

    for model in args.models:
        slug = model_slug(model)
        family_output_dir = output_root_dir / f"Stage2_FamilyReduced_{slug}"
        portfolio_output_dir = output_root_dir / f"Stage2_PortfolioTop_{slug}"

        family_result = run_family_reduction(
            selected=selected,
            api_key=api_key,
            model=model,
            timeout=args.timeout,
            family_divisor=args.family_divisor,
            only_family_ids=only_family_ids,
            max_families=args.max_families,
            output_dir=family_output_dir,
            review_max_side=args.review_max_side,
            review_jpeg_quality=args.review_jpeg_quality,
            review_max_bytes=args.review_max_bytes,
            dry_run=args.dry_run,
        )

        portfolio_summary: dict[str, Any] | None = None
        if not args.dry_run:
            portfolio_summary = run_portfolio_rerank(
                survivors=family_result["kept"],
                api_key=api_key,
                model=model,
                timeout=args.timeout,
                portfolio_divisor=args.portfolio_divisor,
                batch_size=args.batch_size,
                output_dir=portfolio_output_dir,
                review_max_side=args.review_max_side,
                review_jpeg_quality=args.review_jpeg_quality,
                review_max_bytes=args.review_max_bytes,
                dry_run=False,
            )

        suite_runs.append(
            {
                "model": model,
                "stage1_dir": str(stage1_dir),
                "stage1_csv": str(csv_path),
                "stage1_selected_count": len(selected),
                "family_output_dir": str(family_output_dir),
                "family_survivor_count": len(family_result.get("kept", [])),
                "only_family_ids": only_family_ids,
                "processed_family_count": family_result["summary"]["processed_family_count"],
                "portfolio_output_dir": str(portfolio_output_dir),
                "final_top_count": len(portfolio_summary["ranked_filenames"]) if portfolio_summary else 0,
                "top_filename": portfolio_summary["top_filename"] if portfolio_summary else None,
                "ranked_filenames": portfolio_summary["ranked_filenames"] if portfolio_summary else [],
                "dry_run": args.dry_run,
            }
        )

    suite_output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "stage1_dir": str(stage1_dir),
        "stage1_csv": str(csv_path),
        "output_root_dir": str(output_root_dir),
        "models": args.models,
        "family_divisor": args.family_divisor,
        "portfolio_divisor": args.portfolio_divisor,
        "batch_size": args.batch_size,
        "scene_family_ids": only_family_ids,
        "max_families": args.max_families,
        "timeout": args.timeout,
        "api_key_env": args.api_key_env,
        "review_max_side": args.review_max_side,
        "review_jpeg_quality": args.review_jpeg_quality,
        "review_max_bytes": args.review_max_bytes,
        "dry_run": args.dry_run,
        "runs": suite_runs,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = suite_output_dir / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
