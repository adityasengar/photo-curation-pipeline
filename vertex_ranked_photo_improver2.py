#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path
from time import sleep
from typing import Any
import urllib.error
import urllib.request

import google.auth
from google.auth.transport.requests import Request
from PIL import Image, ImageOps


VERTEX_ENDPOINT_TEMPLATE = (
    "https://{service_endpoint}/v1/"
    "projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent"
)
DEFAULT_MODEL = "gemini-3-pro-image-preview"
DEFAULT_ANALYSIS_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "global"
DEFAULT_OUTPUT_DIR_NAME = "ranked_improved_vertex2_unified_full_final"
DEFAULT_ADC_PATH = Path("/tmp/gcloud/application_default_credentials.json")
RETRYABLE_HTTP_CODES = {429, 500, 503, 504}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improve ranked travel photos using Vertex AI Gemini image models.",
    )
    parser.add_argument(
        "--portfolio-dir",
        help=(
            "Path to a portfolio folder containing ranked/ and portfolio_summary.json. "
            "If omitted, the script auto-discovers a suitable Stage2 portfolio folder."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Vertex Gemini image model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--analysis-model",
        default=DEFAULT_ANALYSIS_MODEL,
        help=f"Vertex Gemini model used for the text-only repair analysis step (default: {DEFAULT_ANALYSIS_MODEL})",
    )
    parser.add_argument(
        "--project-id",
        help="Google Cloud project ID for Vertex AI. If omitted, inferred from ADC or environment.",
    )
    parser.add_argument(
        "--location",
        default=DEFAULT_LOCATION,
        help=f"Vertex location (default: {DEFAULT_LOCATION})",
    )
    parser.add_argument(
        "--output-dir-name",
        default=DEFAULT_OUTPUT_DIR_NAME,
        help=f"Name of the output folder created inside the portfolio directory (default: {DEFAULT_OUTPUT_DIR_NAME})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only improve the top N ranked images",
    )
    parser.add_argument(
        "--start-rank",
        type=int,
        default=1,
        help="1-based rank to start from (default: 1)",
    )
    parser.add_argument(
        "--input-max-side",
        type=int,
        default=4096,
        help="Resize long side of input image to at most this size before sending to Vertex (default: 4096)",
    )
    parser.add_argument(
        "--input-jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for resized Vertex input images (default: 90)",
    )
    parser.add_argument(
        "--image-size",
        default="4K",
        help="Requested Vertex output image size, e.g. 2K or 4K (default: 4K)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-request timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--pad-percent",
        type=int,
        default=10,
        help="Pad each edge by this %% of the image dimension with soft blur-fade margins (default: 10, 0 to disable)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing improved outputs if they already exist",
    )
    return parser.parse_args()


def _looks_like_portfolio_dir(path: Path) -> bool:
    return path.is_dir() and (path / "ranked").is_dir() and (path / "portfolio_summary.json").is_file()


def resolve_portfolio_dir(portfolio_dir_arg: str | None) -> Path:
    if portfolio_dir_arg:
        path = Path(portfolio_dir_arg).expanduser().resolve()
        if not _looks_like_portfolio_dir(path):
            raise FileNotFoundError(
                f"Invalid portfolio folder: {path}. Expected ranked/ and portfolio_summary.json."
            )
        return path

    env_portfolio = os.environ.get("VERTEX_PORTFOLIO_DIR", "").strip()
    if env_portfolio:
        path = Path(env_portfolio).expanduser().resolve()
        if _looks_like_portfolio_dir(path):
            return path

    cwd = Path.cwd().resolve()
    if _looks_like_portfolio_dir(cwd):
        return cwd

    candidates: list[Path] = []
    search_roots = [cwd, cwd.parent]
    patterns = [
        "Stage2_PortfolioTop*",
        "*Stage2*PortfolioTop*",
        "*printplans*",
    ]
    for root in search_roots:
        for pattern in patterns:
            for found in root.glob(pattern):
                if _looks_like_portfolio_dir(found):
                    candidates.append(found.resolve())

    if not candidates:
        for child in cwd.iterdir():
            if _looks_like_portfolio_dir(child):
                candidates.append(child.resolve())

    if not candidates:
        cloud_root = Path.home() / "Library" / "CloudStorage"
        if cloud_root.exists():
            for drive_root in cloud_root.glob("GoogleDrive-*"):
                pictures_root = drive_root / "My Drive" / "Pictures"
                if not pictures_root.exists():
                    continue
                for album_dir in pictures_root.iterdir():
                    if not album_dir.is_dir():
                        continue
                    for found in album_dir.glob("Stage2_PortfolioTop*"):
                        if _looks_like_portfolio_dir(found):
                            candidates.append(found.resolve())

    if candidates:
        # Pick the most recently touched candidate so reruns default to latest pipeline output.
        return max(set(candidates), key=lambda path: path.stat().st_mtime)

    raise FileNotFoundError(
        "Could not auto-discover a portfolio folder. "
        "Pass --portfolio-dir or set VERTEX_PORTFOLIO_DIR."
    )


def _try_set_default_adc_env() -> None:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip():
        return

    fallback_paths = [
        DEFAULT_ADC_PATH,
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json",
    ]
    for candidate in fallback_paths:
        if candidate.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate)
            return


def _infer_quota_project_from_adc() -> str:
    adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not adc_path:
        return ""
    try:
        payload = json.loads(Path(adc_path).read_text())
    except Exception:
        return ""
    return str(payload.get("quota_project_id", "")).strip()


def load_vertex_credentials(project_id_arg: str | None) -> tuple[Any, str]:
    _try_set_default_adc_env()
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials, discovered_project = google.auth.default(scopes=scopes)
    request = Request()
    credentials.refresh(request)

    project_id = (
        project_id_arg
        or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        or os.environ.get("GCLOUD_PROJECT", "").strip()
        or _infer_quota_project_from_adc()
        or discovered_project
    )
    if not project_id:
        raise RuntimeError(
            "Could not determine a Google Cloud project ID. "
            "Pass --project-id or set GOOGLE_CLOUD_PROJECT."
        )
    return credentials, project_id


def auth_header(credentials: Any) -> dict[str, str]:
    if not credentials.valid:
        credentials.refresh(Request())
    return {"Authorization": f"Bearer {credentials.token}"}


def load_portfolio_summary(portfolio_dir: Path) -> dict[str, Any]:
    summary_path = portfolio_dir / "portfolio_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing portfolio summary: {summary_path}")
    return json.loads(summary_path.read_text())


def choose_ranked_images(portfolio_dir: Path, start_rank: int, limit: int | None) -> list[Path]:
    ranked_dir = portfolio_dir / "ranked"
    if not ranked_dir.exists():
        raise FileNotFoundError(f"Missing ranked folder: {ranked_dir}")

    ranked_images = sorted(
        path for path in ranked_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    )
    if start_rank < 1:
        raise ValueError("--start-rank must be >= 1")

    start_index = start_rank - 1
    selected = ranked_images[start_index:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_reason_lookup(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in summary.get("ranked_details", []):
        filename = item.get("filename")
        if filename:
            lookup[str(filename)] = item
    return lookup


def pad_image_with_blur_fade(image: Image.Image, pad_percent: int) -> Image.Image:
    """Pad each edge with a blurred fade of the border region.

    Instead of a crisp mirror (which models interpret as real content),
    this creates a soft, progressively blurred extension of the edge
    pixels. The model sees it as "needs to be filled in" rather than
    as additional scene content.
    """
    if pad_percent <= 0:
        return image

    from PIL import ImageFilter
    import numpy as np

    w, h = image.size
    pad_x = max(1, int(w * pad_percent / 100))
    pad_y = max(1, int(h * pad_percent / 100))

    new_w = w + 2 * pad_x
    new_h = h + 2 * pad_y

    # Start with the image pasted in the centre on a neutral background.
    # Compute average colour of the full image for the base fill.
    img_arr = np.array(image)
    avg_color = tuple(int(c) for c in img_arr.mean(axis=(0, 1)))
    padded = Image.new("RGB", (new_w, new_h), avg_color)
    padded.paste(image, (pad_x, pad_y))

    # Create a heavily blurred version of the padded image and composite
    # only the margin regions from it, so edges fade softly.
    blurred = padded.filter(ImageFilter.GaussianBlur(radius=max(pad_x, pad_y)))

    # Build an alpha mask: 255 in the original region, fading to 0 at edges.
    mask = Image.new("L", (new_w, new_h), 0)
    mask.paste(255, (pad_x, pad_y, pad_x + w, pad_y + h))
    # Blur the mask to create a soft transition.
    mask = mask.filter(ImageFilter.GaussianBlur(radius=min(pad_x, pad_y) * 0.8))

    # Composite: where mask is 255 keep original, where 0 keep blurred.
    result = Image.composite(padded, blurred, mask)
    return result


def _load_image_any(path: Path) -> "Image.Image":
    """Open an image file, with HEIC/HEIF support via pillow-heif."""
    if path.suffix.lower() in {".heic", ".heif"}:
        import pillow_heif
        pillow_heif.register_heif_opener()
    return Image.open(path)


def resize_input_image(
    path: Path,
    *,
    input_max_side: int,
    input_jpeg_quality: int,
    pad_percent: int = 0,
    pad_sides: list[str] | None = None,
) -> tuple[str, bytes, dict[str, Any], tuple[int, int]]:
    image = _load_image_any(path)
    with image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        original_width, original_height = image.size
        original_size = image.size

        # Directional padding if specific sides requested, else full padding.
        if pad_sides:
            padded = pad_image_directional(image, pad_sides, pad_percent)
        elif pad_percent > 0:
            padded = pad_image_with_blur_fade(image, pad_percent)
        else:
            padded = image

        if max(padded.size) > input_max_side:
            padded.thumbnail((input_max_side, input_max_side), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        padded.save(
            buffer,
            format="JPEG",
            quality=input_jpeg_quality,
            optimize=True,
        )
        data = buffer.getvalue()
        info = {
            "source_path": str(path),
            "original_width": original_width,
            "original_height": original_height,
            "padded": pad_percent > 0,
            "pad_percent": pad_percent,
            "input_width": padded.size[0],
            "input_height": padded.size[1],
            "input_jpeg_quality": input_jpeg_quality,
            "input_bytes": len(data),
        }
        return "image/jpeg", data, info, original_size


def aspect_ratio_string(size: tuple[int, int]) -> str:
    width, height = size
    if width <= 0 or height <= 0:
        return "1:1"

    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    divisor = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


ANALYSIS_PROMPT = """\
You are analyzing a travel photograph before it is sent for print enhancement.

Look at the image and answer in strict JSON (no markdown fencing):
{
  "repair_mode": "<one of: edge_completion, square_and_level, crowd_cleanup, subject_deblur, tonal_only>",
  "pad_sides": ["<side>", ...],
  "reason": "<one sentence>"
}

Rules for repair_mode:
- edge_completion: a symmetric architectural motif (window, arch, column) is visibly clipped at one or more edges and the missing part is obvious from the intact counterpart
- square_and_level: a framed artwork, facade, or corridor is slightly tilted/skewed
- crowd_cleanup: incidental people block the real subject (architecture, artwork, vista)
- subject_deblur: an intended foreground person is softer than the background
- tonal_only: none of the above clearly applies

Mode selection guidance:
- Choose the single strongest issue that would most improve the print if fixed first.
- Do NOT choose tonal_only until you have ruled out the other four modes.
- If the image is primarily about architecture, artwork, or a public space and nearby people distract from readability, choose crowd_cleanup.
- If a framed relief, corridor, facade, or wall artwork feels slightly tilted or not square, choose square_and_level.
- If a person is clearly the intended subject and is visibly softer than the rest of the frame, choose subject_deblur.
- If there is a single dominant foreground person who appears intentionally included with a landmark or artwork behind them, treat that person as a preserved subject, not as crowd clutter.
- In that situation, prefer subject_deblur or tonal_only; do NOT choose crowd_cleanup unless there are additional clearly incidental bystanders that are the real distraction.
- A single travel companion, family member, or portrait-like foreground figure should never be removed just to reveal more of the landmark.
- If one edge of a repeated architectural motif is cut off and the continuation is obvious, choose edge_completion even if tonal issues are also present.
- If an architecture image has repeated arches, windows, finials, or decorative borders that visibly terminate at the frame edge, prefer edge_completion over tonal_only.
- Slightly dark exposure is not a reason to choose tonal_only when clipped edge geometry is present.
- For bilateral motifs, if one side or top detail is truncated and the counterpart is visible, choose edge_completion.
- Prefer crowd_cleanup over square_and_level when incidental people are the bigger distraction.
- Prefer subject_deblur over tonal_only when a visible person is the intended subject and facial/clothing contours are soft.
- Prefer subject_deblur over crowd_cleanup when one preserved foreground person is the main human subject and only minor local clarity recovery is needed.

Rules for pad_sides:
- List which sides need canvas extension to complete clipped content: "left", "right", "top", "bottom"
- Only include sides where real content is obviously cut off
- If repair_mode is NOT edge_completion, pad_sides must be []
- Do not return all four sides unless the scene truly appears accidentally cropped on every edge (rare). If unsure, prefer square_and_level or tonal_only.
"""


def analyze_image_for_repair(
    *,
    image_bytes: bytes,
    mime_type: str,
    credentials: Any,
    project_id: str,
    location: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    """Cheap text-only call to decide repair mode and which sides need padding."""
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": ANALYSIS_PROMPT},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT"],
        },
    }
    response_json, _ = request_with_retries(
        credentials=credentials,
        project_id=project_id,
        location=location,
        model=model,
        payload=payload,
        timeout=timeout,
    )
    # Extract text from response.
    candidates = response_json.get("candidates") or []
    parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
    text = "".join(str(p.get("text", "")) for p in parts).strip()

    # Parse JSON from the response (strip markdown fencing if present).
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        analysis = json.loads(text)
    except json.JSONDecodeError:
        analysis = {"repair_mode": "tonal_only", "pad_sides": [], "reason": f"Failed to parse: {text[:200]}"}

    # Validate and normalise.
    valid_modes = {"edge_completion", "square_and_level", "crowd_cleanup", "subject_deblur", "tonal_only"}
    if analysis.get("repair_mode") not in valid_modes:
        analysis["repair_mode"] = "tonal_only"
    valid_sides = {"left", "right", "top", "bottom"}
    analysis["pad_sides"] = [s for s in analysis.get("pad_sides", []) if s in valid_sides]
    if analysis["repair_mode"] != "edge_completion":
        analysis["pad_sides"] = []
    elif len(analysis["pad_sides"]) == 4:
        # "All sides clipped" is usually a misclassification for centered architectural photos.
        analysis["repair_mode"] = "square_and_level"
        analysis["pad_sides"] = []
    elif "left" not in analysis["pad_sides"] and "right" not in analysis["pad_sides"]:
        # Top/bottom-only extension is often a framing/leveling issue, not true edge completion.
        analysis["repair_mode"] = "square_and_level"
        analysis["pad_sides"] = []

    return analysis


def pad_image_directional(image: Image.Image, pad_sides: list[str], pad_percent: int) -> Image.Image:
    """Pad only the specified sides with a blurred fade."""
    if not pad_sides or pad_percent <= 0:
        return image

    from PIL import ImageFilter
    import numpy as np

    w, h = image.size
    pad_x = max(1, int(w * pad_percent / 100))
    pad_y = max(1, int(h * pad_percent / 100))

    left_pad = pad_x if "left" in pad_sides else 0
    right_pad = pad_x if "right" in pad_sides else 0
    top_pad = pad_y if "top" in pad_sides else 0
    bottom_pad = pad_y if "bottom" in pad_sides else 0

    new_w = w + left_pad + right_pad
    new_h = h + top_pad + bottom_pad

    img_arr = np.array(image)
    avg_color = tuple(int(c) for c in img_arr.mean(axis=(0, 1)))
    padded = Image.new("RGB", (new_w, new_h), avg_color)
    padded.paste(image, (left_pad, top_pad))

    blurred = padded.filter(ImageFilter.GaussianBlur(radius=max(pad_x, pad_y)))

    mask = Image.new("L", (new_w, new_h), 0)
    mask.paste(255, (left_pad, top_pad, left_pad + w, top_pad + h))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=min(pad_x, pad_y) * 0.8))

    return Image.composite(padded, blurred, mask)


def build_improvement_prompt(
    ranked_filename: str,
    detail: dict[str, Any] | None,
    analysis: dict[str, Any] | None = None,
) -> str:
    # Extract per-image metadata when available.
    safe_edits: list[str] = []
    conditional_edits: list[str] = []
    avoid_edits: list[str] = []
    risk_flags: list[str] = []
    enhancement_goal = ""

    # Keywords that indicate a rotation instruction from the prior ranking model.
    # Rotation must be done in Python (already handled via exif_transpose), never
    # delegated to the generative model — it causes hallucination of subjects.
    _rotation_keywords = {"rotat", "turn 90", "flip", "orient"}

    def _is_rotation_instruction(text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in _rotation_keywords)

    if detail:
        enhancement_goal = str(detail.get("enhancement_goal", "")).strip()
        safe_edits = [
            str(item).strip() for item in detail.get("safe_edits", [])
            if str(item).strip() and not _is_rotation_instruction(str(item))
        ]
        conditional_edits = [
            str(item).strip() for item in detail.get("conditional_edits", [])
            if str(item).strip() and not _is_rotation_instruction(str(item))
        ]
        avoid_edits = [str(item).strip() for item in detail.get("avoid_edits", []) if str(item).strip()]
        risk_flags = [str(item).strip() for item in detail.get("risk_flags", []) if str(item).strip()]

    def joined_unique(items: list[str], fallback: str) -> str:
        ordered = list(dict.fromkeys(item for item in items if item))
        return "; ".join(ordered) if ordered else fallback

    goal = (
        enhancement_goal
        or "Make this photo cleaner, more polished, and print-ready while preserving the exact scene."
    )
    safe = joined_unique(
        safe_edits,
        "Exposure balancing, white balance, local contrast, noise reduction, print sharpening",
    )
    conditional = joined_unique(
        conditional_edits,
        "Recover blown sky only with plausible natural tones; small border expansion only if subject is awkwardly cut; mild horizon/perspective correction only if visibly tilted",
    )
    avoid = joined_unique(
        avoid_edits,
        "Do not change pose, identity, scene layout, text, architecture, or artwork content",
    )
    risks = joined_unique(
        risk_flags,
        "Preserve authenticity; avoid over-processing",
    )

    # ----- The prompt -----
    #
    # Design principles vs. the v1 prompt:
    #   1. One short "hard rule" paragraph instead of a scattered preservation list.
    #   2. Repair modes described as a concise decision table, not five essay paragraphs.
    #   3. Tonal polish section kept short — the per-image metadata already carries detail.
    #   4. Total length roughly halved while covering the same edge cases.

    repair_mode = (analysis or {}).get("repair_mode", "tonal_only")
    pad_sides = (analysis or {}).get("pad_sides", [])
    is_padded = len(pad_sides) > 0

    # Build mode-specific instruction block.
    mode_instructions = {
        "edge_completion": (
            "## Primary task: Edge completion\n"
            f"The image has been padded on: {', '.join(pad_sides)}.\n"
            "The blurred margins are blank canvas — fill them with a natural continuation of the scene.\n"
            "Complete ONLY the clipped portion of existing elements. Do NOT add new windows, arches, columns, beams, or decorative panels beyond what the scene clearly implies.\n"
            "Use the intact opposite side or nearby repetition as a reference for material, ornament, spacing, thickness, alignment, and lighting.\n"
            "Keep the completion minimal and local so the facade/interior still reads as the same photograph.\n"
            "Do not settle for tonal fixes alone while an edge still reads as cut off."
        ),
        "square_and_level": (
            "## Primary task: Square & level\n"
            "Straighten the subject so borders, margins, verticals, and repeated lines read true, parallel, and architecturally correct.\n"
            "For framed artwork, wall reliefs, facades, and corridors, fix slight rotational or perspective skew before tone work.\n"
            "Keep the same scene and nearly the same framing. Do not invent missing content unless a tiny border continuation is clearly necessary."
        ),
        "crowd_cleanup": (
            "## Primary task: Crowd cleanup\n"
            "The real subject is the place, not the people. Thin the most distracting foreground figures, edge clutter, and central sightline blockers.\n"
            "Prioritize removing or reducing the nearest incidental people when they overpower the landmark, room, artwork, or vista.\n"
            "Rebuild only the small patches they occluded — do not redesign the space, change the room geometry, alter the ceiling/floor rhythm, or restage the whole scene.\n"
            "Keep enough distant visitors for believable scale and atmosphere unless the scene already reads as nearly empty.\n"
            "Aim for a cleaner, more readable place while preserving the same location, camera position, and overall scene structure."
        ),
        "subject_deblur": (
            "## Primary task: Subject deblur\n"
            "Apply local clarity recovery to the intended foreground person — face, hairline, glasses edge, clothing contours, and silhouette.\n"
            "If the image is a person-with-landmark travel photo, preserve both the person and the landmark exactly while improving the person's clarity conservatively.\n"
            "Preserve exact identity, pose, expression, and subject count.\n"
            "Do not hallucinate hidden facial detail, change identity, or make the person look AI-generated."
        ),
        "tonal_only": "",
    }

    mode_block = mode_instructions.get(repair_mode, "")

    padding_note = ""
    if is_padded:
        padding_note = (
            "\n## About this image\n"
            f"The image has been pre-padded with soft blurred margins on: {', '.join(pad_sides)}.\n"
            "The sharp original sits in the centre. The blurry borders are canvas for you to extend into.\n"
            "Do NOT duplicate or add extra elements — the scene has exactly the elements visible in the sharp centre.\n"
        )

    output_note = "Deliver one print-ready image: realistic, natural, faithful to the original."
    if is_padded:
        output_note += " Blurred margins must become believable scene continuation — no seams or duplicated elements."

    return f"""\
Improve this travel photograph for print. This is a restoration, not a regeneration.

Filename: {ranked_filename}
{padding_note}
## Hard rules
Preserve: subjects, subject count, relationships, architecture, artwork, signage, reflections, scene geometry, identity, pose, expression, clothing, and the same photographic moment.
Never: add/remove/move major elements unless the chosen repair mode explicitly allows a small local cleanup, repaint, cartoonize, over-HDR, replace the main subject, or hallucinate detail not implied by visible pixels.
Trust the image more than any metadata. If the metadata conflicts with the visible scene, follow the visible scene.

{mode_block}

## Decision discipline
Fix the chosen primary repair mode first. Do not hide from the main issue by doing only exposure, contrast, or color cleanup.
Use the smallest set of changes that clearly improves print quality while keeping the image believable and faithful to the source.

## Tonal guidance (suggestions from a prior model pass — trust what you see over these)
A previous ranking model reviewed this image and offered the following hints.
Apply only what is clearly supported by what you actually see in the image.
Ignore any suggestion that conflicts with the hard rules above or that would require
changing subjects, geometry, or content.
Suggested goal: {goal}
Consider if clearly visible in the image: {safe}
Consider only if it clearly helps and stays faithful: {conditional}
Do not do: {avoid}
Risks to respect: {risks}

## Output
{output_note}"""


def request_with_retries(
    *,
    credentials: Any,
    project_id: str,
    location: str,
    model: str,
    payload: dict[str, Any],
    timeout: int,
) -> tuple[dict[str, Any], float]:
    attempts = 3
    last_error: Exception | None = None

    service_endpoint = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )

    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            VERTEX_ENDPOINT_TEMPLATE.format(
                service_endpoint=service_endpoint,
                project=project_id,
                location=location,
                model=model,
            ),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **auth_header(credentials),
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_json = json.loads(response.read().decode("utf-8"))
            elapsed = time.perf_counter() - started
            return response_json, elapsed
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code in RETRYABLE_HTTP_CODES and attempt < attempts:
                sleep(1.5 * attempt)
                last_error = RuntimeError(f"Retryable Vertex HTTP {exc.code}: {body[:1000]}")
                continue
            raise RuntimeError(
                f"Vertex AI request failed with HTTP {exc.code}: {body[:2500]}"
            ) from exc
        except urllib.error.URLError as exc:
            if attempt < attempts:
                sleep(1.5 * attempt)
                last_error = exc
                continue
            raise RuntimeError(f"Vertex AI request failed: {exc}") from exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("Vertex AI request failed without a captured exception")


def extract_image_output(response_json: dict[str, Any]) -> tuple[bytes, str, list[str]]:
    candidates = response_json.get("candidates") or []
    if not candidates:
        raise RuntimeError(
            f"Vertex AI returned no candidates: {json.dumps(response_json, indent=2)[:1500]}"
        )

    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts: list[str] = []
    image_bytes: bytes | None = None
    mime_type = "image/png"

    for part in parts:
        if part.get("text"):
            texts.append(str(part["text"]))
        inline_data = part.get("inlineData") or part.get("inline_data")
        if inline_data and inline_data.get("data"):
            image_bytes = base64.b64decode(inline_data["data"])
            mime_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"

    if image_bytes is None:
        raise RuntimeError(
            f"Vertex AI returned no image output: {json.dumps(response_json, indent=2)[:2000]}"
        )
    return image_bytes, mime_type, texts


def output_suffix_for_mime(mime_type: str, fallback: str) -> str:
    normalized = mime_type.lower()
    if normalized == "image/png":
        return ".png"
    if normalized in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if normalized == "image/webp":
        return ".webp"
    return fallback


def improve_one_image(
    *,
    image_path: Path,
    detail: dict[str, Any] | None,
    credentials: Any,
    project_id: str,
    location: str,
    model: str,
    analysis_model: str,
    output_dir: Path,
    meta_dir: Path,
    input_max_side: int,
    input_jpeg_quality: int,
    image_size: str,
    pad_percent: int,
    timeout: int,
    overwrite: bool,
) -> dict[str, Any]:
    ranked_filename = image_path.name

    # Step 1: prepare a lightweight (unpadded) version for the analysis call.
    analysis_mime, analysis_bytes, _, original_size = resize_input_image(
        image_path,
        input_max_side=2048,  # higher detail improves structural mode detection on clipped edges
        input_jpeg_quality=85,
    )

    # Step 2: ask Vertex which repair mode applies and which sides to pad.
    analysis = analyze_image_for_repair(
        image_bytes=analysis_bytes,
        mime_type=analysis_mime,
        credentials=credentials,
        project_id=project_id,
        location=location,
        model=analysis_model,
        timeout=timeout,
    )

    # Step 3: prepare the real input, padding only the sides the model requested.
    pad_sides = analysis.get("pad_sides", [])
    mime_type, input_bytes, input_info, original_size = resize_input_image(
        image_path,
        input_max_side=input_max_side,
        input_jpeg_quality=input_jpeg_quality,
        pad_percent=pad_percent if pad_sides else 0,
        pad_sides=pad_sides,
    )
    input_info["analysis"] = analysis

    prompt = build_improvement_prompt(ranked_filename, detail, analysis=analysis)
    ratio = aspect_ratio_string(original_size)

    existing_outputs = sorted(output_dir.glob(f"{image_path.stem}.*"))
    if existing_outputs and not overwrite:
        return {
            "input_path": str(image_path),
            "output_path": str(existing_outputs[0]),
            "status": "skipped_existing",
            "model": model,
            "project_id": project_id,
            "location": location,
        }

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(input_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": ratio,
                "imageSize": image_size,
            },
        },
    }

    response_json, elapsed = request_with_retries(
        credentials=credentials,
        project_id=project_id,
        location=location,
        model=model,
        payload=payload,
        timeout=timeout,
    )
    output_bytes, output_mime_type, output_texts = extract_image_output(response_json)

    suffix = output_suffix_for_mime(output_mime_type, image_path.suffix.lower() or ".png")
    output_path = output_dir / f"{image_path.stem}{suffix}"
    output_path.write_bytes(output_bytes)

    record = {
        "input_path": str(image_path),
        "output_path": str(output_path),
        "ranked_filename": ranked_filename,
        "model": model,
        "project_id": project_id,
        "location": location,
        "elapsed_seconds": round(elapsed, 2),
        "prompt": prompt,
        "input_info": input_info,
        "requested_aspect_ratio": ratio,
        "requested_image_size": image_size,
        "output_mime_type": output_mime_type,
        "output_text": "\n".join(output_texts).strip(),
        "usage_metadata": response_json.get("usageMetadata", {}),
        "model_version": response_json.get("modelVersion"),
        "finish_reason": (response_json.get("candidates") or [{}])[0].get("finishReason"),
        "ranking_detail": detail or {},
        "status": "completed",
    }
    meta_path = meta_dir / f"{image_path.stem}.json"
    meta_path.write_text(json.dumps(record, indent=2))
    return record


def main() -> int:
    args = parse_args()
    portfolio_dir = resolve_portfolio_dir(args.portfolio_dir)

    credentials, project_id = load_vertex_credentials(args.project_id)
    summary = load_portfolio_summary(portfolio_dir)
    reason_lookup = build_reason_lookup(summary)
    ranked_images = choose_ranked_images(portfolio_dir, args.start_rank, args.limit)
    if not ranked_images:
        raise ValueError("No ranked images matched the requested range")

    output_dir = portfolio_dir / args.output_dir_name
    meta_dir = output_dir / "meta"
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for image_path in ranked_images:
        original_filename = image_path.name.split("_", 1)[1] if "_" in image_path.name else image_path.name
        detail = reason_lookup.get(original_filename)
        try:
            result = improve_one_image(
                image_path=image_path,
                detail=detail,
                credentials=credentials,
                project_id=project_id,
                location=args.location,
                model=args.model,
                analysis_model=args.analysis_model,
                output_dir=output_dir,
                meta_dir=meta_dir,
                input_max_side=args.input_max_side,
                input_jpeg_quality=args.input_jpeg_quality,
                image_size=args.image_size,
                pad_percent=args.pad_percent,
                timeout=args.timeout,
                overwrite=args.overwrite,
            )
            results.append(result)
        except Exception as exc:
            failures.append(
                {
                    "input_path": str(image_path),
                    "error": str(exc),
                }
            )

    run_summary = {
        "portfolio_dir": str(portfolio_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "analysis_model": args.analysis_model,
        "project_id": project_id,
        "location": args.location,
        "image_size": args.image_size,
        "input_max_side": args.input_max_side,
        "input_jpeg_quality": args.input_jpeg_quality,
        "pad_percent": args.pad_percent,
        "start_rank": args.start_rank,
        "limit": args.limit,
        "processed_count": len(ranked_images),
        "completed_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
    (output_dir / "improvement_summary.json").write_text(json.dumps(run_summary, indent=2))
    print(json.dumps(run_summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
