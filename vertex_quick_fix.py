#!/usr/bin/env python3
"""Quick-fix individual images through Vertex AI without a full pipeline run.

Usage:
    vertex_quick_fix.py --images path/to/photo.jpg            # default auto prompt
    vertex_quick_fix.py --images photo.jpg --prompt "TEXT"    # manual prompt (top priority)
    vertex_quick_fix.py --images photo.jpg --gemma-analyze    # gemma4:31b writes the prompt

Output is saved in a _vertex_fixed/ folder next to the source images.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request
from pathlib import Path

from vertex_ranked_photo_improver2 import (
    load_vertex_credentials,
    improve_one_image,
    _load_image_any,
)

DEFAULT_MODEL = "gemini-3-pro-image-preview"
DEFAULT_ANALYSIS_MODEL = "gemini-2.5-flash"
OLLAMA_URL = "http://localhost:11434/api/chat"
GEMMA_MODEL = "gemma4:31b"

GEMMA_SYSTEM_PROMPT = """\
You are an expert photo editor and art director. You will be shown a photograph.
Your job is to write the best possible enhancement prompt to send to an AI image editing model (Vertex AI Gemini).

The prompt should:
- Describe exactly what photographic improvements to make (lighting, colour grading, contrast, sharpness, texture, atmosphere, etc.)
- Clearly identify the main subject(s) and explicitly instruct the model NOT to alter them
- Be specific about what to enhance and what to leave untouched
- Sound like instructions from a professional photographer to a retoucher
- Be 3-6 sentences long, focused and precise

Return ONLY the prompt text. No explanations, no preamble, no markdown.
"""

GEMMA_SYSTEM_PROMPT_SHARP = """\
You are an expert photo retoucher specializing in subject clarity. You will be shown a photograph.
Your job is to write an enhancement prompt focused on making the main subject(s) as sharp, clear, and in-focus as possible
while improving the overall image aesthetically.

The prompt should:
- PRIORITY: Enhance sharpness, clarity, micro-contrast, and focus on the main subject's face and body
- Improve facial feature definition, eye clarity, skin texture detail
- Enhance overall image sharpness and local contrast
- Apply colour grading and lighting adjustments WITHOUT modifying the subject's appearance
- Be very explicit about NOT altering the subject's face, features, or identity
- Sound like instructions from a professional retoucher
- Be 3-6 sentences long, focused and precise

Return ONLY the prompt text. No explanations, no preamble, no markdown.
"""


def gemma_analyze_image(image_path: Path, ollama_model: str = GEMMA_MODEL, system_prompt: str | None = None) -> str:
    """Ask gemma4:31b to look at the image and write the best Vertex enhancement prompt."""
    image = _load_image_any(image_path)
    from PIL import ImageOps
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    system = system_prompt or GEMMA_SYSTEM_PROMPT
    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "Analyse this photo and write the best enhancement prompt for it.",
                "images": [b64],
            },
        ],
        "stream": False,
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["message"]["content"].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send individual images through Vertex AI cleanup.",
        epilog="Modes: (1) no flags = default auto prompt, (2) --gemma-analyze --prompt TEXT = gemma analysis + your prompt combined"
    )
    parser.add_argument("--images", nargs="+", required=True, metavar="PATH", help="Image file(s) to process")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--analysis-model", default=DEFAULT_ANALYSIS_MODEL)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--location", default="global")
    parser.add_argument("--image-size", default="4K")
    parser.add_argument("--input-max-side", type=int, default=4096)
    parser.add_argument("--input-jpeg-quality", type=int, default=90)
    parser.add_argument("--pad-percent", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prompt",
        default=None,
        metavar="TEXT",
        help="Custom prompt. Use with --gemma-analyze to combine gemma's analysis with your specific instructions.",
    )
    parser.add_argument(
        "--gemma-analyze",
        action="store_true",
        help=f"Use {GEMMA_MODEL} to analyse the image. Combine with --prompt for smart merging.",
    )
    parser.add_argument(
        "--gemma-model",
        default=GEMMA_MODEL,
        help=f"Ollama model to use for analysis (default: {GEMMA_MODEL})",
    )
    parser.add_argument(
        "--ultra-preserve",
        action="store_true",
        help="Use museum-quality preservation for subject. Every facial pixel, body contour, clothing detail must be IDENTICAL.",
    )
    parser.add_argument(
        "--selective-edit",
        action="store_true",
        help="Use selective inpainting/editing mode: remove/modify only background elements, never touch the main subject.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    credentials, project_id = load_vertex_credentials(args.project_id)

    failures = []
    for raw_path in args.images:
        image_path = Path(raw_path).expanduser().resolve()
        if not image_path.is_file():
            print(f"[SKIP] Not found: {image_path}")
            failures.append(str(image_path))
            continue

        output_dir = image_path.parent / "_vertex_fixed"
        meta_dir = output_dir / "meta"
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        # Resolve prompt
        prompt = None
        if args.gemma_analyze:
            print(f"[gemma] Analysing {image_path.name} ...")
            try:
                gemma_prompt = gemma_analyze_image(image_path, args.gemma_model, system_prompt=GEMMA_SYSTEM_PROMPT)
                gemma_prompt = (
                    gemma_prompt.rstrip()
                    + " CRITICAL: Do not modify the main subject's face, facial features, or facial structure in any way —"
                    " preserve exact facial details, skin tone, and expression as they are in the original."
                    " Additionally, remove any incidental background people or bystanders"
                    " that are not the main subject of the photo."
                )

                if args.prompt:
                    # Merge user's prompt with gemma's analysis
                    custom_prompt = args.prompt.strip()
                    if args.selective_edit:
                        custom_prompt = (
                            "=== SELECTIVE INPAINTING / REMOVAL EDIT ===\n"
                            "Use selective inpainting to EDIT only the background and crowd.\n"
                            "The main subject (woman in red dress with white bag) must remain PIXEL-PERFECT IDENTICAL.\n"
                            "Do NOT regenerate, reshape, or modify the subject in any way.\n"
                            "Only use inpainting to:\n"
                            f"- {custom_prompt}\n"
                            "- Remove or reduce background crowd\n"
                            "- Fill edited areas with plausible background (sky, ground, etc.)\n"
                            "The subject's face, body, dress, bag, expression, and position must be UNTOUCHED.\n"
                        )
                    elif args.ultra_preserve:
                        custom_prompt = (
                            f"{custom_prompt}\n\n"
                            "=== ULTRA-STRICT SUBJECT PRESERVATION (Museum Quality) ===\n"
                            "EVERY PIXEL OF THE SUBJECT'S FACE AND BODY MUST REMAIN ABSOLUTELY IDENTICAL.\n"
                            "Do not modify, warp, reshape, move, or adjust:\n"
                            "- Face shape, facial features, eyes, nose, mouth, jawline, cheekbones\n"
                            "- Skin texture, pores, skin tone, complexion\n"
                            "- Hair, hair texture, hair direction\n"
                            "- Body shape, posture, limbs, hands, fingers\n"
                            "- Clothing, dress fabric, wrinkles, folds, patterns\n"
                            "- Bag, wallet, accessories on the subject\n"
                            "- Expression, gaze direction, eye contact\n"
                            "The subject must appear as if photographed at the exact same moment with archival-quality precision.\n"
                            "Only modify: background, sky, lighting direction (not subject), distant elements, crowd composition.\n"
                        )
                    prompt = (
                        f"{custom_prompt}\n\n"
                        "---\n"
                        "## Gemma's analysis & suggestions (apply if aligned with above):\n"
                        f"{gemma_prompt}"
                    )
                    if args.selective_edit:
                        mode = "[SELECTIVE-EDIT]"
                    elif args.ultra_preserve:
                        mode = "[gemma+custom+ULTRA]"
                    else:
                        mode = "[gemma+custom]"
                    print(f"{mode} Merged prompt → {args.prompt[:50]}...\n")
                else:
                    # Use gemma's prompt as-is
                    prompt = gemma_prompt
                    print(f"[gemma] Prompt → {gemma_prompt}\n")
            except Exception as exc:
                print(f"[WARN] gemma analysis failed ({exc}), falling back to auto prompt")
                prompt = None

        print(f"[→] Processing {image_path.name} ...")
        try:
            result = improve_one_image(
                image_path=image_path,
                detail=None,
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
                manual_prompt=prompt,
            )
            status = result.get("status")
            if status == "skipped_existing":
                print(f"[SKIP] Already exists: {result['output_path']}")
            else:
                print(f"[✓] Saved → {result['output_path']}  ({result.get('elapsed_seconds', '?')}s)")
        except Exception as exc:
            print(f"[FAIL] {image_path.name}: {exc}")
            failures.append(str(image_path))

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
