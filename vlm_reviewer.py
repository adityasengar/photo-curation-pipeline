"""Optional VLM reviewer for human-style photo selection on small groups."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

log = logging.getLogger(__name__)


@dataclass
class VLMReviewResult:
    best_filename: str
    reason: str
    raw_output: str
    load_seconds: float
    inference_seconds: float


class Qwen2VLReviewer:
    """Small wrapper around Qwen2-VL for family-photo group selection."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        cache_dir: str | Path = "/tmp/hf_home",
        max_new_tokens: int = 220,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = str(cache_dir)
        self._max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None
        # Qwen2-VL currently hits an unsupported Conv3D path on MPS in this env,
        # so prefer CPU for a reliable first integration.
        self._device = "cpu"
        self._load_seconds = 0.0

    def review_images(self, image_paths: list[Path]) -> VLMReviewResult:
        if len(image_paths) < 2:
            raise ValueError("Need at least 2 images to compare.")

        self._ensure_loaded()

        images = [Image.open(path).convert("RGB") for path in image_paths]
        numbered = "\n".join(f"{idx + 1}. {path.name}" for idx, path in enumerate(image_paths))
        prompt = (
            "You are selecting the best family travel photo from a small group of similar images.\n"
            "Choose exactly one best keeper.\n"
            "Prefer: eyes open, main subjects facing camera, no awkward pose, no back turned, "
            "no obvious throwaway frame, no major occlusion, and emotionally usable composition.\n"
            "If one image is just a zoomed or slightly shifted version of another, still pick only the strongest keeper.\n"
            "The images appear in this exact order:\n"
            f"{numbered}\n"
            "Return strict JSON only with this schema:\n"
            '{"best_filename":"exact filename from list","reason":"one short sentence"}'
        )

        messages = [
            {
                "role": "user",
                "content": [*([{"type": "image"}] * len(images)), {"type": "text", "text": prompt}],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) if hasattr(value, "to") else value for key, value in inputs.items()}

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )
        inference_seconds = time.perf_counter() - start

        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)]
        raw_output = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed = self._parse_json(raw_output)
        best_filename = parsed.get("best_filename", "").strip()
        reason = parsed.get("reason", "").strip()
        if best_filename not in {path.name for path in image_paths}:
            raise ValueError(f"Model did not return a valid filename. Output: {raw_output}")

        return VLMReviewResult(
            best_filename=best_filename,
            reason=reason,
            raw_output=raw_output,
            load_seconds=self._load_seconds,
            inference_seconds=inference_seconds,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        start = time.perf_counter()
        dtype = torch.float16 if self._device == "mps" else torch.float32
        self._processor = AutoProcessor.from_pretrained(
            self._model_name,
            cache_dir=self._cache_dir,
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._model_name,
            torch_dtype=dtype,
            cache_dir=self._cache_dir,
        )
        self._model.to(self._device)
        self._model.eval()
        self._load_seconds = time.perf_counter() - start
        log.info("Loaded %s on %s in %.1fs", self._model_name, self._device, self._load_seconds)

    def _parse_json(self, raw_output: str) -> dict[str, str]:
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse JSON from model output: {raw_output}")
        return json.loads(match.group(0))
