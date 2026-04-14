"""CLIP-based zero-shot aesthetic scoring using OpenCLIP."""

from __future__ import annotations

import gc
import logging
from functools import lru_cache

import cv2
import numpy as np
import torch
import open_clip
from PIL import Image

from config import PipelineConfig

log = logging.getLogger(__name__)


class AestheticScorer:
    """Scores images against positive/negative text prompts via cosine similarity."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._device = self._pick_device()

        log.info(
            "Loading CLIP model %s/%s on %s …",
            cfg.clip_model_name, cfg.clip_pretrained, self._device,
        )

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            cfg.clip_model_name,
            pretrained=cfg.clip_pretrained,
            device=self._device,
        )
        self._tokenizer = open_clip.get_tokenizer(cfg.clip_model_name)
        self._model.eval()

        # Pre-encode the text prompts once.
        self._pos_emb = self._encode_text(cfg.positive_prompt)
        self._neg_emb = self._encode_text(cfg.negative_prompt)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = self._tokenizer([text]).to(self._device)
        emb = self._model.encode_text(tokens)
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb

    @torch.no_grad()
    def _encode_image(self, pil_img: Image.Image) -> torch.Tensor:
        tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)
        emb = self._model.encode_image(tensor)
        emb /= emb.norm(dim=-1, keepdim=True)
        return emb

    # ── Public API ────────────────────────────────────────────────────

    def score(self, bgr_image: np.ndarray) -> float:
        """Return a score in [0, 1] — fraction of similarity on the *positive* side.

        score > 0.5  → closer to positive prompt
        score > cfg.clip_score_threshold → considered print-worthy
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        img_emb = self._encode_image(pil_img)
        pos_sim = (img_emb @ self._pos_emb.T).item()
        neg_sim = (img_emb @ self._neg_emb.T).item()

        # Softmax-style normalization to [0, 1].
        score = float(np.exp(pos_sim) / (np.exp(pos_sim) + np.exp(neg_sim)))

        # Explicit cleanup.
        del img_emb, pil_img, rgb
        if self._device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return score

    def passes(self, bgr_image: np.ndarray) -> tuple[bool, float]:
        """Convenience: returns (passed, score)."""
        s = self.score(bgr_image)
        passed = s >= self._cfg.clip_score_threshold
        if not passed:
            log.info("SKIP (aesthetic) CLIP score=%.3f < %.3f", s, self._cfg.clip_score_threshold)
        return passed, s
