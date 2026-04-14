"""Optional scene-level heuristics for distracting background people."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import PipelineConfig


@dataclass
class SceneAnalysis:
    passed: bool
    flagged: bool
    face_count: int
    background_face_count: int
    dominant_face_ratio: float
    penalty: float
    rejection_reason: str = ""


class BackgroundPeopleChecker:
    """Conservative face-based detector for likely background-person distractions."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            raise RuntimeError(f"Unable to load Haar cascade from {cascade_path}")

    def analyze(self, bgr_image: np.ndarray) -> SceneAnalysis:
        if bgr_image is None:
            return SceneAnalysis(True, False, 0, 0, 0.0, 0.0)
        gray = cv2.cvtColor(self._resize_for_detection(bgr_image), cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(24, 24),
        )
        if len(faces) <= 1:
            dominant_ratio = self._largest_face_ratio(faces, gray.shape[0] * gray.shape[1])
            return SceneAnalysis(True, False, len(faces), 0, dominant_ratio, 0.0)

        image_area = float(gray.shape[0] * gray.shape[1])
        face_areas = sorted((w * h for (_, _, w, h) in faces), reverse=True)
        largest_area = float(face_areas[0])
        dominant_ratio = largest_area / image_area if image_area else 0.0

        background_faces = 0
        for area in face_areas[1:]:
            face_ratio = area / image_area if image_area else 0.0
            relative_ratio = area / largest_area if largest_area else 0.0
            if (
                face_ratio >= self._cfg.background_face_ratio
                and relative_ratio <= self._cfg.background_face_relative_ratio
            ):
                background_faces += 1

        flagged = dominant_ratio >= self._cfg.dominant_face_ratio and background_faces > self._cfg.max_background_faces
        penalty = self._cfg.background_people_penalty * background_faces if flagged else 0.0
        reason = ""
        if flagged:
            reason = f"background_people_detected ({background_faces} smaller faces behind subject)"

        return SceneAnalysis(
            passed=not (flagged and self._cfg.background_people_hard_reject),
            flagged=flagged,
            face_count=len(faces),
            background_face_count=background_faces,
            dominant_face_ratio=dominant_ratio,
            penalty=penalty,
            rejection_reason=reason,
        )

    def _resize_for_detection(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        max_dim = 1280
        if max(h, w) <= max_dim:
            return image
        scale = max_dim / max(h, w)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def _largest_face_ratio(self, faces, image_area: int) -> float:
        if len(faces) == 0 or image_area == 0:
            return 0.0
        largest_area = max(w * h for (_, _, w, h) in faces)
        return float(largest_area / image_area)
