"""Face detection and identity matching logic."""

from __future__ import annotations

from dataclasses import dataclass
import contextlib
import os
from typing import Sequence

import cv2
import insightface
import numpy as np
from PIL import Image, ImageDraw

from .config import DETECTION_SIZE, SIMILARITY_THRESHOLD


@dataclass(frozen=True)
class FaceValidation:
    image_rgb: np.ndarray
    bbox: tuple[int, int, int, int] | None
    crop_rgb: np.ndarray | None
    preview_rgb: np.ndarray | None
    embedding: np.ndarray | None
    error: str | None

    @property
    def is_valid(self) -> bool:
        return (
            self.error is None
            and self.bbox is not None
            and self.crop_rgb is not None
            and self.embedding is not None
        )


class FaceDetector:
    def __init__(self):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                self.model = insightface.app.FaceAnalysis()
                self.model.prepare(ctx_id=-1, det_size=DETECTION_SIZE)

    def detect_faces(self, image_bgr: np.ndarray):
        return self.model.get(image_bgr)

    def get_face_count(self, image_bgr: np.ndarray) -> int:
        return len(self.detect_faces(image_bgr))


class IdentityChecker:
    def __init__(self, face_detector: FaceDetector, threshold: float = SIMILARITY_THRESHOLD):
        self.face_detector = face_detector
        self.threshold = threshold

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        return float(np.dot(emb1, emb2))

    def get_embedding(self, face_crop_rgb: np.ndarray) -> np.ndarray:
        faces = self.face_detector.detect_faces(rgb_to_bgr(face_crop_rgb))
        if len(faces) != 1:
            raise ValueError(
                "The face-only crop could not be embedded reliably. Please retry with a clearer photo."
            )

        embedding = getattr(faces[0], "embedding", None)
        if embedding is None:
            raise ValueError("The face pipeline did not return an embedding for this face crop.")

        return np.asarray(embedding, dtype=np.float32)

    def verify_embeddings(self, reference_embedding: np.ndarray, verify_embedding: np.ndarray):
        similarity = self.cosine_similarity(reference_embedding, verify_embedding)
        return similarity >= self.threshold, similarity


def load_face_detector() -> FaceDetector:
    return FaceDetector()


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def read_image_file(file_obj) -> np.ndarray:
    image = Image.open(file_obj)
    return pil_to_rgb_array(image)


def load_image_path(path: str) -> np.ndarray:
    image = Image.open(path)
    return pil_to_rgb_array(image)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def normalize_bbox(bbox: Sequence[float], image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )


def crop_face_only(image_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image_rgb[y1:y2, x1:x2].copy()


def draw_detection_box(image_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    preview = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(preview)
    draw.rectangle(bbox, outline=(0, 180, 80), width=max(3, preview.width // 180))
    return np.array(preview)


def validate_single_face(face_detector: FaceDetector, image_rgb: np.ndarray) -> FaceValidation:
    faces = face_detector.detect_faces(rgb_to_bgr(image_rgb))

    if len(faces) == 0:
        return FaceValidation(
            image_rgb=image_rgb,
            bbox=None,
            crop_rgb=None,
            preview_rgb=None,
            embedding=None,
            error="No face detected - please try a clearer, front-facing photo.",
        )

    if len(faces) > 1:
        return FaceValidation(
            image_rgb=image_rgb,
            bbox=None,
            crop_rgb=None,
            preview_rgb=None,
            embedding=None,
            error="Multiple faces detected in this photo - please provide a photo with only one face.",
        )

    embedding = getattr(faces[0], "embedding", None)
    if embedding is None:
        return FaceValidation(
            image_rgb=image_rgb,
            bbox=None,
            crop_rgb=None,
            preview_rgb=None,
            embedding=None,
            error="The face pipeline could not create an embedding for this face - please try a clearer photo.",
        )

    bbox = normalize_bbox(faces[0].bbox, image_rgb.shape)
    crop_rgb = crop_face_only(image_rgb, bbox)

    if crop_rgb.size == 0:
        return FaceValidation(
            image_rgb=image_rgb,
            bbox=None,
            crop_rgb=None,
            preview_rgb=None,
            embedding=None,
            error="The detected face crop was empty - please try a clearer, centered photo.",
        )

    return FaceValidation(
        image_rgb=image_rgb,
        bbox=bbox,
        crop_rgb=crop_rgb,
        preview_rgb=draw_detection_box(image_rgb, bbox),
        embedding=np.asarray(embedding, dtype=np.float32),
        error=None,
    )
