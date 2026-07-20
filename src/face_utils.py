"""InsightFace helpers for detection, tight cropping, embedding, and matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw

from .config import DETECTION_SIZE, MODEL_PACK_NAME


@dataclass(frozen=True)
class FaceValidation:
    image_rgb: np.ndarray
    bbox: tuple[int, int, int, int] | None
    crop_rgb: np.ndarray | None
    preview_rgb: np.ndarray | None
    error: str | None

    @property
    def is_valid(self) -> bool:
        return self.error is None and self.bbox is not None and self.crop_rgb is not None


def load_analyzer() -> FaceAnalysis:
    analyzer = FaceAnalysis(name=MODEL_PACK_NAME, providers=["CPUExecutionProvider"])
    analyzer.prepare(ctx_id=-1, det_size=DETECTION_SIZE)
    return analyzer


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


def detect_faces(analyzer: FaceAnalysis, image_rgb: np.ndarray) -> Sequence:
    return analyzer.get(rgb_to_bgr(image_rgb))


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


def validate_single_face(analyzer: FaceAnalysis, image_rgb: np.ndarray) -> FaceValidation:
    faces = detect_faces(analyzer, image_rgb)

    if len(faces) == 0:
        return FaceValidation(
            image_rgb=image_rgb,
            bbox=None,
            crop_rgb=None,
            preview_rgb=None,
            error="No face detected - please try a clearer, front-facing photo.",
        )

    if len(faces) > 1:
        return FaceValidation(
            image_rgb=image_rgb,
            bbox=None,
            crop_rgb=None,
            preview_rgb=None,
            error="Multiple faces detected in this photo - please provide a photo with only one face.",
        )

    bbox = normalize_bbox(faces[0].bbox, image_rgb.shape)
    crop_rgb = crop_face_only(image_rgb, bbox)

    if crop_rgb.size == 0:
        return FaceValidation(
            image_rgb=image_rgb,
            bbox=None,
            crop_rgb=None,
            preview_rgb=None,
            error="The detected face crop was empty - please try a clearer, centered photo.",
        )

    return FaceValidation(
        image_rgb=image_rgb,
        bbox=bbox,
        crop_rgb=crop_rgb,
        preview_rgb=draw_detection_box(image_rgb, bbox),
        error=None,
    )


def embedding_from_face_crop(analyzer: FaceAnalysis, crop_rgb: np.ndarray) -> np.ndarray:
    faces = detect_faces(analyzer, crop_rgb)

    if len(faces) != 1:
        raise ValueError(
            "The face-only crop could not be embedded reliably. Please retry with a clearer, front-facing photo."
        )

    embedding = faces[0].normed_embedding
    if embedding is None:
        raise ValueError("InsightFace did not return an embedding for this face crop.")

    return np.asarray(embedding, dtype=np.float32)


def cosine_similarity(first_embedding: np.ndarray, second_embedding: np.ndarray) -> float:
    numerator = float(np.dot(first_embedding, second_embedding))
    denominator = float(np.linalg.norm(first_embedding) * np.linalg.norm(second_embedding))
    if denominator == 0:
        return 0.0
    return numerator / denominator
