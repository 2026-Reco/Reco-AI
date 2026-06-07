"""검출 박스 + 재질 라벨 결합."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.services.material_heuristics import (
    is_can_like,
    is_glass_like,
    is_paper_like,
    is_plastic_like,
)
from app.services.object_detector import DetectedObject, ObjectDetector
from app.services.waste_classifier import WasteClassifier

# COCO 이름 → 한글 표시
OBJECT_NAME_KO = {
    "bottle": "병",
    "wine glass": "와인잔",
    "cup": "컵",
    "bowl": "그릇",
    "book": "책",
    "banana": "바나나",
    "apple": "사과",
    "orange": "오렌지",
    "cell phone": "휴대폰",
    "remote": "리모컨",
    "keyboard": "키보드",
    "mouse": "마우스",
    "scissors": "가위",
    "toothbrush": "칫솔",
}


@dataclass(frozen=True)
class LabeledDetection:
    object_name: str
    object_name_ko: str
    material: str
    waste_type_ko: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    corners: Tuple[Tuple[int, int], ...]


def bbox_corners(bbox: Tuple[int, int, int, int]) -> Tuple[Tuple[int, int], ...]:
    x1, y1, x2, y2 = bbox
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def motion_fallback_bbox(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """검출기가 실패할 때 중앙 영역 전경으로 대략적 박스 추정."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < h * w * 0.02:
        return None

    x, y, bw, bh = cv2.boundingRect(best)
    pad = 12
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w - 1, x + bw + pad)
    y2 = min(h - 1, y + bh + pad)
    return (x1, y1, x2, y2)


class DetectionAnalysisService:
    def __init__(self, score_threshold: float = 0.32):
        self._detector = ObjectDetector(score_threshold=score_threshold)

    def label_objects(
        self,
        frame_bgr: np.ndarray,
        session_classifier: WasteClassifier,
    ) -> List[LabeledDetection]:
        objects = self._detector.detect(frame_bgr)
        labeled: List[LabeledDetection] = []

        for obj in objects:
            crop = _safe_crop(frame_bgr, obj.bbox)
            if crop is None:
                continue
            result = session_classifier.classify_bgr_instant(
                crop, object_name=obj.object_name
            )
            name_ko = result.waste_type_ko or _display_name_ko(
                obj.object_name, crop, result.primary_material
            )
            labeled.append(
                LabeledDetection(
                    object_name=obj.object_name,
                    object_name_ko=name_ko,
                    material=result.primary_material,
                    waste_type_ko=result.waste_type_ko,
                    confidence=result.confidence,
                    bbox=obj.bbox,
                    corners=bbox_corners(obj.bbox),
                )
            )

        if not labeled:
            fb = motion_fallback_bbox(frame_bgr)
            if fb is not None:
                crop = _safe_crop(frame_bgr, fb)
                if crop is not None:
                    result = session_classifier.classify_bgr_instant(crop)
                    name_ko = result.waste_type_ko or _display_name_ko(
                        "object", crop, result.primary_material
                    )
                    labeled.append(
                        LabeledDetection(
                            object_name="object",
                            object_name_ko=name_ko,
                            material=result.primary_material,
                            waste_type_ko=result.waste_type_ko,
                            confidence=result.confidence,
                            bbox=fb,
                            corners=bbox_corners(fb),
                        )
                    )

        return labeled


def _display_name_ko(
    object_name: str, crop: np.ndarray, material: str
) -> str:
    if material == "금속" and (
        object_name in ("bottle", "object") or is_can_like(crop)
    ):
        return "캔"
    if material == "플라스틱" and (
        object_name == "bottle" or is_plastic_like(crop)
    ):
        return "플라스틱병"
    if material == "유리":
        return "유리"
    if material == "종이" or (object_name == "book" or is_paper_like(crop)):
        return "종이"
    return OBJECT_NAME_KO.get(object_name, "물체")


def _safe_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop
