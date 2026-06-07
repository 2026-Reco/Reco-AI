"""COCO SSD + 컨투어 기반 쓰레기 영역 검출."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

WASTE_OBJECT_NAMES = frozenset(
    {
        "bottle",
        "wine glass",
        "cup",
        "bowl",
        "book",
        "banana",
        "apple",
        "orange",
        "broccoli",
        "carrot",
        "donut",
        "cake",
        "scissors",
        "toothbrush",
        "cell phone",
        "remote",
        "keyboard",
        "mouse",
        "handbag",
        "backpack",
        "suitcase",
    }
)


@dataclass(frozen=True)
class DetectedObject:
    object_name: str
    bbox: Tuple[int, int, int, int]
    score: float


class ObjectDetector:
    _model = None
    _transform = None
    _categories: List[str] = []

    def __init__(self, score_threshold: float = 0.32):
        self.score_threshold = score_threshold
        self._ensure_loaded()

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._model is not None:
            return
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        cls._model = ssdlite320_mobilenet_v3_large(weights=weights)
        cls._model.eval()
        cls._transform = weights.transforms()
        cls._categories = list(weights.meta["categories"])

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedObject]:
        self._ensure_loaded()
        h, w = frame_bgr.shape[:2]
        results: List[DetectedObject] = []

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(Image.fromarray(rgb))

        with torch.no_grad():
            outputs = self._model([tensor])[0]

        for box, score, label_id in zip(
            outputs["boxes"], outputs["scores"], outputs["labels"]
        ):
            conf = float(score.item())
            if conf < self.score_threshold:
                continue
            name = self._categories[int(label_id)]
            if name not in WASTE_OBJECT_NAMES:
                continue
            x1, y1, x2, y2 = [int(max(0, v)) for v in box.tolist()]
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            results.append(
                DetectedObject(object_name=name, bbox=(x1, y1, x2, y2), score=conf)
            )

        results.extend(self._contour_boxes(frame_bgr))
        results = self._nms(results)
        results.sort(key=lambda o: o.score, reverse=True)
        return results[:8]

    def _contour_boxes(self, frame_bgr: np.ndarray) -> List[DetectedObject]:
        """SSD가 놓친 물체 — 전경 컨투어."""
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out: List[DetectedObject] = []
        min_area = h * w * 0.015
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            pad = 8
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w - 1, x + bw + pad)
            y2 = min(h - 1, y + bh + pad)
            if x2 - x1 < 30 or y2 - y1 < 30:
                continue
            out.append(
                DetectedObject(
                    object_name="object",
                    bbox=(x1, y1, x2, y2),
                    score=0.38,
                )
            )
        return out[:3]

    @staticmethod
    def _nms(objects: List[DetectedObject], iou_thresh: float = 0.45) -> List[DetectedObject]:
        if not objects:
            return []
        kept: List[DetectedObject] = []
        for obj in sorted(objects, key=lambda o: o.score, reverse=True):
            if all(_iou(obj.bbox, k.bbox) < iou_thresh for k in kept):
                kept.append(obj)
        return kept


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)
