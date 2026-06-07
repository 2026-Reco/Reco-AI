"""CLIP + TrashNet + 비전 휴리스틱 앙상블 분류."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from app.services.clip_classifier import predict_clip
from app.services.material_heuristics import (
    enforce_material_rules,
    is_can_like,
    is_glass_like,
    is_paper_like,
    is_plastic_like,
    visual_material_scores,
)
from app.services.trashnet_classifier import is_available as trashnet_available
from app.services.trashnet_classifier import predict_trashnet
from app.services.waste_taxonomy import (
    MATERIAL_LABELS,
    SUMMARY_LABELS,
    to_summary,
)

# 하위 호환
__all__ = ["WasteClassifier", "ClassificationResult", "MATERIAL_LABELS", "SUMMARY_LABELS"]


@dataclass(frozen=True)
class ClassificationResult:
    summary: Dict[str, float]
    detail: Dict[str, float]
    primary_material: str
    confidence: float
    waste_type_id: str = "unknown"
    waste_type_ko: str = "미확인"


class WasteClassifier:
    def __init__(self, device: str | None = None, smooth_frames: int = 8):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.smooth_frames = smooth_frames
        self._history: deque = deque(maxlen=smooth_frames)
        self._type_history: deque = deque(maxlen=smooth_frames)
        self._use_trashnet = trashnet_available()

    def classify_bytes(self, image_bytes: bytes) -> ClassificationResult:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")
        return self.classify_bgr(frame)

    def classify_bgr_instant(
        self,
        frame_bgr: np.ndarray,
        object_name: Optional[str] = None,
    ) -> ClassificationResult:
        dist, type_id, type_ko, _ = self._ensemble_predict(frame_bgr, object_name)
        return self._dist_to_result(dist, type_id, type_ko)

    def classify_bgr(self, frame_bgr: np.ndarray) -> ClassificationResult:
        dist, type_id, type_ko, _ = self._ensemble_predict(frame_bgr, None)
        self._history.append(dist)
        self._type_history.append(type_ko)

        smoothed = np.mean(self._history, axis=0)
        smoothed /= smoothed.sum()

        # 최빈 쓰레기 종류
        from collections import Counter

        type_ko = Counter(self._type_history).most_common(1)[0][0] if self._type_history else type_ko

        return self._dist_to_result(smoothed, type_id, type_ko)

    def reset(self) -> None:
        self._history.clear()
        self._type_history.clear()

    def _ensemble_predict(
        self,
        frame_bgr: np.ndarray,
        object_name: Optional[str],
    ) -> Tuple[np.ndarray, str, str, float]:
        clip_dist, type_id, type_ko, clip_conf = predict_clip(frame_bgr, self.device)

        trash = predict_trashnet(frame_bgr, self.device)
        if trash is not None:
            tn_dist, tn_id, tn_ko, tn_conf = trash
            if tn_conf >= 0.55:
                w_tn, w_clip = 0.62, 0.28
            else:
                w_tn, w_clip = 0.45, 0.45
            fused = w_tn * tn_dist + w_clip * clip_dist
            type_id = tn_id if tn_conf >= clip_conf else type_id
            type_ko = tn_ko if tn_conf >= clip_conf else type_ko
            conf = max(tn_conf, clip_conf)
        else:
            fused = 0.82 * clip_dist + 0.18 * visual_material_scores(frame_bgr)
            conf = clip_conf

        fused /= fused.sum() + 1e-9
        fused = enforce_material_rules(frame_bgr, fused, object_name)

        # 비전 휴리스틱이 확실하면 CLIP보다 우선 (캔·페트 등)
        if is_can_like(frame_bgr):
            type_ko, type_id = "알루미늄 캔", "aluminum_can"
            idx = {n: i for i, n in enumerate(MATERIAL_LABELS)}
            fused = np.zeros(len(MATERIAL_LABELS))
            fused[idx["금속"]] = 0.9
            fused[idx["기타"]] = 0.1
        elif is_plastic_like(frame_bgr) and not is_can_like(frame_bgr):
            type_ko, type_id = "페트병", "pet_bottle"
        elif is_glass_like(frame_bgr) and not is_plastic_like(frame_bgr):
            type_ko, type_id = "유리병", "glass_bottle"
        elif is_paper_like(frame_bgr):
            type_ko, type_id = "종이", "paper"

        primary_mat = MATERIAL_LABELS[int(np.argmax(fused))]
        if primary_mat == "금속" and "캔" not in type_ko:
            type_ko, type_id = "알루미늄 캔", "aluminum_can"
        elif primary_mat == "플라스틱" and type_ko in ("미확인", "일반 쓰레기", "종이"):
            type_ko, type_id = "페트병", "pet_bottle"
        elif primary_mat == "유리" and type_ko in ("미확인", "일반 쓰레기"):
            type_ko, type_id = "유리병", "glass_bottle"
        elif primary_mat == "종이" and type_ko in ("미확인", "일반 쓰레기"):
            type_ko, type_id = "종이", "paper"

        return fused, type_id, type_ko, float(conf)

    def _dist_to_result(
        self, dist: np.ndarray, type_id: str, type_ko: str
    ) -> ClassificationResult:
        detail = {
            label: round(float(dist[i]) * 100, 1)
            for i, label in enumerate(MATERIAL_LABELS)
        }
        summary = to_summary(detail)
        primary = max(MATERIAL_LABELS, key=lambda k: detail[k])
        return ClassificationResult(
            summary=summary,
            detail=detail,
            primary_material=primary,
            confidence=round(detail[primary] / 100.0, 4),
            waste_type_id=type_id,
            waste_type_ko=type_ko,
        )
