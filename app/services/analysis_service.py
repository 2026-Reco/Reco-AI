import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import Settings
from app.schemas.material import (
    MaterialAnalyzeResponse,
    MaterialRatio,
    ObjectDetection,
)
from app.schemas.gemini_analysis import GeminiAnalysisResult
from app.services.chart_generator import chart_to_base64, save_donut_chart
from app.services.detection_analysis import DetectionAnalysisService, LabeledDetection
from app.services.gemini_vision import GeminiVisionService
from app.services.local_recycling_guide import build_local_analysis
from app.services.waste_classifier import ClassificationResult, WasteClassifier
from app.services.waste_taxonomy import MATERIAL_LABELS, SUMMARY_LABELS, to_summary

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
LOCK_CONFIDENCE = 0.52
LOCK_STREAK = 2


class AnalysisService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._lock = Lock()
        self._sessions: Dict[str, WasteClassifier] = {}
        self._detection = DetectionAnalysisService()
        self._stable: Dict[str, Dict] = {}

    def analyze(
        self,
        image_bytes: bytes,
        session_id: Optional[str] = None,
        *,
        save_on_lock: bool = True,
        gemini: Optional[GeminiVisionService] = None,
        mime_type: str = "image/jpeg",
    ) -> MaterialAnalyzeResponse:
        sid = session_id or str(uuid.uuid4())
        classifier = self._get_session_classifier(sid)

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")

        labeled = self._detection.label_objects(frame, classifier)
        result = (
            _aggregate_from_detections(labeled)
            if labeled
            else classifier.classify_bgr(frame)
        )

        capture_path: Optional[str] = None
        chart_path: Optional[str] = None
        chart_b64: Optional[str] = None
        locked = False

        if labeled and save_on_lock:
            top = max(labeled, key=lambda d: d.confidence)
            locked, capture_path, chart_path, chart_b64 = self._maybe_lock_and_save(
                sid, frame, result, top, labeled
            )

        ai_result: Optional[GeminiAnalysisResult] = None
        ai_source: Optional[str] = None
        ai_error: Optional[str] = None

        if gemini is not None and gemini.enabled:
            ai_result, gemini_err = gemini.analyze_image_safe(
                image_bytes,
                mime_type=mime_type,
                local_hint={
                    "waste_type_ko": result.waste_type_ko,
                    "primary_material": result.primary_material,
                    "detections_count": len(labeled),
                },
            )
            if ai_result is not None:
                ai_source = "gemini"
            elif gemini_err:
                ai_error = gemini_err

        if ai_result is None:
            ai_result = build_local_analysis(
                result.waste_type_ko,
                result.primary_material,
                frame=frame,
            )
            ai_source = "local"

        if ai_result is not None:
            result = _merge_gemini_result(result, ai_result)

        return _to_response(
            result,
            sid,
            labeled,
            capture_path=capture_path,
            chart_path=chart_path,
            chart_base64=chart_b64,
            locked=locked,
            gemini_result=ai_result,
            gemini_error=ai_error if ai_source == "local" and ai_error else None,
            ai_source=ai_source,
        )

    def reset_session(self, session_id: str) -> bool:
        with self._lock:
            classifier = self._sessions.pop(session_id, None)
            self._stable.pop(session_id, None)
        if classifier is None:
            return False
        classifier.reset()
        return True

    def _maybe_lock_and_save(
        self,
        session_id: str,
        frame: np.ndarray,
        result: ClassificationResult,
        top: LabeledDetection,
        labeled: List[LabeledDetection],
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        if top.confidence < LOCK_CONFIDENCE:
            self._stable[session_id] = {"key": None, "streak": 0, "saved": False}
            return False, None, None, None

        key = f"{top.waste_type_ko}:{top.material}"
        state = self._stable.get(session_id, {"key": None, "streak": 0, "saved": False})

        if state.get("key") == key and not state.get("saved"):
            state["streak"] = state.get("streak", 0) + 1
        else:
            state = {"key": key, "streak": 1, "saved": False}

        self._stable[session_id] = state

        if state["streak"] < LOCK_STREAK or state.get("saved"):
            return False, None, None, None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_DIR / session_id
        out_dir.mkdir(parents=True, exist_ok=True)

        annotated = _draw_simple_boxes(frame.copy(), labeled)
        cap_file = out_dir / f"capture_{ts}.jpg"
        chart_file = out_dir / f"chart_{ts}.png"
        cv2.imwrite(str(cap_file), annotated)
        save_donut_chart(result.summary, chart_file)
        b64 = chart_to_base64(result.summary)

        state["saved"] = True
        self._stable[session_id] = state

        return True, str(cap_file), str(chart_file), b64

    def _get_session_classifier(self, session_id: str) -> WasteClassifier:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = WasteClassifier(
                    smooth_frames=self._settings.smooth_frames
                )
            return self._sessions[session_id]


def _aggregate_from_detections(
    labeled: List[LabeledDetection],
) -> ClassificationResult:
    """검출된 물체들의 재질을 합쳐 전체 비율 산출."""
    detail_acc = {label: 0.0 for label in MATERIAL_LABELS}
    total_w = 0.0
    for det in labeled:
        w = det.confidence
        total_w += w
        if det.material in detail_acc:
            detail_acc[det.material] += w

    if total_w < 1e-6:
        detail = {label: 20.0 for label in MATERIAL_LABELS}
    else:
        detail = {
            label: round(detail_acc[label] / total_w * 100, 1)
            for label in MATERIAL_LABELS
        }
        s = sum(detail.values()) or 1.0
        detail = {k: round(v / s * 100, 1) for k, v in detail.items()}

    # 검출 물체가 금속(캔)이면 전체 요약도 금속 우선
    metal_dets = [d for d in labeled if d.material == "금속" and d.confidence >= 0.5]
    if metal_dets:
        best = max(metal_dets, key=lambda d: d.confidence)
        detail["금속"] = max(detail["금속"], round(best.confidence * 100, 1))
        remain = 100.0 - detail["금속"]
        others = [k for k in MATERIAL_LABELS if k != "금속"]
        other_sum = sum(detail[k] for k in others) or 1.0
        for k in others:
            detail[k] = round(detail[k] / other_sum * remain, 1)
        detail["금속"] = round(100.0 - sum(detail[k] for k in others), 1)

    summary = to_summary(detail)
    primary = max(MATERIAL_LABELS, key=lambda k: detail[k])
    top_det = max(labeled, key=lambda d: d.confidence)
    return ClassificationResult(
        summary=summary,
        detail=detail,
        primary_material=primary,
        confidence=round(detail[primary] / 100.0, 4),
        waste_type_id="aggregated",
        waste_type_ko=top_det.waste_type_ko or top_det.object_name_ko,
    )


def _draw_simple_boxes(frame: np.ndarray, labeled: List[LabeledDetection]) -> np.ndarray:
    colors = {
        "플라스틱": (92, 184, 92),
        "유리": (70, 120, 50),
        "금속": (60, 140, 200),
        "종이": (80, 160, 220),
        "기타": (190, 190, 190),
    }
    for det in labeled:
        x1, y1, x2, y2 = det.bbox
        c = colors.get(det.material, (100, 220, 100))
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = f"{det.waste_type_ko or det.object_name_ko} ({det.material})"
        cv2.putText(
            frame, label, (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA,
        )
    return frame


def _merge_gemini_result(
    local: ClassificationResult, gemini: GeminiAnalysisResult
) -> ClassificationResult:
    
    detail = dict(local.detail)
    summary = dict(local.summary)

    primary = (
        gemini.material
        if gemini.material in MATERIAL_LABELS
        else local.primary_material
    )
   
    return ClassificationResult(
        summary=summary,
        detail=detail,
        primary_material=primary,
        confidence=max(local.confidence, 0.75),
        waste_type_id="gemini",
        waste_type_ko=gemini.waste_type_ko,
    )


def _to_response(
    result: ClassificationResult,
    session_id: str,
    labeled: Optional[List[LabeledDetection]] = None,
    *,
    capture_path: Optional[str] = None,
    chart_path: Optional[str] = None,
    chart_base64: Optional[str] = None,
    locked: bool = False,
    gemini_result: Optional[GeminiAnalysisResult] = None,
    gemini_error: Optional[str] = None,
    ai_source: Optional[str] = None,
) -> MaterialAnalyzeResponse:
    if labeled is None:
        labeled = []
    summary = [
        MaterialRatio(label=label, percent=result.summary[label])
        for label in SUMMARY_LABELS
    ]
    detail = [
        MaterialRatio(label=label, percent=result.detail[label])
        for label in MATERIAL_LABELS
    ]
    detections = [
        ObjectDetection(
            object_name=d.object_name,
            object_name_ko=d.object_name_ko,
            waste_type_ko=d.waste_type_ko or d.object_name_ko,
            material=d.material,
            confidence=d.confidence,
            bbox=list(d.bbox),
            corners=[[c[0], c[1]] for c in d.corners],
        )
        for d in labeled
    ]

    ai_on = gemini_result is not None
    warnings = list(gemini_result.warnings if gemini_result else [])
    if ai_source == "local" and gemini_error == "quota":
        warnings = [
            "Gemini API 할당량 초과 — 로컬 AI(규칙·화면 분석)로 안내합니다.",
            *warnings,
        ]
    return MaterialAnalyzeResponse(
        summary=summary,
        detail=detail,
        primary_material=result.primary_material,
        waste_type_ko=result.waste_type_ko,
        confidence=result.confidence,
        session_id=session_id,
        detections=detections,
        locked=locked,
        capture_path=capture_path,
        chart_path=chart_path,
        chart_image_base64=chart_base64,
        ai_enabled=ai_on,
        ai_source=ai_source,
        ai_summary=gemini_result.summary if gemini_result else None,
        contamination=gemini_result.contamination if gemini_result else None,
        recyclable=gemini_result.recyclable if gemini_result else None,
        disposal_steps=gemini_result.disposal_steps if gemini_result else [],
        warnings=warnings,
        ai_error=None,
    )
