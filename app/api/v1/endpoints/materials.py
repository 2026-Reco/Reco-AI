import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from app.api.deps import AnalysisServiceDep, GeminiServiceDep, get_settings
from app.core.config import Settings
from app.schemas.gemini_analysis import GeminiAnalysisResult
from app.schemas.material import MaterialAnalyzeResponse, MessageResponse
from app.services.waste_taxonomy import MATERIAL_LABELS, SUMMARY_LABELS, to_summary

router = APIRouter(prefix="/materials", tags=["materials"])


@router.post("/analyze", response_model=MaterialAnalyzeResponse)
async def analyze_material(
    service: AnalysisServiceDep,
    gemini: GeminiServiceDep,
    settings: Settings = Depends(get_settings),
    image: UploadFile = File(..., description="카메라/업로드 이미지 (JPEG, PNG)"),
    x_session_id: Optional[str] = Header(
        default=None,
        description="연속 촬영 시 동일 세션 ID를 사용하면 비율이 평활화됩니다.",
    ),
    use_gemini: bool = True,
) -> MaterialAnalyzeResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        run_gemini = use_gemini and settings.gemini_enabled and gemini.enabled
        return service.analyze(
            data,
            session_id=x_session_id,
            gemini=gemini if run_gemini else None,
            mime_type=image.content_type or "image/jpeg",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reanalyze")
async def reanalyze_material(
    gemini: GeminiServiceDep,
    previous_result: str = Form(...),
    additional_answers: str = Form("[]"),
    question_type: str = Form("general_reanalysis"),
    image: UploadFile | None = File(None),
):
    previous_result_dict = _safe_json_object(previous_result)
    additional_answers_list = _safe_json_list(additional_answers)
    image_bytes = await image.read() if image else None
    image_mime_type = image.content_type if image else None

    try:
        result = gemini.reanalyze(
            previous_result=previous_result_dict,
            additional_answers=additional_answers_list,
            question_type=question_type,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
        )
        return _reanalyze_response_from_gemini(result)
    except Exception:
        return _normalize_reanalyze_response(previous_result_dict)


@router.delete("/sessions/{session_id}", response_model=MessageResponse)
def reset_session(
    session_id: str,
    service: AnalysisServiceDep,
) -> MessageResponse:
    if service.reset_session(session_id):
        return MessageResponse(message="세션 분석 기록이 초기화되었습니다.")
    raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")


def _safe_json_object(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_json_list(raw: str) -> list[Any]:
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _reanalyze_response_from_gemini(result: GeminiAnalysisResult) -> dict[str, Any]:
    primary_material = result.material or "기타"
    detail = _detail_from_material(primary_material)
    waste_type = result.waste_type_ko or "미확인"
    return _normalize_reanalyze_response(
        {
            "waste_type_ko": waste_type,
            "item": waste_type,
            "itemName": waste_type,
            "primary_material": primary_material,
            "material": primary_material,
            "confidence": 0.85,
            "summary": _ratios_from_dict(to_summary(detail), SUMMARY_LABELS),
            "detail": _ratios_from_dict(detail, MATERIAL_LABELS),
            "contamination": result.contamination.model_dump(),
            "recyclable": result.recyclable.model_dump(),
            "disposal_steps": result.disposal_steps,
            "warnings": result.warnings,
            "ai_enabled": True,
            "ai_source": "gemini",
            "ai_summary": result.summary,
            "materials": [material.model_dump() for material in result.materials],
        }
    )


def _normalize_reanalyze_response(data: dict[str, Any]) -> dict[str, Any]:
    waste_type = str(
        data.get("waste_type_ko")
        or data.get("itemName")
        or data.get("item")
        or "미확인"
    )
    primary_material = str(
        data.get("primary_material")
        or data.get("material")
        or _first_material_name(data)
        or "기타"
    )
    detail_dict = _ratio_dict(data.get("detail"), MATERIAL_LABELS)
    if not any(detail_dict.values()):
        detail_dict = _detail_from_material(primary_material)
    summary_dict = _ratio_dict(data.get("summary"), SUMMARY_LABELS)
    if not any(summary_dict.values()):
        summary_dict = to_summary(detail_dict)

    return {
        **data,
        "waste_type_ko": waste_type,
        "item": str(data.get("item") or waste_type),
        "itemName": str(data.get("itemName") or waste_type),
        "primary_material": primary_material,
        "material": str(data.get("material") or primary_material),
        "confidence": _safe_confidence(data.get("confidence")),
        "summary": _ratios_from_dict(summary_dict, SUMMARY_LABELS),
        "detail": _ratios_from_dict(detail_dict, MATERIAL_LABELS),
        "contamination": data.get("contamination") or {
            "level": "low",
            "score": 30,
            "detail": "오염 상태를 확인할 수 없습니다.",
        },
        "recyclable": data.get("recyclable") or {
            "possible": False,
            "label": "확인 필요",
            "reason": "이전 분석 결과를 기반으로 한 재분석 fallback입니다.",
        },
        "disposal_steps": _string_list(data.get("disposal_steps")),
        "warnings": _string_list(data.get("warnings")),
        "ai_enabled": bool(data.get("ai_enabled", False)),
        "ai_source": data.get("ai_source") or "fallback",
        "ai_summary": (
            data.get("ai_summary")
            or data.get("summary_text")
            or (data.get("summary") if isinstance(data.get("summary"), str) else "")
        ),
    }


def _first_material_name(data: dict[str, Any]) -> str | None:
    materials = data.get("materials")
    if not isinstance(materials, list) or not materials:
        return None
    first = materials[0]
    if isinstance(first, dict):
        return str(first.get("name") or "") or None
    return None


def _detail_from_material(material: str) -> dict[str, float]:
    detail = {label: 0.0 for label in MATERIAL_LABELS}
    detail[material if material in MATERIAL_LABELS else "기타"] = 100.0
    return detail


def _ratio_dict(value: Any, labels: tuple[str, ...]) -> dict[str, float]:
    ratios = {label: 0.0 for label in labels}
    if isinstance(value, dict):
        for label in labels:
            ratios[label] = _safe_percent(value.get(label))
    elif isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            label = item.get("label") or item.get("name")
            if label in ratios:
                ratios[label] = _safe_percent(item.get("percent"))
    return ratios


def _ratios_from_dict(value: dict[str, float], labels: tuple[str, ...]) -> list[dict[str, float | str]]:
    return [{"label": label, "percent": _safe_percent(value.get(label))} for label in labels]


def _safe_percent(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _safe_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
