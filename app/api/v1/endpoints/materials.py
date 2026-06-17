import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from app.api.deps import AnalysisServiceDep, GeminiServiceDep, get_settings
from app.core.config import Settings
from app.schemas.gemini_analysis import GeminiAnalysisResult
from app.schemas.material import MaterialAnalyzeResponse, MessageResponse
from app.services.disposal_steps import format_disposal_steps, summarize_disposal_steps
from app.services.waste_taxonomy import MATERIAL_LABELS, SUMMARY_LABELS, to_summary

router = APIRouter(prefix="/materials", tags=["materials"])

UNKNOWN_ITEM_NAME = "알 수 없는 품목"
MATERIAL_NAMES = {"플라스틱", "유리", "금속", "종이", "비닐", "스티로폼", "전자부품", "고무", "섬유", "목재", "기타", "미확인", ""}


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
        return _reanalyze_response_from_gemini(result, previous_result_dict)
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


def _reanalyze_response_from_gemini(
    result: GeminiAnalysisResult,
    previous_result: dict[str, Any],
) -> dict[str, Any]:
    primary_material = result.material or "기타"
    detail = _detail_from_material(primary_material)
    waste_type = _normalize_item_name(result.item_name or result.waste_type_ko)
    new_data = {
        "item_name": waste_type,
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
    return _normalize_reanalyze_response(
        new_data,
        previous_result=previous_result,
        is_new_result=True,
    )


def _normalize_reanalyze_response(
    data: dict[str, Any],
    *,
    previous_result: dict[str, Any] | None = None,
    is_new_result: bool = False,
) -> dict[str, Any]:
    waste_type = _normalize_item_name(
        data.get("item_name")
        or data.get("waste_type_ko")
        or data.get("itemName")
        or data.get("item")
    )
    primary_material = str(
        data.get("primary_material")
        or data.get("material")
        or _first_material_name(data)
        or "기타"
    )
    if primary_material in MATERIAL_NAMES:
        primary_material = primary_material or "기타"
    else:
        primary_material = "기타"
    material = str(data.get("material") or primary_material)
    if material not in MATERIAL_NAMES:
        material = primary_material
    if material == "":
        material = "기타"

    legacy_waste_type = str(
        data.get("waste_type_ko")
        or data.get("itemName")
        or data.get("item")
        or ""
    )
    if waste_type == UNKNOWN_ITEM_NAME and _normalize_item_name(legacy_waste_type) != UNKNOWN_ITEM_NAME:
        waste_type = _normalize_item_name(legacy_waste_type)

    detail_dict = _ratio_dict(data.get("detail"), MATERIAL_LABELS)
    if not any(detail_dict.values()):
        detail_dict = _detail_from_material(primary_material)
    summary_dict = _ratio_dict(data.get("summary"), SUMMARY_LABELS)
    if not any(summary_dict.values()):
        summary_dict = to_summary(detail_dict)
    contamination = data.get("contamination") or {
        "level": "low",
        "score": 30,
        "detail": "오염 상태를 확인할 수 없습니다.",
    }
    recyclable = data.get("recyclable") or {
        "possible": False,
        "label": "확인 필요",
        "reason": "이전 분석 결과를 기반으로 한 재분석 fallback입니다.",
    }
    changed = (
        is_new_result
        and previous_result is not None
        and _analysis_core_changed(previous_result, data)
    )
    disposal_steps = _reanalyze_disposal_steps(
        item_name=waste_type,
        material=primary_material,
        recyclable=recyclable,
        new_steps=_string_list(data.get("disposal_steps")),
        previous_steps=_string_list((previous_result or {}).get("disposal_steps")),
        changed=changed,
        is_new_result=is_new_result,
    )

    return {
        **data,
        "item_name": waste_type,
        "waste_type_ko": waste_type,
        "item": str(data.get("item") or waste_type),
        "itemName": str(data.get("itemName") or waste_type),
        "primary_material": primary_material,
        "material": material,
        "confidence": _safe_confidence(data.get("confidence")),
        "summary": _ratios_from_dict(summary_dict, SUMMARY_LABELS),
        "detail": _ratios_from_dict(detail_dict, MATERIAL_LABELS),
        "materialProbabilities": _material_probabilities(data, primary_material),
        "contamination": contamination,
        "recyclable": recyclable,
        "disposal_steps": disposal_steps,
        "warnings": _string_list(data.get("warnings")),
        "ai_enabled": bool(data.get("ai_enabled", False)),
        "ai_source": data.get("ai_source") or "fallback",
        "ai_summary": (
            summarize_disposal_steps(waste_type, primary_material)
            or data.get("ai_summary")
            or data.get("summary_text")
            or (data.get("summary") if isinstance(data.get("summary"), str) else "")
        ),
    }


def _normalize_item_name(value: Any) -> str:
    item_name = str(value or "").strip()
    if item_name in MATERIAL_NAMES:
        return UNKNOWN_ITEM_NAME
    return item_name or UNKNOWN_ITEM_NAME


def _analysis_core_changed(
    previous: dict[str, Any],
    new: dict[str, Any],
) -> bool:
    fields = (
        ("item_name", _item_name_of(previous), _item_name_of(new)),
        ("waste_type_ko", _normalize_item_name(previous.get("waste_type_ko")), _normalize_item_name(new.get("waste_type_ko"))),
        ("primary_material", _material_of(previous, "primary_material"), _material_of(new, "primary_material")),
        ("material", _material_of(previous, "material"), _material_of(new, "material")),
        ("contamination.level", _contamination_level(previous), _contamination_level(new)),
        ("recyclable.possible", _recyclable_possible(previous), _recyclable_possible(new)),
    )
    return any(prev != cur for _, prev, cur in fields)


def _reanalyze_disposal_steps(
    *,
    item_name: str,
    material: str,
    recyclable: Any,
    new_steps: list[str],
    previous_steps: list[str],
    changed: bool,
    is_new_result: bool,
) -> list[str]:
    recyclable_possible = _recyclable_possible({"recyclable": recyclable})

    if is_new_result and new_steps:
        return format_disposal_steps(
            item_name,
            material,
            new_steps,
            recyclable_possible=recyclable_possible,
        )

    if is_new_result and changed:
        return format_disposal_steps(
            item_name,
            material,
            None,
            recyclable_possible=recyclable_possible,
        )

    fallback_steps = new_steps or previous_steps
    return format_disposal_steps(
        item_name,
        material,
        fallback_steps,
        recyclable_possible=recyclable_possible,
    )


def _item_name_of(data: dict[str, Any]) -> str:
    return _normalize_item_name(
        data.get("item_name")
        or data.get("waste_type_ko")
        or data.get("itemName")
        or data.get("item")
    )


def _material_of(data: dict[str, Any], key: str) -> str:
    value = str(data.get(key) or "").strip()
    if value:
        return value
    if key == "primary_material":
        return str(data.get("material") or _first_material_name(data) or "기타")
    return str(data.get("primary_material") or _first_material_name(data) or "기타")


def _contamination_level(data: dict[str, Any]) -> str:
    contamination = data.get("contamination")
    if isinstance(contamination, dict):
        return str(contamination.get("level") or "").strip()
    return ""


def _recyclable_possible(data: dict[str, Any]) -> bool | None:
    recyclable = data.get("recyclable")
    if not isinstance(recyclable, dict) or "possible" not in recyclable:
        return None
    value = recyclable.get("possible")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    return bool(value)


def _material_probabilities(data: dict[str, Any], primary_material: str) -> list[dict[str, float | str]]:
    materials = data.get("materials")
    if isinstance(materials, list) and materials:
        ratios = []
        for item in materials[:3]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("name") or item.get("label") or "").strip()
            if not label:
                continue
            ratios.append(
                {
                    "label": label,
                    "percent": _safe_percent(item.get("percentage", item.get("percent"))),
                }
            )
        if ratios:
            ratios.sort(key=lambda item: float(item["percent"]), reverse=True)
            if ratios[0]["label"] == primary_material:
                return ratios
    return [{"label": primary_material, "percent": 100.0}]


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
