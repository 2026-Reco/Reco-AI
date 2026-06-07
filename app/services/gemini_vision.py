"""Google Gemini Vision — 오염도·재활용·분리배출 안내."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from app.schemas.gemini_analysis import (
    ContaminationInfo,
    GeminiAnalysisResult,
    RecyclableInfo,
)

_SYSTEM_PROMPT = """당신은 대한민국 분리수거·재활용 전문가입니다.
사용자가 보낸 쓰레기 사진을 보고 반드시 아래 JSON 형식만 출력하세요. 다른 텍스트는 금지합니다.

판단 기준:
- 오염도: 음식물·기름·이물질이 묻었는지 (clean/low/high)
- 재활용: 한국 주택·상가 분리배출 기준 (가능/불가/조건부)
- disposal_steps: 이 사진 속 물체에 맞는 구체적 행동 3~6단계 (라벨 제거, 헹굼 등)

material 값은 반드시 다음 중 하나: 플라스틱, 유리, 종이, 금속, 기타"""


def _build_user_prompt(local_hint: Optional[Dict[str, Any]]) -> str:
    hint = ""
    if local_hint:
        hint = (
            f"\n[로컬 AI 참고] 종류={local_hint.get('waste_type_ko')}, "
            f"재질={local_hint.get('primary_material')}, "
            f"검출={local_hint.get('detections_count', 0)}개"
        )
    return f"""이 사진의 쓰레기를 분석해 주세요.{hint}

JSON 스키마:
{{
  "waste_type_ko": "예: 알루미늄 캔",
  "material": "플라스틱|유리|종이|금속|기타",
  "contamination": {{
    "level": "clean|low|high",
    "score": 0,
    "detail": "오염 상태 설명"
  }},
  "recyclable": {{
    "possible": true,
    "label": "재활용 가능|재활용 불가|조건부 가능",
    "reason": "근거"
  }},
  "disposal_steps": ["1. ...", "2. ..."],
  "warnings": ["주의사항"],
  "summary": "한 줄 요약"
}}"""


_FALLBACK_MODELS = (
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash",
)


class GeminiVisionService:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model_name = model
        self._models: Dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _get_model(self, model_name: str):
        if model_name in self._models:
            return self._models[model_name]
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self._models[model_name] = genai.GenerativeModel(
            model_name,
            system_instruction=_SYSTEM_PROMPT,
        )
        return self._models[model_name]

    def _model_candidates(self) -> tuple:
        seen = set()
        out = []
        for name in (self.model_name,) + _FALLBACK_MODELS:
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return tuple(out)

    def _call_model(
        self,
        model_name: str,
        image_bytes: bytes,
        mime_type: str,
        local_hint: Optional[Dict[str, Any]],
    ) -> GeminiAnalysisResult:
        import google.generativeai as genai

        model = self._get_model(model_name)
        image_part = {"mime_type": mime_type, "data": image_bytes}
        response = model.generate_content(
            [image_part, _build_user_prompt(local_hint)],
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        text = (response.text or "").strip()
        return _to_result(_parse_json(text))

    def analyze_image(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        local_hint: Optional[Dict[str, Any]] = None,
    ) -> GeminiAnalysisResult:
        if not self.enabled:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

        last_exc: Optional[Exception] = None
        for model_name in self._model_candidates():
            try:
                return self._call_model(
                    model_name, image_bytes, mime_type, local_hint
                )
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "404" in msg or "not found" in msg:
                    continue
                if "429" in msg or "quota" in msg:
                    continue
                if "403" in msg or "denied" in msg:
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("사용 가능한 Gemini 모델이 없습니다.")

    def analyze_image_safe(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        local_hint: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[GeminiAnalysisResult], Optional[str]]:
        try:
            return self.analyze_image(image_bytes, mime_type, local_hint), None
        except Exception as exc:
            msg = str(exc).strip()
            if "429" in msg or "quota" in msg.lower():
                return None, "quota"
            if "403" in msg or "denied" in msg.lower():
                return None, "denied"
            return None, f"gemini:{msg[:120]}"

    def reanalyze(
        self,
        previous_result: dict,
        additional_answers: list,
        question_type: str,
    ) -> GeminiAnalysisResult:
        if not self.enabled:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

        model = self._get_model(self.model_name)
        prompt = f"""
당신은 대한민국 분리배출 전문가입니다.

기존 분석 결과:
{json.dumps(previous_result, ensure_ascii=False)}

추가 질문 답변:
{json.dumps(additional_answers, ensure_ascii=False)}

질문 유형:
{question_type}

사용자의 답변을 반영하여 기존 분석 결과를 재검토하세요.
반드시 아래 JSON 형식만 반환하세요.

{{
  "waste_type_ko": "예: 페트병",
  "material": "플라스틱|유리|종이|금속|기타",
  "contamination": {{
    "level": "clean|low|high",
    "score": 0,
    "detail": "오염 상태 설명"
  }},
  "recyclable": {{
    "possible": true,
    "label": "재활용 가능",
    "reason": "근거"
  }},
  "disposal_steps": ["1. ...", "2. ..."],
  "warnings": [],
  "summary": "한 줄 요약"
}}
"""
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        return _to_result(_parse_json(text))


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        raise


def _to_result(data: Dict[str, Any]) -> GeminiAnalysisResult:
    cont = data.get("contamination") or {}
    recy = data.get("recyclable") or {}
    level = str(cont.get("level", "low")).lower()
    if level not in ("clean", "low", "high"):
        level = "low"

    material = str(data.get("material", "기타"))
    if material not in ("플라스틱", "유리", "종이", "금속", "기타"):
        material = "기타"

    return GeminiAnalysisResult(
        waste_type_ko=str(data.get("waste_type_ko", "미확인")),
        material=material,
        contamination=ContaminationInfo(
            level=level,
            score=float(cont.get("score", 30)),
            detail=str(cont.get("detail", "오염 상태를 확인할 수 없습니다.")),
        ),
        recyclable=RecyclableInfo(
            possible=bool(recy.get("possible", False)),
            label=str(recy.get("label", "재활용 불가")),
            reason=str(recy.get("reason", "")),
        ),
        disposal_steps=[str(s) for s in (data.get("disposal_steps") or []) if s],
        warnings=[str(w) for w in (data.get("warnings") or []) if w],
        summary=str(data.get("summary", "")),
    )
