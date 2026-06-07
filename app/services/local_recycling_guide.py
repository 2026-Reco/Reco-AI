"""Gemini 실패 시 로컬 규칙 + 간단 비전 휴리스틱으로 재활용 안내 생성."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.schemas.gemini_analysis import (
    ContaminationInfo,
    GeminiAnalysisResult,
    RecyclableInfo,
)

# waste_type_ko 또는 material 키로 조회
_RULES: Dict[str, Dict[str, Any]] = {
    "페트병": {
        "material": "플라스틱",
        "recyclable": (True, "조건부 가능", "라벨·뚜껑 분리 후 깨끗하면 재활용 가능"),
        "steps": [
            "내용물을 완전히 비우고 헹굽니다.",
            "라벨과 뚜껑을 분리합니다 (재질이 다를 수 있음).",
            "물기를 제거한 뒤 플라스틱류로 배출합니다.",
            "찌그러뜨려 부피를 줄이면 수거에 도움이 됩니다.",
        ],
        "warnings": ["기름·음료 찌꺼기가 남으면 재활용 불가입니다."],
    },
    "플라스틱 컵": {
        "material": "플라스틱",
        "recyclable": (False, "재활용 불가", "일회용 컵은 지자체별로 일반쓰레기 처리"),
        "steps": [
            "내용물을 비우고 가볍게 헹굽니다.",
            "지역 분리배출 안내에 따라 일반쓰레기 또는 플라스틱으로 배출합니다.",
        ],
        "warnings": ["코팅·스트로 홀은 재질 혼합으로 재활용이 어려울 수 있습니다."],
    },
    "비닐봉투": {
        "material": "플라스틱",
        "recyclable": (True, "조건부 가능", "깨끗한 비닐만 비닐류 전용 수거함"),
        "steps": [
            "이물질·음식물을 제거하고 물기를 말립니다.",
            "비닐류 전용 수거함에 모아 배출합니다.",
        ],
        "warnings": ["오염된 비닐·카라비너 봉지는 일반쓰레기입니다."],
    },
    "알루미늄 캔": {
        "material": "금속",
        "recyclable": (True, "재활용 가능", "알루미늄 캔은 캔류로 분리배출"),
        "steps": [
            "내용물을 비우고 헹굽니다.",
            "라벨이 있으면 제거합니다.",
            "캔류(금속) 수거함에 배출합니다.",
        ],
        "warnings": ["담배·기름 오염 시 재활용이 어렵습니다."],
    },
    "철캔": {
        "material": "금속",
        "recyclable": (True, "재활용 가능", "철캔·통조림 캔은 캔류"),
        "steps": [
            "내용물을 비우고 헹굽니다.",
            "뚜껑을 분리해 함께 캔류로 배출합니다.",
        ],
        "warnings": [],
    },
    "유리병": {
        "material": "유리",
        "recyclable": (True, "재활용 가능", "색 없는 투명 유리병 우선 재활용"),
        "steps": [
            "내용물을 비우고 헹굽니다.",
            "금속 뚜껑·코르크는 분리합니다.",
            "유리병 전용 수거함에 배출합니다.",
        ],
        "warnings": ["깨진 유리는 일반쓰레기 봉투에 안전하게 포장하세요."],
    },
    "종이": {
        "material": "종이",
        "recyclable": (True, "재활용 가능", "깨끗한 종이·서류류"),
        "steps": [
            "스테이플러·테이프 등 이물질을 제거합니다.",
            "비닐 코팅·오염된 부분은 떼어 일반쓰레기로 버립니다.",
            "종이류 수거함에 배출합니다.",
        ],
        "warnings": ["음식 오염·코팅지는 재활용 불가입니다."],
    },
    "골판지": {
        "material": "종이",
        "recyclable": (True, "재활용 가능", "골판지는 종이류"),
        "steps": [
            "테이프·스티로폼 패킹을 제거합니다.",
            "박스를 펼쳐 부피를 줄입니다.",
            "종이류로 배출합니다.",
        ],
        "warnings": [],
    },
    "음식물": {
        "material": "기타",
        "recyclable": (False, "재활용 불가", "음식물 쓰레기 전용 배출"),
        "steps": [
            "음식물 쓰레기 전용 봉투·수거함에 배출합니다.",
            "물기를 제거하면 부패 냄새를 줄일 수 있습니다.",
        ],
        "warnings": ["일반 재활용함에 넣지 마세요."],
    },
}

_MATERIAL_DEFAULT: Dict[str, Dict[str, Any]] = {
    "플라스틱": {
        "recyclable": (True, "조건부 가능", "깨끗한 플라스틱 포장재만 재활용"),
        "steps": [
            "내용물을 비우고 헹굽니다.",
            "라벨·뚜껑 재질을 확인해 분리합니다.",
            "플라스틱류 수거함에 배출합니다.",
        ],
        "warnings": ["오염·복합재질은 재활용 불가일 수 있습니다."],
    },
    "금속": {
        "recyclable": (True, "재활용 가능", "캔·철제 용기는 캔류"),
        "steps": ["내용물을 비우고 헹굽니다.", "캔류로 배출합니다."],
        "warnings": [],
    },
    "유리": {
        "recyclable": (True, "재활용 가능", "유리병 전용 수거"),
        "steps": ["내용물을 비우고 헹굽니다.", "유리병 수거함에 배출합니다."],
        "warnings": ["깨진 유리는 별도 안전 포장"],
    },
    "종이": {
        "recyclable": (True, "재활용 가능", "깨끗한 종이류"),
        "steps": ["이물질을 제거합니다.", "종이류로 배출합니다."],
        "warnings": ["코팅·오염 종이는 제외"],
    },
    "기타": {
        "recyclable": (False, "재활용 불가", "일반쓰레기 또는 지역 안내 확인"),
        "steps": ["지역 주민센터·분리배출 안내를 확인합니다."],
        "warnings": [],
    },
}


def _match_rule(waste_type_ko: str, material: str) -> Dict[str, Any]:
    name = (waste_type_ko or "").strip()
    for key, rule in _RULES.items():
        if key in name or name in key:
            return rule
    return _MATERIAL_DEFAULT.get(material, _MATERIAL_DEFAULT["기타"])


def _estimate_contamination_from_image(frame: np.ndarray) -> Tuple[str, float, str]:
    """HSV 기반 간이 오염 추정 (음식물·기름 색상 비율)."""
    h, w = frame.shape[:2]
    y1, y2 = int(h * 0.15), int(h * 0.85)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "low", 25.0, "촬영 영역이 작습니다. 내용물을 비우고 헹군 뒤 배출하세요."

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 갈색·노란 음식물/기름 톤
    mask1 = cv2.inRange(hsv, np.array([8, 40, 40]), np.array([35, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([10, 255, 200]))
    stain = cv2.bitwise_or(mask1, mask2)
    ratio = float(np.count_nonzero(stain)) / max(stain.size, 1)

    if ratio > 0.12:
        return (
            "high",
            min(95.0, 50 + ratio * 200),
            "음식물·기름 등 오염이 의심됩니다. 헹굼·세척 후에도 오염이 남으면 일반쓰레기로 배출하세요.",
        )
    if ratio > 0.04:
        return (
            "low",
            30 + ratio * 100,
            "경미한 얼룩이 보일 수 있습니다. 내용물을 비우고 깨끗이 헹군 뒤 배출하세요.",
        )
    return (
        "clean",
        max(5.0, ratio * 50),
        "표면이 비교적 깨끗해 보입니다. 내용물만 비우고 배출하면 재활용에 유리합니다.",
    )


def build_local_analysis(
    waste_type_ko: str,
    material: str,
    frame: Optional[np.ndarray] = None,
) -> GeminiAnalysisResult:
    rule = _match_rule(waste_type_ko, material)
    mat = rule.get("material", material)

    if frame is not None:
        level, score, detail = _estimate_contamination_from_image(frame)
    else:
        level, score, detail = "low", 25.0, "사진 기준으로는 경미 오염으로 추정됩니다. 배출 전 헹굼을 권장합니다."

    possible, label, reason = rule["recyclable"]
    if level == "high" and possible:
        label = "조건부 가능"
        reason = "오염이 심하면 재활용이 불가합니다. 세척 후 재활용 가능 여부를 판단하세요."

    summary = f"{waste_type_ko} — {label}. {detail[:40]}…" if len(detail) > 40 else f"{waste_type_ko} — {label}. {detail}"

    return GeminiAnalysisResult(
        waste_type_ko=waste_type_ko or "미확인",
        material=mat if mat in ("플라스틱", "유리", "종이", "금속", "기타") else material,
        contamination=ContaminationInfo(level=level, score=round(score, 1), detail=detail),
        recyclable=RecyclableInfo(possible=possible, label=label, reason=reason),
        disposal_steps=list(rule.get("steps", [])),
        warnings=list(rule.get("warnings", [])),
        summary=summary,
    )
