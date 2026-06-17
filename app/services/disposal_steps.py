from __future__ import annotations

from typing import Iterable


DEFAULT_ITEM_NAME = "알 수 없는 품목"
MAX_DESCRIPTION_LENGTH = 25


_ITEM_STEP_TEMPLATES: tuple[tuple[tuple[str, ...], list[tuple[str, str]], str], ...] = (
    (
        ("페트병", "플라스틱 병", "플라스틱병"),
        [
            ("내용물 완전히 비우기", "병 안의 음료를 비워요"),
            ("라벨 분리", "비닐 라벨을 제거해요"),
            ("찌그러뜨리기", "부피를 줄여요"),
            ("뚜껑 닫아서 배출", "뚜껑을 닫아 배출해요"),
        ],
        "페트병은 비우고 라벨을 분리한 뒤 찌그러뜨려 플라스틱류로 배출해요.",
    ),
    (
        ("플라스틱 컵", "플라스틱컵"),
        [
            ("내용물 비우기", "남은 음료를 비워요"),
            ("가볍게 헹구기", "컵 안쪽을 헹궈요"),
            ("다른 재질 분리", "빨대와 뚜껑을 분리해요"),
            ("플라스틱류 배출", "깨끗한 컵만 배출해요"),
        ],
        "플라스틱 컵은 비우고 헹군 뒤 다른 재질을 분리해 배출해요.",
    ),
    (
        ("종이컵",),
        [
            ("내용물 비우기", "컵 안의 음료를 비워요"),
            ("가볍게 헹구기", "안쪽을 가볍게 헹궈요"),
            ("코팅 여부 확인", "코팅과 오염을 확인해요"),
            ("종이류 구분 배출", "상태에 맞게 배출해요"),
        ],
        "종이컵은 비우고 헹군 뒤 코팅과 오염 상태에 따라 종이류나 일반쓰레기로 배출해요.",
    ),
    (
        ("알루미늄 캔", "철캔", "캔"),
        [
            ("내용물 비우기", "캔 안을 완전히 비워요"),
            ("가볍게 헹구기", "캔 안쪽을 헹궈요"),
            ("압착하기", "캔을 눌러 줄여요"),
            ("캔류로 배출", "캔류 수거함에 배출해요"),
        ],
        "캔은 비우고 헹군 뒤 압착해서 캔류로 배출해요.",
    ),
    (
        ("유리병", "유리 병"),
        [
            ("내용물 비우기", "병 안을 완전히 비워요"),
            ("뚜껑 분리", "뚜껑을 따로 분리해요"),
            ("깨진 병 구분", "깨진 병은 따로 버려요"),
            ("유리류 배출", "유리류로 배출해요"),
        ],
        "유리병은 비우고 뚜껑을 분리한 뒤 깨지지 않은 병만 유리류로 배출해요.",
    ),
    (
        ("비닐봉투", "비닐"),
        [
            ("이물질 제거", "음식물과 스티커를 떼요"),
            ("물기 제거", "젖은 비닐은 말려요"),
            ("깨끗한 비닐만 모으기", "오염된 비닐은 제외해요"),
            ("비닐류 배출", "비닐류로 배출해요"),
        ],
        "비닐봉투는 이물질과 물기를 제거한 뒤 깨끗한 것만 비닐류로 배출해요.",
    ),
    (
        ("택배상자", "골판지", "박스", "상자"),
        [
            ("테이프 제거", "테이프를 모두 떼어내요"),
            ("송장 제거", "개인정보가 있는 송장을 제거해요"),
            ("상자 펼치기", "상자를 납작하게 펼쳐요"),
            ("종이류 배출", "종이류로 배출해요"),
        ],
        "택배상자는 테이프와 송장을 제거한 뒤 펼쳐서 종이류로 배출해요.",
    ),
    (
        ("종이", "신문지"),
        [
            ("이물질 제거", "테이프와 비닐을 떼요"),
            ("오염 확인", "젖거나 묻은 부분을 빼요"),
            ("반듯하게 모으기", "종이를 가지런히 모아요"),
            ("종이류 배출", "종이류로 배출해요"),
        ],
        "종이는 이물질과 오염된 부분을 제거한 뒤 깨끗한 것만 종이류로 배출해요.",
    ),
)


_MATERIAL_STEP_TEMPLATES: dict[str, tuple[list[tuple[str, str]], str]] = {
    "플라스틱": (
        [
            ("내용물 비우기", "용기 안을 비워요"),
            ("가볍게 헹구기", "안쪽을 가볍게 헹궈요"),
            ("다른 재질 분리", "라벨과 뚜껑을 분리해요"),
            ("플라스틱류 배출", "플라스틱류로 배출해요"),
        ],
        "플라스틱류는 비우고 헹군 뒤 다른 재질을 분리해 배출해요.",
    ),
    "금속": (
        [
            ("내용물 비우기", "안쪽을 완전히 비워요"),
            ("가볍게 헹구기", "안쪽을 가볍게 헹궈요"),
            ("부피 줄이기", "눌러서 부피를 줄여요"),
            ("금속류 배출", "금속류로 배출해요"),
        ],
        "금속류는 비우고 헹군 뒤 가능한 압착해 금속류로 배출해요.",
    ),
    "유리": (
        [
            ("내용물 비우기", "안쪽을 완전히 비워요"),
            ("뚜껑 분리", "뚜껑을 따로 분리해요"),
            ("파손 여부 확인", "깨진 유리는 따로 버려요"),
            ("유리류 배출", "유리류로 배출해요"),
        ],
        "유리류는 비우고 부속품을 분리한 뒤 깨지지 않은 것만 배출해요.",
    ),
    "종이": (
        [
            ("이물질 제거", "테이프와 비닐을 떼요"),
            ("오염 확인", "젖은 부분은 제외해요"),
            ("납작하게 정리", "펼치거나 접어 정리해요"),
            ("종이류 배출", "종이류로 배출해요"),
        ],
        "종이류는 이물질과 오염을 제거한 뒤 깨끗한 것만 배출해요.",
    ),
}


def format_disposal_steps(
    item_name: str,
    material: str,
    provided_steps: Iterable[str] | None = None,
    recyclable_possible: bool | None = None,
) -> list[str]:
    item = (item_name or "").strip()
    cleaned = [step.strip() for step in (provided_steps or []) if str(step).strip()]
    if len(cleaned) >= 4 and all(_is_compact_step(step) for step in cleaned[:4]):
        return cleaned[:4]

    if recyclable_possible is False and _is_unknown_item(item):
        return _numbered_steps(_general_waste_steps())

    steps, _ = _template_for(item, material)
    if steps:
        return _numbered_steps(steps)

    return _numbered_steps(
        [
            ("품목 확인", f"{item or DEFAULT_ITEM_NAME}의 재질과 오염 상태를 확인해요"),
            ("이물질 제거", "먼지나 오염을 닦아요"),
            ("분리 가능 부품 분리", "다른 재질 부품은 따로 떼어내요"),
            ("지역 기준에 맞게 배출", "지역 기준을 확인해요"),
        ]
    )


def summarize_disposal_steps(item_name: str, material: str) -> str:
    item = (item_name or "").strip()
    _, summary = _template_for(item, material)
    if summary:
        return summary
    return f"{item or DEFAULT_ITEM_NAME}은 이물질을 제거하고 재질별 분리배출 기준에 맞게 배출해요."


def _template_for(
    item_name: str,
    material: str,
) -> tuple[list[tuple[str, str]] | None, str | None]:
    for keywords, steps, summary in _ITEM_STEP_TEMPLATES:
        if any(keyword in item_name for keyword in keywords):
            return steps, summary
    material_steps = _MATERIAL_STEP_TEMPLATES.get((material or "").strip())
    if material_steps:
        return material_steps
    return None, None


def _numbered_steps(steps: list[tuple[str, str]]) -> list[str]:
    return [
        f"{idx}. {title}\n→{description}"
        for idx, (title, description) in enumerate(steps[:4], start=1)
    ]


def _general_waste_steps() -> list[tuple[str, str]]:
    return [
        ("이물질 제거", "먼지나 오염을 닦아요"),
        ("부속품 분리", "금속 부품은 분리해요"),
        ("재질 확인", "복합 재질인지 확인해요"),
        ("일반쓰레기 배출", "종량제 봉투에 버려요"),
    ]


def _is_unknown_item(item_name: str) -> bool:
    known_keywords = [
        keyword
        for keywords, _, _ in _ITEM_STEP_TEMPLATES
        for keyword in keywords
    ]
    return not any(keyword in item_name for keyword in known_keywords)


def _is_compact_step(step: str) -> bool:
    if "\n" not in step or "→" not in step:
        return False
    description = step.split("→", 1)[1].strip()
    return bool(description) and "\n" not in description and len(description) <= MAX_DESCRIPTION_LENGTH
