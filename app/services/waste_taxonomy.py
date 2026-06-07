"""쓰레기 종류 정의 — CLIP 프롬프트 + 재질 매핑."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

MATERIAL_LABELS: Tuple[str, ...] = ("플라스틱", "유리", "종이", "금속", "기타")
SUMMARY_LABELS: Tuple[str, ...] = ("플라스틱", "유리", "금속", "기타")

# TrashNet 6-class → (material, waste_type_ko, type_id)
TRASHNET_CLASS_MAP: Dict[str, Tuple[str, str, str]] = {
    "cardboard": ("종이", "골판지", "cardboard"),
    "glass": ("유리", "유리병", "glass_bottle"),
    "metal": ("금속", "알루미늄 캔", "aluminum_can"),
    "paper": ("종이", "종이", "paper"),
    "plastic": ("플라스틱", "페트병", "pet_bottle"),
    "trash": ("기타", "일반 쓰레기", "trash"),
}


@dataclass(frozen=True)
class WasteCategory:
    type_id: str
    material: str
    name_ko: str
    prompts: Tuple[str, ...]


# 세부 종류별 CLIP 텍스트 (영문 — CLIP 사전학습 기준)
WASTE_CATEGORIES: Tuple[WasteCategory, ...] = (
    WasteCategory(
        "pet_bottle",
        "플라스틱",
        "페트병",
        (
            "a plastic PET water bottle for recycling",
            "a transparent plastic drink bottle",
            "a disposable plastic beverage bottle",
        ),
    ),
    WasteCategory(
        "plastic_cup",
        "플라스틱",
        "플라스틱 컵",
        (
            "a disposable plastic cup",
            "a white plastic coffee cup",
            "a plastic takeaway cup",
        ),
    ),
    WasteCategory(
        "plastic_bag",
        "플라스틱",
        "비닐봉투",
        (
            "a crumpled plastic shopping bag",
            "a polyethylene plastic bag",
            "a transparent plastic wrapper",
        ),
    ),
    WasteCategory(
        "aluminum_can",
        "금속",
        "알루미늄 캔",
        (
            "an aluminum soda can",
            "a silver metal beverage can",
            "a crushed aluminum drink can",
        ),
    ),
    WasteCategory(
        "steel_can",
        "금속",
        "철캔",
        (
            "a steel food can",
            "a metal tin can",
            "a canned food metal container",
        ),
    ),
    WasteCategory(
        "glass_bottle",
        "유리",
        "유리병",
        (
            "a glass bottle for drinks",
            "an empty clear glass bottle",
            "a green glass beer bottle",
        ),
    ),
    WasteCategory(
        "glass_jar",
        "유리",
        "유리 Jar",
        (
            "a glass jar with lid",
            "a transparent glass food jar",
        ),
    ),
    WasteCategory(
        "paper",
        "종이",
        "종이",
        (
            "white paper sheet for recycling",
            "stack of office paper",
            "a piece of printed paper",
        ),
    ),
    WasteCategory(
        "cardboard",
        "종이",
        "골판지",
        (
            "a cardboard box",
            "corrugated cardboard packaging",
            "a brown cardboard carton",
        ),
    ),
    WasteCategory(
        "newspaper",
        "종이",
        "신문지",
        (
            "crumpled newspaper",
            "a stack of newsprint paper",
        ),
    ),
    WasteCategory(
        "food_waste",
        "기타",
        "음식물",
        (
            "food waste scraps",
            "leftover food on a plate",
            "organic kitchen waste",
        ),
    ),
    WasteCategory(
        "general_trash",
        "기타",
        "일반 쓰레기",
        (
            "mixed general waste",
            "unrecognizable trash item",
        ),
    ),
)


def material_index() -> Dict[str, int]:
    return {name: i for i, name in enumerate(MATERIAL_LABELS)}


def to_summary(detail: Dict[str, float]) -> Dict[str, float]:
    plastic = detail.get("플라스틱", 0.0)
    glass = detail.get("유리", 0.0)
    metal = detail.get("금속", 0.0)
    other = detail.get("종이", 0.0) + detail.get("기타", 0.0)
    total = plastic + glass + metal + other or 1.0
    p = round(plastic / total * 100, 1)
    g = round(glass / total * 100, 1)
    m = round(metal / total * 100, 1)
    o = round(100.0 - p - g - m, 1)
    return {"플라스틱": p, "유리": g, "금속": m, "기타": max(o, 0.0)}
