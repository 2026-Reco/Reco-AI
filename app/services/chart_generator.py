"""재질 분석 도넛 차트 이미지 생성."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = [
    "Apple SD Gothic Neo",
    "Malgun Gothic",
    "NanumGothic",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# 목업 UI 색상 (플라스틱·유리·금속·기타)
CHART_COLORS = {
    "플라스틱": "#5CB85C",
    "유리": "#2E6B4E",
    "금속": "#2E6B4E",
    "기타": "#D9D9D9",
}

CHART_ORDER = ("플라스틱", "유리", "금속", "기타")


def render_donut_chart(
    summary: Dict[str, float],
    title: str = "재질 분석",
) -> bytes:
    labels: List[str] = []
    sizes: List[float] = []
    colors: List[str] = []

    for name in CHART_ORDER:
        v = max(summary.get(name, 0.0), 0.0)
        if v <= 0.1:
            continue
        labels.append(f"{name}\n{v:.0f}%")
        sizes.append(v)
        colors.append(CHART_COLORS[name])

    if not sizes:
        sizes, labels, colors = [100.0], ["기타\n100%"], [CHART_COLORS["기타"]]

    fig, ax = plt.subplots(figsize=(4.2, 3.8), facecolor="white")
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops={"fontsize": 9},
    )
    fig.text(0.06, 0.94, title, fontsize=14, fontweight="bold", va="top")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def save_donut_chart(summary: Dict[str, float], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_donut_chart(summary))
    return path


def chart_to_base64(summary: Dict[str, float]) -> str:
    return base64.b64encode(render_donut_chart(summary)).decode("ascii")
