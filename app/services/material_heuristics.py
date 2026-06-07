"""색·형태 기반 재질 보정 — 종이·플라스틱·캔(금속)·유리."""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from app.services.waste_taxonomy import MATERIAL_LABELS

OBJECT_MATERIAL_PRIOR: Dict[str, Dict[str, float]] = {
    "wine glass": {"유리": 0.82, "기타": 0.18},
    "book": {"종이": 0.9, "기타": 0.1},
    "cup": {"유리": 0.35, "플라스틱": 0.35, "기타": 0.3},
    "bowl": {"유리": 0.45, "플라스틱": 0.25, "기타": 0.3},
    "bottle": {},
    "banana": {"기타": 1.0},
    "apple": {"기타": 1.0},
}


def _rgb_stats(crop_bgr: np.ndarray):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    b_mean, g_mean, r_mean = [float(x) for x in crop_bgr.mean(axis=(0, 1))]
    return {
        "sat_mean": float(np.mean(s_ch)),
        "sat_std": float(np.std(s_ch)),
        "val_mean": float(np.mean(v_ch)),
        "val_std": float(np.std(v_ch)),
        "hue_std": float(np.std(h_ch)),
        "specular": float(np.sum(gray > 185)) / gray.size,
        "dark": float(np.sum(gray < 50)) / gray.size,
        "aspect": crop_bgr.shape[0] / max(crop_bgr.shape[1], 1),
        "b_mean": b_mean,
        "g_mean": g_mean,
        "r_mean": r_mean,
        "rgb_spread": max(b_mean, g_mean, r_mean) - min(b_mean, g_mean, r_mean),
    }


def visual_material_scores(crop_bgr: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(MATERIAL_LABELS), dtype=np.float64)
    idx = {n: i for i, n in enumerate(MATERIAL_LABELS)}
    h, w = crop_bgr.shape[:2]
    if h < 16 or w < 16:
        scores[idx["기타"]] = 1.0
        return scores

    s = _rgb_stats(crop_bgr)
    metal = plastic = glass = paper = other = 0.06

    # 금속 캔
    if s["sat_mean"] < 60:
        metal += 0.28
    if s["specular"] > 0.03 and s["sat_mean"] < 68:
        metal += 0.35
    if 0.85 <= s["aspect"] <= 4.5:
        metal += 0.14
    if s["rgb_spread"] < 38 and s["sat_mean"] < 52:
        metal += 0.22

    # 플라스틱 (유색 페트·플라스틱)
    if s["sat_mean"] > 70:
        plastic += 0.38
    if s["sat_mean"] > 100:
        plastic += 0.22
    if s["g_mean"] > s["r_mean"] * 1.06 and s["sat_mean"] > 45:
        plastic += 0.18
    if s["b_mean"] > s["r_mean"] * 1.08 and s["sat_mean"] > 40:
        plastic += 0.2
    if s["specular"] < 0.07 and s["sat_mean"] > 55:
        plastic += 0.12

    # 유리 (투명·반사·저채도 고대비)
    if s["sat_mean"] < 48 and s["val_std"] > 38:
        glass += 0.32
    if s["specular"] > 0.06 and s["sat_mean"] < 42:
        glass += 0.28
    if s["val_mean"] > 100 and s["hue_std"] < 30:
        glass += 0.12

    # 종이 (밝은 베이지·저채도·납작)
    if s["sat_mean"] < 55 and s["val_mean"] > 130:
        paper += 0.28
    if 0.45 <= s["aspect"] <= 2.2 and s["val_mean"] > 120:
        paper += 0.22
    if 15 < s["r_mean"] - s["b_mean"] < 45 and s["sat_mean"] < 60:
        paper += 0.18
    if s["rgb_spread"] < 50 and 100 < s["val_mean"] < 230:
        paper += 0.12

    total = metal + plastic + glass + paper + other
    scores[idx["금속"]] = metal / total
    scores[idx["플라스틱"]] = plastic / total
    scores[idx["유리"]] = glass / total
    scores[idx["종이"]] = paper / total
    scores[idx["기타"]] = other / total
    return scores


def is_can_like(crop_bgr: np.ndarray) -> bool:
    if crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
        return False
    s = _rgb_stats(crop_bgr)
    return (
        s["sat_mean"] < 65
        and s["specular"] > 0.025
        and 0.8 <= s["aspect"] <= 4.2
        and s["rgb_spread"] < 45
        and s["val_std"] < 52
    )


def is_plastic_like(crop_bgr: np.ndarray) -> bool:
    s = _rgb_stats(crop_bgr)
    return s["sat_mean"] > 72 or (
        s["sat_mean"] > 52 and (s["b_mean"] > s["r_mean"] * 1.1 or s["g_mean"] > s["r_mean"] * 1.05)
    )


def is_glass_like(crop_bgr: np.ndarray) -> bool:
    s = _rgb_stats(crop_bgr)
    return (s["sat_mean"] < 52 and s["val_std"] > 36) or (
        s["specular"] > 0.05 and s["sat_mean"] < 42 and s["val_std"] > 28
    )


def is_paper_like(crop_bgr: np.ndarray) -> bool:
    s = _rgb_stats(crop_bgr)
    return (
        s["sat_mean"] < 58
        and s["val_mean"] > 125
        and 0.4 <= s["aspect"] <= 2.5
        and s["specular"] < 0.12
    )


def object_prior_vector(object_name: Optional[str]) -> np.ndarray:
    scores = np.zeros(len(MATERIAL_LABELS), dtype=np.float64)
    idx = {n: i for i, n in enumerate(MATERIAL_LABELS)}
    if not object_name:
        scores += 0.2
        return scores / scores.sum()
    prior = OBJECT_MATERIAL_PRIOR.get(object_name, {})
    if not prior:
        scores += 0.2
        return scores / scores.sum()
    for name, w in prior.items():
        if name in idx:
            scores[idx[name]] = w
    s = scores.sum()
    return scores / s if s > 0 else scores


def fuse_material_scores(
    ml_scores: np.ndarray,
    visual_scores: np.ndarray,
    prior_scores: np.ndarray,
    object_name: Optional[str] = None,
    *,
    can_boost: bool = False,
    plastic_boost: bool = False,
    glass_boost: bool = False,
    paper_boost: bool = False,
) -> np.ndarray:
    idx = {n: i for i, n in enumerate(MATERIAL_LABELS)}

    if can_boost:
        w_ml, w_vis, w_prior = 0.10, 0.82, 0.08
    elif paper_boost:
        w_ml, w_vis, w_prior = 0.15, 0.72, 0.13
    elif glass_boost:
        w_ml, w_vis, w_prior = 0.18, 0.68, 0.14
    elif plastic_boost:
        w_ml, w_vis, w_prior = 0.18, 0.68, 0.14
    elif object_name == "bottle":
        w_ml, w_vis, w_prior = 0.22, 0.63, 0.15
    else:
        w_ml, w_vis, w_prior = 0.32, 0.53, 0.15

    fused = w_ml * ml_scores + w_vis * visual_scores + w_prior * prior_scores

    if can_boost:
        fused[idx["금속"]] = max(fused[idx["금속"]], 0.85)
        fused[idx["플라스틱"]] *= 0.12
    if plastic_boost:
        fused[idx["플라스틱"]] = max(fused[idx["플라스틱"]], 0.78)
        fused[idx["금속"]] *= 0.4
    if glass_boost:
        fused[idx["유리"]] = max(fused[idx["유리"]], 0.78)
    if paper_boost:
        fused[idx["종이"]] = max(fused[idx["종이"]], 0.80)
        fused[idx["플라스틱"]] *= 0.5

    fused = np.clip(fused, 1e-6, None)
    return fused / fused.sum()


def enforce_material_rules(
    crop_bgr: np.ndarray,
    dist: np.ndarray,
    object_name: Optional[str] = None,
) -> np.ndarray:
    """재질별 강제 보정 (우선순위: 객체 prior > 유리 > 캔 > 종이 > 플라스틱)."""
    idx = {n: i for i, n in enumerate(MATERIAL_LABELS)}
    out = np.zeros(len(MATERIAL_LABELS), dtype=np.float64)

    if object_name == "book":
        out[idx["종이"]] = 0.9
        out[idx["기타"]] = 0.1
        return out
    if object_name in ("wine glass", "cup", "bowl") and not is_plastic_like(crop_bgr):
        out[idx["유리"]] = 0.82
        out[idx["기타"]] = 0.18
        return out

    if is_glass_like(crop_bgr) and not is_plastic_like(crop_bgr):
        out[idx["유리"]] = 0.86
        out[idx["기타"]] = 0.14
        return out
    if is_can_like(crop_bgr):
        out[idx["금속"]] = 0.9
        out[idx["기타"]] = 0.1
        return out
    if is_paper_like(crop_bgr):
        out[idx["종이"]] = 0.86
        out[idx["기타"]] = 0.14
        return out
    if is_plastic_like(crop_bgr) and float(dist[idx["금속"]]) < 0.35:
        out[idx["플라스틱"]] = 0.84
        out[idx["기타"]] = 0.16
        return out

    return dist
