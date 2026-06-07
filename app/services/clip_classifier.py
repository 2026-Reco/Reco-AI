"""CLIP zero-shot 쓰레기 종류 분류."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.services.waste_taxonomy import (
    MATERIAL_LABELS,
    WASTE_CATEGORIES,
    material_index,
)

_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_TEXT_EMBEDS = None
_PROMPT_TO_CATEGORY: List[int] = []


def _load_clip(device: str) -> None:
    global _CLIP_MODEL, _CLIP_PROCESSOR, _TEXT_EMBEDS, _PROMPT_TO_CATEGORY
    if _CLIP_MODEL is not None:
        return

    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"
    _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
    _CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(device)
    _CLIP_MODEL.eval()

    prompts: List[str] = []
    mapping: List[int] = []
    for ci, cat in enumerate(WASTE_CATEGORIES):
        for p in cat.prompts:
            prompts.append(p)
            mapping.append(ci)

    _PROMPT_TO_CATEGORY = mapping
    tokens = _CLIP_PROCESSOR(text=prompts, return_tensors="pt", padding=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        text_features = _CLIP_MODEL.get_text_features(**tokens)
        text_features = F.normalize(text_features, dim=-1)
    _TEXT_EMBEDS = text_features


def predict_clip(
    crop_bgr: np.ndarray,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, str, str, float]:
    """
    Returns:
        material distribution (5,),
        waste_type_id,
        waste_type_ko,
        confidence
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _load_clip(dev)

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = _CLIP_PROCESSOR(images=pil, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        img_feat = _CLIP_MODEL.get_image_features(**inputs)
        img_feat = F.normalize(img_feat, dim=-1)
        sims = (img_feat @ _TEXT_EMBEDS.T)[0]
        probs = F.softmax(sims * 100.0, dim=0).cpu().numpy()

    # 프롬프트별 확률 → 카테고리별 합산
    cat_scores = np.zeros(len(WASTE_CATEGORIES), dtype=np.float64)
    for pi, ci in enumerate(_PROMPT_TO_CATEGORY):
        cat_scores[ci] += probs[pi]
    cat_scores /= cat_scores.sum() + 1e-9

    best_ci = int(np.argmax(cat_scores))
    best_cat = WASTE_CATEGORIES[best_ci]
    confidence = float(cat_scores[best_ci])

    idx = material_index()
    material_dist = np.zeros(len(MATERIAL_LABELS), dtype=np.float64)
    for ci, cat in enumerate(WASTE_CATEGORIES):
        material_dist[idx[cat.material]] += cat_scores[ci]
    material_dist /= material_dist.sum() + 1e-9

    return material_dist, best_cat.type_id, best_cat.name_ko, confidence
