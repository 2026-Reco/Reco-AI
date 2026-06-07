"""TrashNet 학습 가중치 기반 분류 (있을 때)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from app.services.waste_taxonomy import (
    MATERIAL_LABELS,
    TRASHNET_CLASS_MAP,
    material_index,
)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "trashnet_resnet18.pth"
TRASHNET_CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")

_MODEL = None
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def is_available() -> bool:
    return MODEL_PATH.is_file()


def _load(device: str) -> None:
    global _MODEL
    if _MODEL is not None:
        return
    if not is_available():
        return

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(TRASHNET_CLASSES))
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    _MODEL = model


def predict_trashnet(
    crop_bgr: np.ndarray,
    device: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, str, str, float]]:
    if not is_available():
        return None

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _load(dev)
    if _MODEL is None:
        return None

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = _TRANSFORM(pil).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = _MODEL(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    best_i = int(np.argmax(probs))
    class_name = TRASHNET_CLASSES[best_i]
    material, name_ko, type_id = TRASHNET_CLASS_MAP[class_name]
    confidence = float(probs[best_i])

    idx = material_index()
    dist = np.zeros(len(MATERIAL_LABELS), dtype=np.float64)
    for i, cname in enumerate(TRASHNET_CLASSES):
        mat, _, _ = TRASHNET_CLASS_MAP[cname]
        dist[idx[mat]] += probs[i]
    dist /= dist.sum() + 1e-9

    return dist, type_id, name_ko, confidence
