#!/usr/bin/env python3
"""
TrashNet 데이터셋으로 ResNet18 fine-tune.
실행: PYTHONPATH=. python scripts/train_trashnet.py

완료 후 models/trashnet_resnet18.pth 생성 → 서버 재시작 시 자동 적용.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "trashnet"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "trashnet_resnet18.pth"
def download_dataset() -> Path:
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    if DATA_DIR.is_dir() and any(DATA_DIR.iterdir()):
        return DATA_DIR

    zip_path = DATA_DIR.parent / "dataset-resized.zip"
    if not zip_path.is_file():
        print("Downloading TrashNet from Hugging Face (garythung/trashnet) ...")
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id="garythung/trashnet",
            repo_type="dataset",
            filename="dataset-resized.zip",
            local_dir=str(DATA_DIR.parent),
        )
        zip_path = Path(downloaded)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR.parent)
    extracted = DATA_DIR.parent / "dataset-resized"
    if extracted.is_dir() and not DATA_DIR.is_dir():
        extracted.rename(DATA_DIR)
    return DATA_DIR


def train(epochs: int = 12, batch_size: int = 32) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = download_dataset()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    full = datasets.ImageFolder(str(data_path), transform=transform)
    n_val = max(1, len(full) // 5)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(full.classes))
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        print(f"epoch {epoch+1}/{epochs} loss={train_loss/len(train_loader):.4f} val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  saved -> {MODEL_PATH}")

    print(f"Done. best val_acc={best_acc:.3f} classes={full.classes}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    train()
