"""
TinyMed — Stage 1: Train Baseline Medical Image Classifier
Dataset: NIH ChestX-ray14 (or Kaggle pneumonia as fallback)
Model: ResNet-18 baseline (~45MB)
Logs everything to MLflow.
"""

import os
import time
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import mlflow
import mlflow.pytorch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
log = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ─── Config ──────────────────────────────────────────────────────────────────
CFG = {
    "num_classes": 2,          # pneumonia vs normal (Kaggle subset)
    "img_size": 224,
    "batch_size": 32,
    "epochs": 15,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "val_split": 0.2,
    "seed": 42,
    "num_workers": 4,
}

# ─── Dataset helpers ─────────────────────────────────────────────────────────
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((CFG["img_size"] + 32, CFG["img_size"] + 32)),
            transforms.RandomCrop(CFG["img_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((CFG["img_size"], CFG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_dataloaders(data_root: Path):
    """
    Expects data_root with subfolders: train/NORMAL, train/PNEUMONIA
    (Kaggle chest X-ray dataset structure).
    Falls back to synthetic data if dataset not found.
    """
    train_path = data_root / "chest_xray" / "train"
    if train_path.exists():
        log.info(f"Loading real dataset from {train_path}")
        full_dataset = ImageFolder(str(train_path), transform=get_transforms(train=True))
        val_size = int(len(full_dataset) * CFG["val_split"])
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(CFG["seed"])
        )
        val_ds.dataset.transform = get_transforms(train=False)
    else:
        log.warning("Dataset not found — using synthetic data for pipeline demo.")
        log.warning("Download real data: kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
        train_ds, val_ds = _synthetic_datasets()

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True
    )
    log.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    return train_loader, val_loader


class SyntheticXrayDataset(Dataset):
    """Synthetic dataset for CI / demo runs when real data is absent."""
    def __init__(self, size=1000, img_size=224, num_classes=2, transform=None):
        self.size = size
        self.img_size = img_size
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.randn(3, self.img_size, self.img_size)
        label = idx % self.num_classes
        return img, label


def _synthetic_datasets():
    train_ds = SyntheticXrayDataset(size=800, transform=get_transforms(True))
    val_ds = SyntheticXrayDataset(size=200, transform=get_transforms(False))
    return train_ds, val_ds


# ─── Model ───────────────────────────────────────────────────────────────────
def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


# ─── Training loop ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 20 == 0:
            log.info(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                     f"loss={loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def measure_inference_latency(model, device, img_size=224, n_runs=100) -> float:
    """Measure average inference latency in ms."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    # Warm-up
    for _ in range(10):
        _ = model(dummy)
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model(dummy)
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    return elapsed


def model_size_mb(model: nn.Module) -> float:
    tmp = MODELS_DIR / "_tmp_size_check.pt"
    torch.save(model.state_dict(), tmp)
    size = tmp.stat().st_size / (1024 ** 2)
    tmp.unlink()
    return size


# ─── Main ────────────────────────────────────────────────────────────────────
def main(args):
    torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(DATA_DIR)
    model = build_model(CFG["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

    mlflow.set_experiment("tinymed-baseline")
    with mlflow.start_run(run_name="resnet18-baseline"):
        mlflow.log_params(CFG)
        mlflow.log_param("device", str(device))
        mlflow.log_param("model", "resnet18")

        best_val_acc = 0.0
        for epoch in range(1, CFG["epochs"] + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            log.info(f"Epoch {epoch:02d} | "
                     f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                     f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODELS_DIR / "baseline_best.pt")
                log.info(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

        # Final metrics
        latency_ms = measure_inference_latency(model, device)
        size_mb = model_size_mb(model)

        mlflow.log_metric("inference_latency_ms", latency_ms)
        mlflow.log_metric("model_size_mb", size_mb)
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.pytorch.log_model(model, "baseline_model")

        log.info("=" * 60)
        log.info(f"BASELINE RESULTS")
        log.info(f"  Best Val Accuracy : {best_val_acc:.4f}")
        log.info(f"  Model Size        : {size_mb:.2f} MB")
        log.info(f"  Inference Latency : {latency_ms:.2f} ms (CPU)")
        log.info("=" * 60)

    # Save full checkpoint for downstream stages
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": CFG,
        "best_val_acc": best_val_acc,
        "size_mb": size_mb,
        "latency_ms": latency_ms,
    }, MODELS_DIR / "baseline_checkpoint.pt")
    log.info(f"Checkpoint saved to {MODELS_DIR / 'baseline_checkpoint.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=CFG["lr"])
    args = parser.parse_args()
    CFG.update({"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr})
    main(args)
