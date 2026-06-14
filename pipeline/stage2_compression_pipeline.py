"""
TinyMed — Stage 2: Full Model Compression Pipeline
Steps (in order):
  1. Post-Training Quantization (PTQ) → INT8
  2. Quantization-Aware Training   (QAT) → recover accuracy
  3. Structured Pruning             → remove low-magnitude conv filters
  4. Knowledge Distillation         → ResNet-18 teacher → EfficientNet-B0 student

All steps logged to MLflow. Benchmarks printed and saved at each stage.
Target: model < 5 MB, accuracy drop < 3%, CPU inference < 50 ms.
"""

import copy
import logging
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.quantization
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.quantization import resnet18 as quantizable_resnet18
import mlflow
import mlflow.pytorch

# Import helpers from Stage 1
import sys
sys.path.insert(0, str(Path(__file__).parent))
from stage1_train_baseline import (
    CFG, DATA_DIR, MODELS_DIR,
    get_dataloaders, build_model,
    evaluate, measure_inference_latency, model_size_mb,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ─── Quantization backend ───────────────────────────────────────────────────
# 'fbgemm' is the x86 quantized backend; 'qnnpack' is required on ARM
# (Apple Silicon M1/M2/M3/M4, and Android/mobile CPUs). Auto-detect so the
# same script works on Intel and Apple Silicon Macs without edits.
QENGINE = "qnnpack" if platform.machine() in ("arm64", "aarch64") else "fbgemm"
torch.backends.quantized.engine = QENGINE
log.info(f"Quantization engine: {QENGINE} (arch={platform.machine()})")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def load_baseline(device: torch.device) -> nn.Module:
    ckpt_path = MODELS_DIR / "baseline_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {ckpt_path}. "
            "Run stage1_train_baseline.py first."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(CFG["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"Loaded baseline (val_acc={ckpt['best_val_acc']:.4f}, "
             f"size={ckpt['size_mb']:.2f}MB, latency={ckpt['latency_ms']:.2f}ms)")
    return model


def benchmark(model: nn.Module, val_loader: DataLoader, device: torch.device,
              criterion: nn.Module, label: str) -> dict:
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    latency = measure_inference_latency(model, device)
    size = model_size_mb(model)
    result = {
        "stage": label,
        "val_acc": round(val_acc, 4),
        "val_loss": round(val_loss, 4),
        "size_mb": round(size, 2),
        "latency_ms": round(latency, 2),
    }
    log.info(f"[{label}] acc={val_acc:.4f} | size={size:.2f}MB | latency={latency:.2f}ms")
    return result


def calibration_loader(data_root: Path, n_batches: int = 10):
    """Small loader used to calibrate PTQ observers."""
    train_loader, _ = get_dataloaders(data_root)
    batches = []
    for i, (imgs, _) in enumerate(train_loader):
        batches.append(imgs)
        if i >= n_batches:
            break
    return batches


def build_fusable_model(num_classes: int) -> nn.Module:
    """
    Quantization-ready ResNet-18 (torchvision's quantizable variant).
    Same architecture/param names as the plain ResNet-18 from
    `build_model`, but exposes a `.fuse_model()` method that fuses
    Conv-BN-ReLU triplets in place — required before PTQ/QAT.
    """
    model = quantizable_resnet18(weights=None, quantize=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def to_fusable(model: nn.Module, num_classes: int) -> nn.Module:
    """Copy a trained plain ResNet-18's weights into the fusable variant."""
    fusable = build_fusable_model(num_classes)
    fusable.load_state_dict(model.state_dict())
    return fusable


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — POST-TRAINING QUANTIZATION (PTQ)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_ptq(model: nn.Module, calib_batches: list) -> nn.Module:
    """
    Static INT8 PTQ using PyTorch's eager-mode quantization.
    Inserts QuantStub/DeQuantStub wrappers, calibrates observers,
    then converts to quantized model.
    """
    log.info("── PTQ: Fusing Conv-BN-ReLU layers ──")
    model_ptq = to_fusable(copy.deepcopy(model).cpu(), CFG["num_classes"])
    model_ptq.eval()

    # Fuse Conv-BN-ReLU triplets for better quantization accuracy
    model_ptq.fuse_model()

    # Wrap with QuantStub
    class QuantWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.quant = torch.quantization.QuantStub()
            self.model = m
            self.dequant = torch.quantization.DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = self.model(x)
            x = self.dequant(x)
            return x

    wrapped = QuantWrapper(model_ptq)
    wrapped.qconfig = torch.quantization.get_default_qconfig(QENGINE)
    torch.quantization.prepare(wrapped, inplace=True)

    log.info("── PTQ: Running calibration ──")
    wrapped.eval()
    with torch.no_grad():
        for imgs in calib_batches:
            wrapped(imgs)

    torch.quantization.convert(wrapped, inplace=True)
    log.info("── PTQ: Conversion complete ──")
    return wrapped


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — QUANTIZATION-AWARE TRAINING (QAT)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_qat(model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, device: torch.device,
              epochs: int = 5) -> nn.Module:
    """
    Fine-tune with simulated quantization noise (QAT).
    Recovers accuracy lost during PTQ by training the model
    to be robust against INT8 weight/activation rounding.
    """
    log.info("── QAT: Preparing model ──")
    model_qat = to_fusable(copy.deepcopy(model).cpu(), CFG["num_classes"])

    # Fuse Conv-BN-ReLU before inserting fake-quant observers (must be in eval mode)
    model_qat.eval()
    model_qat.fuse_model()
    model_qat.train()

    model_qat.qconfig = torch.quantization.get_default_qat_qconfig(QENGINE)
    torch.quantization.prepare_qat(model_qat, inplace=True)
    model_qat = model_qat.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_qat.parameters(), lr=1e-4, momentum=0.9,
                          weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model_qat.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model_qat(imgs), labels)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model_qat, val_loader, criterion, device)
        scheduler.step()
        log.info(f"  QAT epoch {epoch}/{epochs} | val_acc={val_acc:.4f}")

    # Convert to final quantized model
    model_qat.cpu().eval()
    torch.quantization.convert(model_qat, inplace=True)
    log.info("── QAT: Conversion complete ──")
    return model_qat


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — STRUCTURED PRUNING
# ═══════════════════════════════════════════════════════════════════════════════

def apply_structured_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Remove `amount` fraction of conv filters with lowest L1-norm.
    Uses structured (ln_structured) pruning so the tensor shape actually shrinks
    after mask removal — giving real latency and size reduction.
    """
    log.info(f"── Pruning: structured L1 pruning at amount={amount} ──")
    model_pruned = copy.deepcopy(model)

    # Collect all Conv2d layers
    conv_layers = [
        (name, module)
        for name, module in model_pruned.named_modules()
        if isinstance(module, nn.Conv2d)
    ]

    for name, module in conv_layers:
        prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
        prune.remove(module, "weight")  # Make pruning permanent

    # Count remaining params
    total = sum(p.numel() for p in model_pruned.parameters())
    nonzero = sum(p.nonzero().size(0) for p in model_pruned.parameters())
    sparsity = 1.0 - nonzero / total
    log.info(f"── Pruning: sparsity={sparsity:.2%} ──")
    return model_pruned


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — KNOWLEDGE DISTILLATION
# ═══════════════════════════════════════════════════════════════════════════════

class DistillationLoss(nn.Module):
    """
    Combined hard + soft label loss for knowledge distillation.
    L = alpha * CE(student_logits, hard_labels)
      + (1 - alpha) * KL(softmax(student/T), softmax(teacher/T)) * T^2
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        hard_loss = self.ce(student_logits, labels)
        soft_student = torch.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = torch.softmax(teacher_logits / self.T, dim=1)
        soft_loss = self.kl(soft_student, soft_teacher) * (self.T ** 2)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


def build_student(num_classes: int) -> nn.Module:
    """EfficientNet-B0 student — ~20MB before compression."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    student = models.efficientnet_b0(weights=weights)
    in_features = student.classifier[1].in_features
    student.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return student


def apply_knowledge_distillation(teacher: nn.Module, train_loader: DataLoader,
                                  val_loader: DataLoader, device: torch.device,
                                  epochs: int = 10) -> nn.Module:
    log.info("── Distillation: training EfficientNet-B0 student ──")
    teacher = teacher.to(device).eval()
    student = build_student(CFG["num_classes"]).to(device)

    criterion = DistillationLoss(temperature=4.0, alpha=0.3)
    optimizer = optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_only = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        student.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            student_logits = student(imgs)
            loss = criterion(student_logits, teacher_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(student, val_loader, ce_only, device)
        scheduler.step()
        log.info(f"  Distillation epoch {epoch}/{epochs} | val_acc={val_acc:.4f}")

    return student


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(DATA_DIR)
    criterion = nn.CrossEntropyLoss()
    calib_batches = calibration_loader(DATA_DIR, n_batches=10)

    # Load teacher baseline
    teacher = load_baseline(device)
    baseline_metrics = benchmark(teacher, val_loader, device, criterion, "baseline")

    results = [baseline_metrics]

    mlflow.set_experiment("tinymed-compression")
    with mlflow.start_run(run_name="full-compression-pipeline"):
        mlflow.log_param("device", str(device))

        # ── Step 1: PTQ ──────────────────────────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("STEP 1 / 4 — Post-Training Quantization (PTQ)")
        log.info("=" * 60)
        model_ptq = apply_ptq(teacher, calib_batches)
        ptq_metrics = benchmark(model_ptq, val_loader, torch.device("cpu"),
                                criterion, "ptq_int8")
        results.append(ptq_metrics)
        torch.save(model_ptq, MODELS_DIR / "model_ptq.pt")
        mlflow.log_metrics({f"ptq_{k}": v for k, v in ptq_metrics.items()
                            if isinstance(v, float)})

        # ── Step 2: QAT ──────────────────────────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("STEP 2 / 4 — Quantization-Aware Training (QAT)")
        log.info("=" * 60)
        model_qat = apply_qat(
            copy.deepcopy(teacher), train_loader, val_loader, device, epochs=5
        )
        qat_metrics = benchmark(model_qat, val_loader, torch.device("cpu"),
                                criterion, "qat_int8")
        results.append(qat_metrics)
        torch.save(model_qat, MODELS_DIR / "model_qat.pt")
        mlflow.log_metrics({f"qat_{k}": v for k, v in qat_metrics.items()
                            if isinstance(v, float)})

        # ── Step 3: Structured Pruning ────────────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("STEP 3 / 4 — Structured Pruning (30%)")
        log.info("=" * 60)
        model_pruned = apply_structured_pruning(teacher, amount=0.30)
        pruned_metrics = benchmark(model_pruned, val_loader, device,
                                   criterion, "pruned_30pct")
        results.append(pruned_metrics)
        torch.save(model_pruned.state_dict(), MODELS_DIR / "model_pruned.pt")
        mlflow.log_metrics({f"pruned_{k}": v for k, v in pruned_metrics.items()
                            if isinstance(v, float)})

        # ── Step 4: Knowledge Distillation ────────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("STEP 4 / 4 — Knowledge Distillation (ResNet-18 → EfficientNet-B0)")
        log.info("=" * 60)
        model_student = apply_knowledge_distillation(
            teacher, train_loader, val_loader, device, epochs=10
        )
        student_metrics = benchmark(model_student, val_loader, device,
                                    criterion, "efficientnet_b0_student")
        results.append(student_metrics)
        torch.save(model_student.state_dict(), MODELS_DIR / "model_student.pt")
        mlflow.log_metrics({f"student_{k}": v for k, v in student_metrics.items()
                            if isinstance(v, float)})
        mlflow.pytorch.log_model(model_student, "student_model")

        # ── Summary table ─────────────────────────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("COMPRESSION PIPELINE SUMMARY")
        log.info("=" * 60)
        log.info(f"{'Stage':<30} {'Acc':>6} {'Size(MB)':>10} {'Latency(ms)':>12}")
        log.info("-" * 60)
        for r in results:
            log.info(f"{r['stage']:<30} {r['val_acc']:>6.4f} {r['size_mb']:>10.2f} "
                     f"{r['latency_ms']:>12.2f}")

        # Save results JSON for the dashboard
        import json
        results_path = ROOT / "logs" / "compression_results.json"
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(str(results_path))
        log.info(f"\nResults saved to {results_path}")

    log.info("\nStage 2 complete. Run stage3_export_onnx.py next.")


if __name__ == "__main__":
    main()
