#!/usr/bin/env python3
"""
TinyMed — Master Pipeline Runner
Runs all 5 stages in sequence with checkpointing.

Usage:
    python run_pipeline.py              # Run all stages
    python run_pipeline.py --stage 1    # Run specific stage
    python run_pipeline.py --stage 2 3  # Run multiple stages
    python run_pipeline.py --skip-data  # Skip dataset download prompt
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

STAGES = {
    1: {
        "name": "Train Baseline (ResNet-18)",
        "script": ROOT / "pipeline" / "stage1_train_baseline.py",
        "output_check": ROOT / "models" / "baseline_checkpoint.pt",
    },
    2: {
        "name": "Compression Pipeline (PTQ → QAT → Pruning → Distillation)",
        "script": ROOT / "pipeline" / "stage2_compression_pipeline.py",
        "output_check": ROOT / "models" / "model_student.pt",
    },
    3: {
        "name": "ONNX Export + CoreML + TFLite",
        "script": ROOT / "pipeline" / "stage3_export_onnx.py",
        "output_check": ROOT / "models" / "exported" / "tinymed.onnx",
    },
    4: {
        "name": "Android App (Kotlin — open in Android Studio)",
        "script": None,  # Not a Python script — Android Studio project
        "output_check": ROOT / "android" / "app" / "src" / "main" / "java" / "com" / "tinymed" / "MainActivity.kt",
    },
    5: {
        "name": "Hardware Profiling Dashboard",
        "script": ROOT / "profiler" / "stage5_profiler.py",
        "output_check": ROOT / "logs" / "profiling_report.html",
    },
}


def print_banner():
    print("\n" + "═" * 65)
    print("  TINYMED — On-Device Medical Image Classifier")
    print("  Full Model Compression Pipeline")
    print("═" * 65)
    print("  Stages:")
    for num, s in STAGES.items():
        check = "✓" if (s["output_check"] and s["output_check"].exists()) else "○"
        print(f"  [{check}] Stage {num}: {s['name']}")
    print("═" * 65 + "\n")


def check_dependencies():
    log.info("Checking Python dependencies...")
    missing = []
    required = ["torch", "torchvision", "mlflow", "onnx", "onnxruntime"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(f"Missing packages: {', '.join(missing)}")
        log.error("Run: pip install -r requirements.txt")
        return False
    log.info("Core dependencies OK.")
    return True


def download_dataset_prompt():
    print("\n" + "─" * 65)
    print("DATASET SETUP")
    print("─" * 65)
    print("This pipeline uses the Kaggle Chest X-Ray Images (Pneumonia) dataset.")
    print()
    print("Option 1 — Kaggle CLI (recommended):")
    print("  pip install kaggle")
    print("  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
    print("  unzip chest-xray-pneumonia.zip -d data/")
    print()
    print("Option 2 — Manual download:")
    print("  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("  Extract to: tinymed/data/chest_xray/")
    print()
    print("Expected structure:")
    print("  tinymed/data/chest_xray/train/NORMAL/")
    print("  tinymed/data/chest_xray/train/PNEUMONIA/")
    print()
    print("If dataset is not found, synthetic data will be used (for pipeline demo).")
    print("─" * 65 + "\n")


def run_stage(stage_num: int, dry_run: bool = False) -> bool:
    stage = STAGES[stage_num]
    log.info(f"\n{'═' * 60}")
    log.info(f"STAGE {stage_num}: {stage['name']}")
    log.info(f"{'═' * 60}")

    # Stage 4 is Android Studio — just print instructions
    if stage["script"] is None:
        log.info("Stage 4 is an Android Studio project.")
        log.info(f"  1. Open Android Studio")
        log.info(f"  2. File → Open → {ROOT / 'android'}")
        log.info(f"  3. Copy {ROOT / 'models' / 'exported' / 'tinymed.tflite'}")
        log.info(f"     → {ROOT / 'android' / 'app' / 'src' / 'main' / 'assets' / 'tinymed.tflite'}")
        log.info(f"  4. Build → Run on device")
        log.info("Skipping automated execution for Stage 4.")
        return True

    if not stage["script"].exists():
        log.error(f"Script not found: {stage['script']}")
        return False

    if dry_run:
        log.info(f"[DRY RUN] Would execute: python {stage['script']}")
        return True

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(stage["script"])],
        cwd=str(ROOT),
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        log.error(f"Stage {stage_num} failed (exit code {result.returncode})")
        return False

    log.info(f"Stage {stage_num} completed in {elapsed:.1f}s")

    # Verify output
    if stage["output_check"] and not stage["output_check"].exists():
        log.warning(f"Expected output not found: {stage['output_check']}")
        log.warning("Pipeline may have completed with warnings.")

    return True


def main():
    parser = argparse.ArgumentParser(description="TinyMed Pipeline Runner")
    parser.add_argument("--stage", type=int, nargs="+",
                        choices=list(STAGES.keys()),
                        help="Stage(s) to run. Default: all stages.")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip dataset download instructions.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    args = parser.parse_args()

    print_banner()

    if not check_dependencies():
        sys.exit(1)

    if not args.skip_data:
        download_dataset_prompt()

    stages_to_run = args.stage if args.stage else list(STAGES.keys())

    log.info(f"Running stages: {stages_to_run}")

    overall_start = time.time()
    failed = []

    for stage_num in stages_to_run:
        success = run_stage(stage_num, dry_run=args.dry_run)
        if not success:
            failed.append(stage_num)
            log.error(f"Stage {stage_num} failed. Check logs above.")
            # Continue to next stage — some stages can run independently
            continue

    total_time = time.time() - overall_start

    print("\n" + "═" * 65)
    print("PIPELINE SUMMARY")
    print("═" * 65)
    for num, s in STAGES.items():
        if num not in stages_to_run:
            status = "SKIPPED"
        elif num in failed:
            status = "FAILED ✗"
        else:
            status = "DONE   ✓"
        print(f"  Stage {num}: {status}  — {s['name']}")

    print(f"\nTotal time: {total_time:.1f}s")
    print("═" * 65)

    if failed:
        print(f"\n⚠ Failed stages: {failed}")
        sys.exit(1)
    else:
        print("\nAll stages complete.")
        print(f"\nKey outputs:")
        print(f"  ONNX model  : {ROOT / 'models' / 'exported' / 'tinymed.onnx'}")
        print(f"  TFLite      : {ROOT / 'models' / 'exported' / 'tinymed.tflite'}")
        print(f"  CoreML      : {ROOT / 'models' / 'exported' / 'TinyMed.mlpackage'}")
        print(f"  HTML report : {ROOT / 'logs' / 'profiling_report.html'}")
        print(f"  MLflow UI   : mlflow ui --backend-store-uri {ROOT / 'mlruns'}")
        print(f"  TensorBoard : tensorboard --logdir {ROOT / 'logs' / 'profiler_traces'}")


if __name__ == "__main__":
    main()
