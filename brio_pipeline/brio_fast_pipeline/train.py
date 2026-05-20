"""
Train YOLOv8n on the exported BRIO component detection dataset.

Prerequisites
─────────────
1. Run label_exporter.py first to populate dataset/
2. pip install ultralytics

Usage
─────
    python train.py
    python train.py --epochs 150 --batch 8 --device cpu
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from logger import setup_logging


def train(epochs: int = cfg.YOLO_EPOCHS,
          batch: int = cfg.YOLO_BATCH,
          imgsz: int = cfg.YOLO_IMGSZ,
          device: str = "cuda",
          resume: bool = False) -> Path:
    """
    Fine-tune YOLOv8n from COCO pretrained weights on the BRIO dataset.

    Returns path to the best weights file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")

    dataset_yaml = cfg.DATASET_DIR / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {dataset_yaml}. "
            "Run label_exporter.py first."
        )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_brio"
    model = YOLO(cfg.YOLO_BASE_MODEL)

    model.train(
        data    = str(dataset_yaml),
        epochs  = epochs,
        batch   = batch,
        imgsz   = imgsz,
        device  = device,
        project = str(cfg.RUNS_DIR / "detect"),
        name    = run_name,
        resume  = resume,
        # Augmentation — important for a small dataset
        flipud  = 0.0,    # don't flip vertically (top-down view matters)
        fliplr  = 0.5,
        mosaic  = 0.5,
        degrees = 15.0,   # small rotation (turntable has fixed elevation)
        scale   = 0.3,
        hsv_h   = 0.015,
        hsv_s   = 0.5,
        hsv_v   = 0.3,
    )

    best_weights = cfg.RUNS_DIR / "detect" / run_name / "weights" / "best.pt"
    print(f"\n[Train] Complete. Best weights → {best_weights}")
    return best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on BRIO dataset")
    parser.add_argument("--epochs", type=int, default=cfg.YOLO_EPOCHS)
    parser.add_argument("--batch",  type=int, default=cfg.YOLO_BATCH)
    parser.add_argument("--imgsz",  type=int, default=cfg.YOLO_IMGSZ)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    setup_logging("train")
    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
          device=args.device, resume=args.resume)
