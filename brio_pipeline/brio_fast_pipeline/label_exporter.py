"""
Convert brio_3d_pipeline outputs into YOLO-format training labels.

For each completed sample the slow pipeline produced:
  - outputs/sample_N/sam/sam_masks_topK.pkl   — per-image SAM masks
  - outputs/sample_N/dust3r/dust3r_cache.pkl  — DUSt3R pts3d (3D per pixel)
  - outputs/sample_N/results.json             — classifier output (instance centroids)
  - outputs/sample_N/cropped/*.jpg            — cropped source images

Label assignment strategy
─────────────────────────
Each SAM mask is back-projected to 3D using the DUSt3R pts3d field.
Its median 3D centroid is compared to every instance centroid in results.json.
The nearest instance determines the class label for that mask.
The 2D bounding box is computed directly from the boolean mask pixels.

Output layout (per image)
─────────────────────────
  dataset/images/{train|val}/sample_{N}_img_{i}.jpg
  dataset/labels/{train|val}/sample_{N}_img_{i}.txt
      one line per mask: <class_idx> <xc> <yc> <w> <h>   (all normalised 0–1)
"""
import sys
import json
import pickle
import shutil
import random
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from logger import setup_logging

sys.path.insert(0, str(cfg.SLOW_PIPELINE))


# ── Helpers ───────────────────────────────────────────────────────────────

def _mask_bbox_normalised(seg: np.ndarray) -> tuple[float, float, float, float] | None:
    """Convert boolean mask to (xc, yc, w, h) normalised to [0,1]."""
    rows = np.where(seg.any(axis=1))[0]
    cols = np.where(seg.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    H, W = seg.shape
    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    if (x1 - x0) * (y1 - y0) < cfg.MIN_MASK_AREA_PX:
        return None
    xc = (x0 + x1) / 2 / W
    yc = (y0 + y1) / 2 / H
    w  = (x1 - x0) / W
    h  = (y1 - y0) / H
    return xc, yc, w, h


def _backproject_centroid(pts3d_frame: np.ndarray, seg: np.ndarray) -> np.ndarray | None:
    """Median 3D centroid of mask pixels in DUSt3R world space."""
    if seg.shape != pts3d_frame.shape[:2]:
        seg = cv2.resize(
            seg.astype(np.uint8),
            (pts3d_frame.shape[1], pts3d_frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    pts = pts3d_frame[seg]
    valid = np.linalg.norm(pts, axis=1) > 1e-6
    pts = pts[valid]
    if len(pts) < 10:
        return None
    return np.median(pts, axis=0).astype(np.float32)


def _nearest_instance_class(centroid_3d: np.ndarray,
                             instances: list[dict]) -> str | None:
    """Return the class of the instance whose 3D centroid is nearest."""
    best_cls, best_dist = None, np.inf
    for inst in instances:
        c = np.array(inst["centroid"], np.float32)
        d = float(np.linalg.norm(centroid_3d - c))
        if d < best_dist:
            best_dist = d
            best_cls  = inst["cls"]
    return best_cls


# ── Per-sample export ─────────────────────────────────────────────────────

def export_sample(sample_id: int, split: str) -> int:
    """
    Export YOLO labels for one sample.  Returns number of images exported.

    Args:
        sample_id : sample number (e.g. 113)
        split     : "train" or "val"
    """
    out_dir = cfg.SLOW_OUTPUTS / f"sample_{sample_id}"

    results_path  = out_dir / "results.json"
    dust3r_path   = out_dir / "dust3r" / "dust3r_cache.pkl"
    cropped_dir   = out_dir / "cropped"

    if not results_path.exists():
        print(f"[Export] Sample {sample_id}: no results.json — skipping")
        return 0

    # Find SAM cache (top-N where N = component count)
    sam_dir   = out_dir / "sam"
    sam_files = list(sam_dir.glob("sam_masks_top*.pkl"))
    if not sam_files:
        print(f"[Export] Sample {sample_id}: no SAM cache — skipping")
        return 0
    sam_path = sam_files[0]

    with open(results_path)  as f:  instances = json.load(f)["instances"]
    with open(dust3r_path, "rb") as f: dust3r   = pickle.load(f)
    with open(sam_path,    "rb") as f: sam_masks = pickle.load(f)

    pts3d_all = dust3r["pts3d"]      # list of (H, W, 3)
    img_paths = sorted(cropped_dir.glob("*.jpg")) + \
                sorted(cropped_dir.glob("*.JPG"))

    img_out = cfg.DATASET_DIR / "images" / split
    lbl_out = cfg.DATASET_DIR / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    exported = 0
    for img_idx, (masks, pts3d_frame) in enumerate(zip(sam_masks, pts3d_all)):
        src_img = img_paths[img_idx] if img_idx < len(img_paths) else None
        stem    = f"sample_{sample_id}_img_{img_idx}"
        lbl_lines = []

        for mask_dict in masks:
            seg = mask_dict["segmentation"]   # (H, W) bool

            # 3D centroid → nearest instance class
            c3d = _backproject_centroid(pts3d_frame, seg)
            if c3d is None:
                continue
            cls = _nearest_instance_class(c3d, instances)
            if cls not in cfg.CLASS_TO_IDX:
                continue
            class_idx = cfg.CLASS_TO_IDX[cls]

            # 2D bounding box (from the mask's native resolution)
            bbox = _mask_bbox_normalised(seg)
            if bbox is None:
                continue
            xc, yc, w, h = bbox
            lbl_lines.append(f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not lbl_lines:
            continue

        # Copy image
        if src_img and src_img.exists():
            shutil.copy2(src_img, img_out / f"{stem}.jpg")
        else:
            continue

        # Write label file
        (lbl_out / f"{stem}.txt").write_text("\n".join(lbl_lines))
        exported += 1

    print(f"[Export] Sample {sample_id} → {exported} images into '{split}'")
    return exported


# ── Dataset-level export ──────────────────────────────────────────────────

def export_dataset(sample_ids: list[int] | None = None,
                   train_val_split: float = cfg.TRAIN_VAL_SPLIT) -> None:
    """
    Export all completed samples from the slow pipeline as a YOLO dataset.

    Args:
        sample_ids      : list of sample IDs to include; None = all in outputs/
        train_val_split : fraction of samples used for training
    """
    if sample_ids is None:
        sample_ids = sorted(
            int(d.name.split("_")[1])
            for d in cfg.SLOW_OUTPUTS.iterdir()
            if d.is_dir() and d.name.startswith("sample_")
               and (d / "results.json").exists()
        )

    if not sample_ids:
        print("[Export] No completed samples found in slow pipeline outputs.")
        return

    random.shuffle(sample_ids)
    n_train = max(1, int(len(sample_ids) * train_val_split))
    train_ids = sample_ids[:n_train]
    val_ids   = sample_ids[n_train:]

    print(f"[Export] {len(sample_ids)} samples → "
          f"{len(train_ids)} train / {len(val_ids)} val")

    total = 0
    for sid in train_ids:
        total += export_sample(sid, "train")
    for sid in val_ids:
        total += export_sample(sid, "val")

    # Write dataset.yaml
    yaml_path = cfg.DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {cfg.DATASET_DIR}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: {cfg.NUM_CLASSES}\n"
        f"names: {cfg.CLASSES}\n"
    )
    print(f"[Export] dataset.yaml written to {yaml_path}")
    print(f"[Export] Total images exported: {total}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="*", type=int,
                        help="Sample IDs to export (default: all completed)")
    args = parser.parse_args()
    setup_logging("label_export")
    export_dataset(sample_ids=args.samples)
