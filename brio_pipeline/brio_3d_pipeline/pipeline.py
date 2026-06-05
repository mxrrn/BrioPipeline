#!/usr/bin/env python3
"""
BRIO 3D Pipeline — main orchestration script.

Optimised run settings
──────────────────────
* Multi-elevation composite: Images90 (all 8, top-down) + Images45 (every 4th = 6)
  → 14 images with much better XY and Z separation than 24 from one ring
* Fixed global scale: all samples in the same run share the same crop window size
  (the largest construction sets the scale; smaller ones get white padding)
* Agglomerative clustering with geometry + colour features for instance grouping
* DBSCAN cleanup removes background noise per cluster

Usage
─────
    python pipeline.py --samples 113 114 115 116 117
    python pipeline.py --samples 113 --device cpu
"""
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path("/mnt/c/BA/07-dust3r")))
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from logger        import setup_logging
from puml_parser   import find_puml, parse_puml
from preprocessor  import compute_global_halfsize, crop_images
from dust3r_runner import run_dust3r
from sam_runner    import run_sam_on_images
from backprojector          import compute_proposals
from classifier             import assign_classes
from component_classifier   import ComponentClassifier


# ── Run directory ────────────────────────────────────────────────────────────

def _make_run_dir() -> Path:
    """Create and return a new timestamped run folder inside OUTPUTS_ROOT.

    Folder format: run_NNN_YYYYMMDD_HHMM
    The run number (NNN) is one more than the number of existing run_* dirs.
    """
    cfg.OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    existing = [d for d in cfg.OUTPUTS_ROOT.iterdir()
                if d.is_dir() and d.name.startswith("run_")]
    run_num  = len(existing) + 1
    stamp    = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir  = cfg.OUTPUTS_ROOT / f"run_{run_num:03d}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ── Image selection ───────────────────────────────────────────────────────────

def collect_images(sample_id: int) -> list[Path]:
    """
    Sample every 4th image (stride=4) from every available elevation folder:
    Images30, Images45, Images60, Images90.  Missing folders are silently skipped.
    """
    base = cfg.MULTI_VIEW / f"Sample_{sample_id}"
    imgs = []
    for elev in ["Images30", "Images45", "Images60", "Images90"]:
        folder = base / elev
        if not folder.exists():
            continue
        ring = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.JPG"))
        imgs.extend(ring[::4])
    if not imgs:
        raise FileNotFoundError(f"No images found for sample {sample_id}")
    return imgs


# ── Per-sample processing ─────────────────────────────────────────────────────

def _load_classifier(device: str) -> ComponentClassifier | None:
    """Load component visual classifier if weights exist, otherwise return None."""
    if not cfg.CLASSIFIER_WEIGHTS.exists():
        print(f"[Classifier] No weights at {cfg.CLASSIFIER_WEIGHTS} — visual assignment disabled")
        return None
    try:
        clf = ComponentClassifier(cfg.CLASSIFIER_WEIGHTS, device=device)
        print(f"[Classifier] Loaded from {cfg.CLASSIFIER_WEIGHTS}")
        return clf
    except Exception as exc:
        print(f"[Classifier] Failed to load ({exc}) — visual assignment disabled")
        return None


def process_sample(sample_id: int, fixed_half: int, device: str,
                   run_dir: Path, classifier: ComponentClassifier | None = None) -> dict:
    print(f"\n{'='*60}")
    print(f" Sample {sample_id}")
    print(f"{'='*60}")

    puml_path = find_puml(sample_id)
    manifest  = parse_puml(puml_path)
    print(f"[PUML] {manifest.n_components} components: {manifest.class_list}")

    raw_img_paths = collect_images(sample_id)
    folder_counts: dict[str, int] = {}
    for p in raw_img_paths:
        folder_counts[p.parent.name] = folder_counts.get(p.parent.name, 0) + 1
    count_str = ", ".join(f"{k}: {v}" for k, v in sorted(folder_counts.items()))
    print(f"[Images] {len(raw_img_paths)} images ({count_str})")

    out_dir  = run_dir / f"sample_{sample_id}"
    crop_dir = out_dir / "cropped"

    # ── Fixed-scale crop ─────────────────────────────────────────────────
    img_paths = crop_images(raw_img_paths, crop_dir, fixed_half=fixed_half)

    # Save ordered image list so the visualizer uses the same sequence as
    # SAM and DUSt3R (which index into img_paths by position)
    (out_dir / "image_order.json").write_text(
        json.dumps([str(p) for p in img_paths], indent=2)
    )

    # ── DUSt3R ──────────────────────────────────────────────────────────
    dust3r_result = run_dust3r(
        img_paths, out_dir / "dust3r", device=device,
        size=cfg.DUST3R_SIZE, niter=cfg.DUST3R_NITER, batch_size=cfg.DUST3R_BATCH,
    )

    # ── SAM ─────────────────────────────────────────────────────────────
    sam_masks = run_sam_on_images(
        img_paths, manifest.n_components, out_dir / "sam",
        weights_path=cfg.SAM_WEIGHTS, model_type=cfg.SAM_MODEL_TYPE, device=device,
        points_per_side=cfg.SAM_POINTS_SIDE, iou_thresh=cfg.SAM_IOU_THRESH,
        stability_thresh=cfg.SAM_STABILITY, min_area=cfg.SAM_MIN_AREA,
    )

    # ── Back-projection + clustering ─────────────────────────────────────
    proposals, visual_cls, cluster_colours = compute_proposals(
        dust3r_result, sam_masks, manifest.n_components,
        out_dir / "proposals",
        image_paths=img_paths,
        classifier=classifier,
        conf_thresh=cfg.CLASSIFIER_CONF_THRESH,
    )

    # ── Classification ───────────────────────────────────────────────────
    results = assign_classes(proposals, manifest.components,
                             visual_cls=visual_cls, cluster_colours=cluster_colours)

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n[Results] Sample {sample_id}")
    header = f"  {'Instance':<20} {'Points':>8}  {'Centroid':>32}  {'BBox':>24}"
    print(header)
    for r in results:
        c, b = r.centroid, r.bbox_size
        print(f"  {r.instance_id:<20} {r.n_points:>8d}  "
              f"({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})  "
              f"({b[0]:.3f},{b[1]:.3f},{b[2]:.3f})")

    summary = {
        "sample_id":    sample_id,
        "n_components": manifest.n_components,
        "crop_half":    fixed_half,
        "instances": [
            {"instance_id": r.instance_id, "cls": r.cls,
             "n_points": r.n_points,
             "centroid": r.centroid.tolist(),
             "bbox_size": r.bbox_size.tolist()}
            for r in results
        ],
    }
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Results] Saved → {out_path}")
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BRIO 3D pipeline")
    parser.add_argument("--samples", nargs="+", type=int, default=[113],
                        help="Sample IDs to process")
    parser.add_argument("--device",  default=cfg.DEVICE)
    args = parser.parse_args()
    label = "samples_" + "_".join(str(s) for s in args.samples)
    setup_logging(label)

    # ── Create timestamped run directory ──────────────────────────────────
    run_dir = _make_run_dir()
    print(f"[Run] Output directory: {run_dir}")

    # ── Load visual classifier (optional) ────────────────────────────────
    classifier = _load_classifier(args.device)

    # ── Pre-pass: compute global fixed crop size ──────────────────────────
    print("[Pre-pass] Computing global crop scale across all samples...")
    all_raw = [collect_images(sid) for sid in args.samples]
    fixed_half = compute_global_halfsize(all_raw, padding=cfg.AUTO_CROP_PADDING)

    # ── Main pass: process each sample ───────────────────────────────────
    for sid, raw_paths in zip(args.samples, all_raw):
        process_sample(sid, fixed_half=fixed_half, device=args.device,
                       run_dir=run_dir, classifier=classifier)

    print(f"\nAll samples processed → {run_dir}")


if __name__ == "__main__":
    main()
