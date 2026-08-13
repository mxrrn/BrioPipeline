#!/usr/bin/env python3
"""
BRIO 3D Pipeline — main orchestration script.

Optimised run settings
──────────────────────
* Multi-elevation composite: every 4th image from Images30/45/60/90 → ~20 images
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
from detection.preprocessor  import compute_global_halfsize, crop_images
from detection.dust3r_runner import run_dust3r
from detection.sam_runner    import run_sam_on_images
from detection.backprojector          import compute_proposals
from classification.classifier             import assign_classes
from classification.component_classifier   import ComponentClassifier


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
    Collect up to IMAGE_VIEWS_PER_RING images from each available elevation ring
    (Images30/45/60/90) using adaptive stride = max(1, len(ring)//target).

    This ensures small rings (e.g. Images90 with only 8 images) contribute fully
    while large rings (24 images) are still sub-sampled for DUSt3R budget.
    Missing folders are silently skipped.
    """
    base   = cfg.MULTI_VIEW / f"Sample_{sample_id}"
    target = getattr(cfg, "IMAGE_VIEWS_PER_RING", 8)
    imgs   = []
    for elev in ["Images30", "Images45", "Images60", "Images90"]:
        folder = base / elev
        if not folder.exists():
            continue
        ring   = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.JPG"))
        stride = max(1, len(ring) // target)
        imgs.extend(ring[::stride])
    if not imgs:
        raise FileNotFoundError(f"No images found for sample {sample_id}")
    return imgs


def _amodal_extend_rods(results, proposals, thjo_pairs, protrusion_m: float) -> None:
    """
    For each through-joint rod (thjo_pairs from the PUML manifest), extend its
    axis-aligned bbox to cover all components it passes through plus a fixed
    protrusion on each end.

    The rod axis is estimated from the rod's own point cloud (PCA principal axis).
    If the rod cloud is too sparse (< 30 pts), the through-component's minor PCA
    axis is used as a fallback (the rod goes through the thinnest dimension of the
    block).  All changes are in-place on results[i].bbox_size / centroid.
    """
    import numpy as np
    id_to_idx = {r.instance_id: i for i, r in enumerate(results)}

    for rod_id, through_ids in thjo_pairs:
        if rod_id not in id_to_idx:
            continue
        rod_idx  = id_to_idx[rod_id]
        rod_pts  = proposals[rod_idx]

        # Gather all 3D points: rod + every component it threads through
        part_clouds = [rod_pts] if len(rod_pts) >= 10 else []
        for tid in through_ids:
            if tid in id_to_idx:
                tp = proposals[id_to_idx[tid]]
                if len(tp) >= 10:
                    part_clouds.append(tp)
        if not part_clouds:
            continue

        all_pts = np.concatenate(part_clouds, axis=0)

        # Determine rod axis: prefer rod cloud PCA, fall back to through-comp minor axis
        def _pca_axes(pts):
            _, _, Vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
            return Vt   # rows are principal axes, descending variance

        if len(rod_pts) >= 30:
            rod_axis = _pca_axes(rod_pts)[0]
        else:
            # Use through-component's minor axis (rod goes through its thin dimension)
            through_pts_list = [proposals[id_to_idx[tid]]
                                for tid in through_ids if tid in id_to_idx
                                and len(proposals[id_to_idx[tid]]) >= 30]
            if through_pts_list:
                through_pts = np.concatenate(through_pts_list)
                rod_axis = _pca_axes(through_pts)[-1]  # minor axis = through direction
            else:
                rod_axis = _pca_axes(all_pts)[0]

        # Project all assembly points onto the rod axis and extend by protrusion
        origin  = all_pts.mean(axis=0)
        proj    = (all_pts - origin) @ rod_axis
        ext_lo  = float(proj.min()) - protrusion_m
        ext_hi  = float(proj.max()) + protrusion_m

        # Synthesise extended rod axis as a dense line, then compute AABB of (assembly + line)
        t          = np.linspace(ext_lo, ext_hi, 50)
        rod_line   = origin + np.outer(t, rod_axis)   # (50, 3)
        combined   = np.vstack([all_pts, rod_line])
        aabb_lo    = combined.min(axis=0)
        aabb_hi    = combined.max(axis=0)

        new_bbox  = (aabb_hi - aabb_lo).astype(np.float32)
        new_cent  = ((aabb_lo + aabb_hi) / 2).astype(np.float32)

        old_bbox  = results[rod_idx].bbox_size
        print(f"[Amodal] {rod_id}: bbox {old_bbox.round(3)} → {new_bbox.round(3)}")
        results[rod_idx].bbox_size = new_bbox
        results[rod_idx].centroid  = new_cent


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
                   run_dir: Path, classifier: ComponentClassifier | None = None,
                   sam_mode: str = "prompted") -> dict:
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
        max_area_frac=cfg.SAM_MAX_AREA_FRAC, fg_overlap_min=cfg.SAM_FG_OVERLAP_MIN,
        dedup_iou=cfg.SAM_DEDUP_IOU, keep_factor=cfg.SAM_KEEP_FACTOR,
        mode=sam_mode, grid_spacing=cfg.SAM_GRID_SPACING,
        dt_seeds=cfg.SAM_DT_SEEDS,
    )

    # ── Back-projection + clustering ─────────────────────────────────────
    proposals, visual_cls, cluster_colours, cluster_elongations = compute_proposals(
        dust3r_result, sam_masks, manifest.n_components,
        out_dir / "proposals",
        image_paths=img_paths,
        classifier=classifier,
        conf_thresh=cfg.CLASSIFIER_CONF_THRESH,
    )

    # ── Classification ───────────────────────────────────────────────────
    results = assign_classes(proposals, manifest.components,
                             visual_cls=visual_cls, cluster_colours=cluster_colours,
                             cluster_elongations=cluster_elongations)

    # ── Amodal bbox extension for through-joint rods ─────────────────────
    if manifest.thjo_pairs:
        _amodal_extend_rods(results, proposals, manifest.thjo_pairs,
                            protrusion_m=cfg.ROD_PROTRUSION_M)

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n[Results] Sample {sample_id}")
    header = f"  {'Instance':<20} {'Points':>8}  {'Centroid':>32}  {'BBox':>24}"
    print(header)
    for r in results:
        c, b = r.centroid, r.bbox_size
        print(f"  {r.instance_id:<20} {r.n_points:>8d}  "
              f"({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})  "
              f"({b[0]:.3f},{b[1]:.3f},{b[2]:.3f})")

    # A zero-point instance means a manifest component ended up with an
    # empty cluster — it exists in the output but is represented nowhere.
    # Surface it loudly instead of leaving it discoverable only by reading
    # the JSON by hand (that's how the stwo3_1/nu_2 regression went
    # unnoticed across several runs).
    warnings = [f"{r.instance_id}: 0 points — not represented in any view"
                for r in results if r.n_points == 0]
    for w in warnings:
        print(f"[WARN] Sample {sample_id}: {w}")

    summary = {
        "sample_id":    sample_id,
        "n_components": manifest.n_components,
        "crop_half":    fixed_half,
        "warnings":     warnings,
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
    parser.add_argument("--device",   default=cfg.DEVICE)
    parser.add_argument("--sam-mode", default="prompted",
                        choices=["prompted", "auto"],
                        help="SAM segmentation mode: 'prompted' (default, tight-crop "
                             "+ foreground-only grid) or 'auto' (original uniform grid)")
    parser.add_argument("--resume",  default=None,
                        help="Resume an existing run dir (e.g. run_015_20260605_1450). "
                             "Reuses cached DUSt3R/SAM outputs and skips completed samples.")
    args = parser.parse_args()
    label = "samples_" + "_".join(str(s) for s in args.samples)
    setup_logging(label)

    # ── Create or resume run directory ────────────────────────────────────
    if args.resume:
        run_dir = cfg.OUTPUTS_ROOT / args.resume
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume dir not found: {run_dir}")
        print(f"[Run] Resuming: {run_dir}")
    else:
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
                       run_dir=run_dir, classifier=classifier,
                       sam_mode=args.sam_mode)

    print(f"\nAll samples processed → {run_dir}")


if __name__ == "__main__":
    main()
