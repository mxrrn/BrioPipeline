#!/usr/bin/env python3
"""
BRIO Annotator — redesigned annotation pipeline.

Architecture (replaces DUSt3R + SAM-automatic):
  1. View selection   — score all multi-view images; pick the clearest view
  2. SAM prompted     — K-means → N bbox prompts → SamPredictor (one mask per component)
  3. DINOv2 matching  — batch-embed mask-isolated crops; cosine sim to reference library
  4. Hungarian assign — log-prob visual cost + relative-size prior → class/id per mask
  5. Amodal bboxes    — YOLO labels on the primary view
  6. Propagation      — ORB homography → YOLO labels for every other multi-view image

Output layout (under OUTPUTS_ROOT/run_NNN_YYYYMMDD_HHMM/<sample_N>/):
  images/   — copies of annotated images
  labels/   — YOLO .txt label files (same stem as images)
  annotation_summary.json

Usage:
  python annotator.py --sample 113
  python annotator.py --samples 113 114 115 --device cuda
  python annotator.py --samples 113 --device cpu
"""
import sys
import json
import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from scipy.optimize import linear_sum_assignment

# ── Path setup: brio_annotator first, then brio_3d_pipeline ──────────────────
_HERE    = Path(__file__).parent
_3D_DIR  = _HERE.parent / "brio_3d_pipeline"
for _p in [str(_HERE), str(_3D_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ann_config as cfg
from view_selector       import collect_candidate_images, select_best_views
from sam_prompted        import load_predictor, segment_image, foreground_mask, grey_fill
from dino_matcher        import DINOMatcher, mask_isolated_crop
from amodal_bbox         import build_annotations, Annotation
from homography_propagator import propagate_to_image
from yolo_exporter       import export_labels, export_dataset_yaml

from puml_parser import find_puml, parse_puml


# ── Size prior for Hungarian cost ─────────────────────────────────────────────
# Full explicit table so blwo/plwo classes don't collapse to max-digit ambiguity.
_NOMINAL: dict[str, float] = {
    "rolo": 8.0, "rome": 6.0, "rosm": 4.0,
    "sclo": 4.0, "scme": 3.0, "scsm": 2.0,
    "nu": 1.2,  "wa": 0.8,   "pl": 1.0,  "sl": 2.0,
    "bo": 1.5,  "no": 2.0,   "ti": 3.0,
    "whre": 2.0, "whwh": 2.0,
    "blwo11": 1.0, "blwo21": 2.0,
    "plwo21": 2.0, "plwo31": 3.0, "plwo33": 3.0,
    "plwo41": 4.0, "plwo42": 4.0, "plwo43": 4.0,
    "plwo53": 5.0, "plpl53": 5.0, "stpl5": 5.0,
    "stwo3": 3.0, "stwo4": 4.0, "stwo5": 5.0,
    "stwo6": 6.0, "stwo7": 7.0,  "stwo9": 9.0,
}

def _nominal(cls: str) -> float:
    if cls in _NOMINAL:
        return _NOMINAL[cls]
    import re
    digits = [int(d) for d in re.findall(r"\d", cls)]
    return float(max(digits)) if digits else 1.0


# ── Hungarian assignment ──────────────────────────────────────────────────────

def assign_classes(masks             : list[np.ndarray],
                   dino_preds        : list[list[tuple[str, float]]],
                   manifest_components,
                   size_weight       : float = 0.8,
                   ) -> list[tuple[str, str]]:
    """
    Assign N masks to N PUML manifest components via Hungarian matching.

    Cost per (mask_i, component_j) cell:
      visual_cost = −log(cosine_similarity + ε)   lower sim → higher cost
      size_cost   = |rel_observed − rel_nominal|   normalised within the sample
      total       = visual_cost + size_weight × size_cost

    The log-prob visual cost makes the assignment soft: a low-confidence DINOv2
    prediction defers to the size term rather than forcing a hard mismatch penalty.
    """
    n = len(masks)
    m = len(manifest_components)
    assert n == m, f"mask count {n} ≠ manifest count {m}"

    # Build sim lookup: mask_idx → {cls: similarity}
    sim_map: list[dict[str, float]] = [dict(preds) for preds in dino_preds]

    # Scale-free size feature: sqrt(pixel count) as proxy for physical extent
    obs_ext = np.array([max(float(msk.sum()) ** 0.5, 1.0) for msk in masks],
                        np.float32)
    rel_obs = obs_ext / max(obs_ext.max(), 1e-6)
    nom     = np.array([_nominal(c.cls) for c in manifest_components], np.float32)
    rel_nom = nom / max(nom.max(), 1e-6)

    cost = np.zeros((n, m), np.float32)
    for pi in range(n):
        for mi, comp in enumerate(manifest_components):
            sim         = max(sim_map[pi].get(comp.cls, 1e-6), 1e-6)
            visual_cost = float(-np.log(sim))
            size_cost   = abs(float(rel_obs[pi]) - float(rel_nom[mi]))
            cost[pi, mi] = visual_cost + size_weight * size_cost

    row_ind, col_ind = linear_sum_assignment(cost)

    result: list[tuple[str, str]] = [("", "")] * n
    for pi, mi in zip(row_ind, col_ind):
        comp = manifest_components[mi]
        sim  = sim_map[pi].get(comp.cls, 0.0)
        print(f"  [Assign] mask {pi} → {comp.instance_id:<16} "
              f"sim={sim:.3f}  cost={cost[pi, mi]:.3f}")
        result[pi] = (comp.cls, comp.instance_id)
    return result


# ── Per-sample processing ─────────────────────────────────────────────────────

def annotate_sample(sample_id : int,
                    run_dir   : Path,
                    device    : str,
                    predictor ,
                    matcher   : DINOMatcher,
                    ) -> dict:
    print(f"\n{'='*60}")
    print(f" Sample {sample_id}")
    print(f"{'='*60}")

    # 1. PUML manifest
    manifest = parse_puml(find_puml(sample_id))
    n        = manifest.n_components
    print(f"[PUML] {n} components: {manifest.class_list}")

    # 2. View selection
    all_images  = collect_candidate_images(sample_id, cfg.MULTI_VIEW,
                                            cfg.VIEW_ELEVATIONS, cfg.VIEW_STRIDE)
    if not all_images:
        raise FileNotFoundError(f"No images for sample {sample_id}")

    best_views   = select_best_views(all_images, top_k=cfg.TOP_K_VIEWS)
    primary_path = best_views[0]
    print(f"[Views] {len(all_images)} candidates  "
          f"primary: {primary_path.parent.name}/{primary_path.name}")

    # 3. Load and preprocess primary image
    primary_bgr = cv2.imread(str(primary_path))

    # 4. K-means proposals → SamPredictor → N masks
    masks, boxes = segment_image(primary_bgr, n, predictor,
                                  min_area=cfg.SAM_MIN_MASK_AREA)
    print(f"[SAM] {len(masks)} masks from {len(boxes)} K-means proposals")

    # Pad with empty masks if SAM returned fewer than N (sparse foreground)
    H_p, W_p = primary_bgr.shape[:2]
    while len(masks) < n:
        masks.append(np.zeros((H_p, W_p), bool))
    masks = masks[:n]

    # 5. DINOv2 batch matching (one forward pass per image encoding)
    fg_filled = grey_fill(primary_bgr, foreground_mask(primary_bgr))
    dino_preds = matcher.predict_batch_from_masks(fg_filled, masks, top_k=5)
    for i, preds in enumerate(dino_preds):
        top = preds[0] if preds else ("—", 0.0)
        print(f"  [DINO] mask {i:>2}  top-1: {top[0]:<14} sim={top[1]:.3f}")

    # 6. Hungarian assignment
    assignments = assign_classes(masks, dino_preds, manifest.components)

    # 7. Build annotations (tight bboxes; amodal merge available post-hoc)
    annotations = build_annotations(masks, assignments, cfg.ROD_CLASSES)
    n_amodal    = sum(1 for a in annotations if a.amodal)
    print(f"[BBox] {len(annotations)} annotations  ({n_amodal} amodal)")

    # 8. YOLO export — primary view
    out_dir    = run_dir / f"sample_{sample_id}"
    imgs_dir   = out_dir / "images"
    labels_dir = out_dir / "labels"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{primary_path.parent.name}_{primary_path.stem}"
    n_written = export_labels(annotations, cfg.CLASS_TO_IDX,
                               W_p, H_p, labels_dir / f"{stem}.txt")
    shutil.copy(primary_path, imgs_dir / f"{stem}.jpg")
    print(f"[YOLO] Primary label: {n_written} boxes → {labels_dir / f'{stem}.txt'}")

    # Save SAM masks for visualizer (primary view only — propagated views have bboxes only)
    np.savez_compressed(
        out_dir / "annotation_masks.npz",
        **{ann.instance_id: ann.mask.astype(np.uint8) for ann in annotations},
    )
    # Record which image stem is the primary view for the visualizer
    (out_dir / "primary_stem.txt").write_text(stem)

    # 9. Homography propagation to all other multi-view images
    n_targets, n_labels_total = 0, 0
    for target_path in all_images:
        if target_path == primary_path:
            continue
        target_bgr = cv2.imread(str(target_path))
        if target_bgr is None:
            continue

        warped = propagate_to_image(
            primary_bgr, annotations, target_bgr,
            min_matches   = cfg.HOMOG_MIN_MATCHES,
            min_fg_overlap= cfg.HOMOG_MIN_OVERLAP,
        )
        if not warped:
            continue

        valid_anns = [ann for ann, ok in warped if ok]
        if not valid_anns:
            continue

        H_t, W_t = target_bgr.shape[:2]
        t_stem    = f"{target_path.parent.name}_{target_path.stem}"
        n_w = export_labels(valid_anns, cfg.CLASS_TO_IDX,
                             W_t, H_t, labels_dir / f"{t_stem}.txt")
        shutil.copy(target_path, imgs_dir / f"{t_stem}.jpg")
        n_targets      += 1
        n_labels_total += n_w

    print(f"[Propagate] {n_targets} target views labelled  "
          f"({n_labels_total} total box instances)")

    summary = {
        "sample_id"              : sample_id,
        "n_components"           : n,
        "primary_view"           : str(primary_path),
        "n_target_views_labelled": n_targets,
        "annotations"            : [
            {"cls"        : a.cls,
             "instance_id": a.instance_id,
             "bbox_xyxy"  : list(a.bbox_xyxy),
             "amodal"     : a.amodal}
            for a in annotations
        ],
    }
    (out_dir / "annotation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ── Run directory ─────────────────────────────────────────────────────────────

def _make_run_dir() -> Path:
    cfg.OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    existing = [d for d in cfg.OUTPUTS_ROOT.iterdir()
                if d.is_dir() and d.name.startswith("run_")]
    n   = len(existing) + 1
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    d   = cfg.OUTPUTS_ROOT / f"run_{n:03d}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BRIO annotator (redesigned pipeline)")
    parser.add_argument("--samples", nargs="+", type=int, default=[113],
                        help="Sample IDs to annotate")
    parser.add_argument("--device",  default=cfg.DEVICE,
                        help="cuda or cpu")
    args = parser.parse_args()

    run_dir = _make_run_dir()
    print(f"[Run] Output: {run_dir}")

    # Load models once, share across all samples
    print("\n[Models] Loading SAM predictor...")
    predictor = load_predictor(cfg.SAM_WEIGHTS, cfg.SAM_MODEL_TYPE, args.device)

    print("[Models] Loading DINOv2 matcher...")
    dino_cache = cfg.OUTPUTS_ROOT / "dino_reference_library.pkl"
    matcher    = DINOMatcher(
        ref_dir    = cfg.COMPONENT_IMAGES,
        cache_path = dino_cache,
        model_name = cfg.DINO_MODEL,
        device     = args.device,
    )

    # Write dataset.yaml now so it exists even if annotation is partial
    export_dataset_yaml(
        class_names  = cfg.CLASSES,
        train_img_dir= run_dir / "images",
        val_img_dir  = None,
        output_path  = run_dir / "dataset.yaml",
    )

    for sid in args.samples:
        try:
            annotate_sample(sid, run_dir, args.device, predictor, matcher)
        except Exception as exc:
            print(f"[ERROR] Sample {sid} failed: {exc}")
            import traceback; traceback.print_exc()

    print(f"\nAll samples done → {run_dir}")


if __name__ == "__main__":
    main()
