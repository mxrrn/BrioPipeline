"""
2D verification overlay for slow-pipeline results.

For each cropped view image, colors every SAM mask by the instance it was
assigned to (matched by nearest 3D centroid) and labels it with the instance_id.
Saves a grid of annotated views as  outputs/sample_N/viz_2d.png

Usage
─────
    python visualize_2d.py --sample 114
    python visualize_2d.py --sample 114 --max-views 6
"""
import sys
import json
import pickle
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


def _resolve_run_dir(run_arg: str | None) -> Path:
    """Return the run directory to visualize.

    If run_arg is given, look it up directly under OUTPUTS_ROOT.
    Otherwise pick the most recently created run_* folder.
    """
    if run_arg:
        d = cfg.OUTPUTS_ROOT / run_arg
        if not d.exists():
            raise FileNotFoundError(f"Run directory not found: {d}")
        return d
    candidates = sorted(
        [d for d in cfg.OUTPUTS_ROOT.iterdir()
         if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found in {cfg.OUTPUTS_ROOT}")
    return candidates[-1]


# ── Colour palette (one per instance, BGR) ───────────────────────────────────

_PALETTE = [
    (  0, 114, 189), (217,  83,  25), ( 237, 177,  32),
    (126,  47, 142), ( 119, 172,  48), ( 77, 190, 238),
    (162,  20,  47), (  0, 168, 107), (255, 127,   0),
    ( 77,  77, 255),
]


def _bgr(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


# ── Centroid from mask via pts3d ─────────────────────────────────────────────

def _mask_centroid_3d(pts3d_frame: np.ndarray, seg: np.ndarray) -> np.ndarray | None:
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
    return np.median(pts, axis=0)


# ── Draw one annotated image ─────────────────────────────────────────────────

def annotate_image(img_bgr: np.ndarray,
                   masks: list[dict],
                   pts3d_frame: np.ndarray,
                   instance_centroids: list[np.ndarray],
                   instance_ids: list[str],
                   alpha: float = 0.45) -> np.ndarray:
    """
    Overlay SAM masks coloured by nearest instance centroid.
    Returns annotated copy of img_bgr.
    """
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.copy().astype(np.float32)

    for mask_dict in masks:
        seg = mask_dict["segmentation"]

        c3d = _mask_centroid_3d(pts3d_frame, seg)
        if c3d is None:
            continue

        # Nearest instance centroid
        dists = [np.linalg.norm(c3d - ic) for ic in instance_centroids]
        best  = int(np.argmin(dists))
        color = _bgr(best)

        # Resize seg to image size if needed
        if seg.shape[:2] != (H, W):
            seg = cv2.resize(
                seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        # Fill mask with colour
        mask_f = np.zeros((H, W, 3), np.float32)
        mask_f[seg] = color
        overlay = np.where(seg[:, :, None], overlay * (1 - alpha) + mask_f * alpha, overlay)

        # Label at mask centroid
        ys, xs = np.where(seg)
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            label  = instance_ids[best]
            fs = 0.45
            th = 1
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            # dark background for readability
            cv2.rectangle(overlay,
                          (cx - tw // 2 - 2, cy - 10),
                          (cx + tw // 2 + 2, cy + 4),
                          (20, 20, 20), -1)
            cv2.putText(overlay, label,
                        (cx - tw // 2, cy + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), th, cv2.LINE_AA)

    return overlay.astype(np.uint8)


# ── Legend strip ─────────────────────────────────────────────────────────────

def make_legend(instance_ids: list[str], width: int, row_h: int = 24) -> np.ndarray:
    n   = len(instance_ids)
    img = np.ones((row_h * n + 8, width, 3), np.uint8) * 30
    for i, iid in enumerate(instance_ids):
        y = i * row_h + 6
        cv2.rectangle(img, (6, y), (20, y + row_h - 6), _bgr(i), -1)
        cv2.putText(img, iid, (28, y + row_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def visualize_2d(sample_id: int, run_dir: Path, max_views: int = 8) -> Path:
    out_dir    = run_dir / f"sample_{sample_id}"
    results_path = out_dir / "results.json"
    dust3r_path  = out_dir / "dust3r" / "dust3r_cache.pkl"
    sam_dir      = out_dir / "sam"
    crop_dir     = out_dir / "cropped"

    if not results_path.exists():
        raise FileNotFoundError(f"No results.json for sample {sample_id}. Run pipeline first.")

    with open(results_path) as f:
        results = json.load(f)
    with open(dust3r_path, "rb") as f:
        dust3r = pickle.load(f)

    sam_files = sorted(sam_dir.glob("sam_masks_top*.pkl"))
    with open(sam_files[0], "rb") as f:
        sam_masks = pickle.load(f)

    pts3d_all = dust3r["pts3d"]          # list of (H, W, 3)

    # Use the pipeline's original image order so that img_paths[i] matches
    # sam_masks[i] and pts3d_all[i] — a simple alphabetical sort is wrong
    # when images come from multiple elevation folders with colliding names.
    order_file = out_dir / "image_order.json"
    if order_file.exists():
        with open(order_file) as f:
            img_paths = [Path(p) for p in json.load(f)]
    else:
        img_paths = sorted(crop_dir.glob("*.jpg")) + sorted(crop_dir.glob("*.JPG"))

    instance_ids      = [inst["instance_id"] for inst in results["instances"]]
    instance_centroids = [np.array(inst["centroid"], np.float32)
                          for inst in results["instances"]]

    # Select views to annotate
    n_views  = min(max_views, len(sam_masks), len(pts3d_all), len(img_paths))
    indices  = np.linspace(0, min(len(sam_masks), len(img_paths)) - 1,
                           n_views, dtype=int)

    annotated = []
    for vi in indices:
        bgr = cv2.imread(str(img_paths[vi]))
        if bgr is None:
            continue
        ann = annotate_image(bgr, sam_masks[vi], pts3d_all[vi],
                             instance_centroids, instance_ids)
        # Add view label
        cv2.putText(ann, f"view {vi}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
        annotated.append(ann)

    if not annotated:
        raise RuntimeError("No annotated views produced.")

    # Resize all to same height
    target_h = 400
    resized  = []
    for a in annotated:
        h, w = a.shape[:2]
        new_w = int(w * target_h / h)
        resized.append(cv2.resize(a, (new_w, target_h)))

    # Arrange in rows of 4
    cols     = min(4, len(resized))
    rows     = []
    for i in range(0, len(resized), cols):
        row_imgs = resized[i:i + cols]
        # Pad last row if needed
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(resized[0]))
        rows.append(np.hstack(row_imgs))

    grid = np.vstack(rows)

    # Attach legend on the right
    legend = make_legend(instance_ids, width=160, row_h=28)
    legend_h = legend.shape[0]
    grid_h   = grid.shape[0]
    if legend_h < grid_h:
        pad = np.ones((grid_h - legend_h, 160, 3), np.uint8) * 30
        legend = np.vstack([legend, pad])
    else:
        pad = np.ones((legend_h - grid_h, grid.shape[1], 3), np.uint8) * 20
        grid = np.vstack([grid, pad])

    final = np.hstack([grid, legend])

    out_path = out_dir / "viz_2d.png"
    cv2.imwrite(str(out_path), final)
    print(f"[Viz2D] Saved → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D instance overlay visualisation")
    parser.add_argument("--sample",    type=int, required=True)
    parser.add_argument("--run",       type=str, default=None,
                        help="Run folder name (e.g. run_001_20260526_2123). "
                             "Defaults to the most recent run_* folder.")
    parser.add_argument("--max-views", type=int, default=8,
                        help="Max number of views to include in the grid (default 8)")
    args = parser.parse_args()
    run_dir = _resolve_run_dir(args.run)
    print(f"[Viz2D] Using run: {run_dir.name}")
    visualize_2d(args.sample, run_dir, args.max_views)
