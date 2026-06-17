"""
2D verification overlay for slow-pipeline results.

For each detected component, shows up to N views where that component's SAM
masks are highlighted in colour.  All other masks are rendered as dim grey so
the construction context is preserved.  One component → one labelled row-strip.

Usage
─────
    python visualize_2d.py --sample 114
    python visualize_2d.py --sample 114 --views-per-component 6
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


# ── Constants ────────────────────────────────────────────────────────────────

VIEW_W   = 320   # fixed width per view tile
VIEW_H   = 240   # fixed height per view tile
HEADER_H = 32    # height of the per-component label bar


# ── Colour palette (BGR) ─────────────────────────────────────────────────────

_PALETTE = [
    (  0, 114, 189), (217,  83,  25), (237, 177,  32),
    (126,  47, 142), (119, 172,  48), ( 77, 190, 238),
    (162,  20,  47), (  0, 168, 107), (255, 127,   0),
    ( 77,  77, 255),
]


def _bgr(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


# ── Run-dir resolver ─────────────────────────────────────────────────────────

def _resolve_run_dir(run_arg: str | None) -> Path:
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


# ── 3-D centroid of a mask via pts3d ────────────────────────────────────────

def _mask_centroid_3d(pts3d_frame: np.ndarray, seg: np.ndarray) -> np.ndarray | None:
    if seg.shape != pts3d_frame.shape[:2]:
        seg = cv2.resize(
            seg.astype(np.uint8),
            (pts3d_frame.shape[1], pts3d_frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    pts   = pts3d_frame[seg]
    valid = np.linalg.norm(pts, axis=1) > 1e-6
    pts   = pts[valid]
    if len(pts) < 10:
        return None
    return np.median(pts, axis=0)


# ── Annotate one view for a single target component ──────────────────────────

def _annotate_for_component(
    img_bgr: np.ndarray,
    masks: list[dict],
    pts3d_frame: np.ndarray,
    instance_centroids: list[np.ndarray],
    target_idx: int,
    instance_id: str,
    color: tuple[int, int, int],
    alpha: float = 0.50,
) -> tuple[np.ndarray, bool]:
    """
    Returns (annotated_image, component_was_found).
    Target component masks: full colour + label.
    Other masks: dim grey overlay for spatial context.
    """
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.copy().astype(np.float32)
    found   = False

    for mask_dict in masks:
        seg = mask_dict["segmentation"]

        c3d = _mask_centroid_3d(pts3d_frame, seg)
        if c3d is None:
            continue

        dists = [np.linalg.norm(c3d - ic) for ic in instance_centroids]
        best  = int(np.argmin(dists))

        if seg.shape[:2] != (H, W):
            seg = cv2.resize(
                seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        mask_f = np.zeros((H, W, 3), np.float32)

        if best == target_idx:
            # Highlight target component
            mask_f[seg] = color
            overlay = np.where(seg[:, :, None],
                               overlay * (1 - alpha) + mask_f * alpha, overlay)
            found = True

            # Label at mask pixel centroid
            ys, xs = np.where(seg)
            if len(xs):
                cx, cy = int(xs.mean()), int(ys.mean())
                fs, th = 0.45, 1
                (tw, _), _ = cv2.getTextSize(
                    instance_id, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                cv2.rectangle(overlay,
                              (cx - tw // 2 - 2, cy - 10),
                              (cx + tw // 2 + 2, cy + 4),
                              (20, 20, 20), -1)
                cv2.putText(overlay, instance_id,
                            (cx - tw // 2, cy + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (255, 255, 255), th, cv2.LINE_AA)
        else:
            # Dim grey for other masks (context only)
            mask_f[seg] = (60, 60, 60)
            overlay = np.where(seg[:, :, None],
                               overlay * 0.82 + mask_f * 0.18, overlay)

    return overlay.astype(np.uint8), found


# ── Build one component strip (header + row of view tiles) ───────────────────

def _make_strip(
    views: list[np.ndarray],
    inst_id: str,
    color_bgr: tuple[int, int, int],
    n_cols: int,
) -> np.ndarray:
    blank = np.zeros((VIEW_H, VIEW_W, 3), np.uint8)

    tiles = [cv2.resize(v, (VIEW_W, VIEW_H)) for v in views[:n_cols]]
    while len(tiles) < n_cols:
        tiles.append(blank.copy())

    strip   = np.hstack(tiles)
    strip_w = strip.shape[1]

    # Header bar: coloured swatch on left, instance label
    header = np.full((HEADER_H, strip_w, 3), 35, np.uint8)
    cv2.rectangle(header, (6, 4), (22, HEADER_H - 4), color_bgr, -1)
    cv2.putText(header, inst_id, (30, HEADER_H - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)

    return np.vstack([header, strip])


# ── Main ─────────────────────────────────────────────────────────────────────

def visualize_2d(sample_id: int, run_dir: Path, views_per_component: int = 8) -> Path:
    out_dir      = run_dir / f"sample_{sample_id}"
    results_path = out_dir / "results.json"
    dust3r_path  = out_dir / "dust3r" / "dust3r_cache.pkl"
    sam_dir      = out_dir / "sam"
    crop_dir     = out_dir / "cropped"

    if not results_path.exists():
        raise FileNotFoundError(
            f"No results.json for sample {sample_id}. Run pipeline first.")

    with open(results_path) as f:
        results = json.load(f)
    with open(dust3r_path, "rb") as f:
        dust3r = pickle.load(f)

    # Prefer the newest filter generation (filt3 > filt2 > filt > legacy)
    sam_files = (sorted(sam_dir.glob("sam_masks_top*_filt3.pkl"))
                 or sorted(sam_dir.glob("sam_masks_top*_filt2.pkl"))
                 or sorted(sam_dir.glob("sam_masks_top*_filt.pkl"))
                 or sorted(sam_dir.glob("sam_masks_top*.pkl")))
    with open(sam_files[0], "rb") as f:
        sam_masks = pickle.load(f)

    pts3d_all = dust3r["pts3d"]

    order_file = out_dir / "image_order.json"
    if order_file.exists():
        with open(order_file) as f:
            img_paths = [Path(p) for p in json.load(f)]
    else:
        img_paths = sorted(crop_dir.glob("*.jpg")) + sorted(crop_dir.glob("*.JPG"))

    instance_ids       = [inst["instance_id"] for inst in results["instances"]]
    instance_centroids = [np.array(inst["centroid"], np.float32)
                          for inst in results["instances"]]

    n_frames = min(len(sam_masks), len(pts3d_all), len(img_paths))
    print(f"[Viz2D] {len(instance_ids)} components, {n_frames} views available")

    all_strips = []

    for ci, inst_id in enumerate(instance_ids):
        color        = _bgr(ci)
        found_views  = []

        for vi in range(n_frames):
            bgr = cv2.imread(str(img_paths[vi]))
            if bgr is None:
                continue

            ann, found = _annotate_for_component(
                bgr, sam_masks[vi], pts3d_all[vi],
                instance_centroids, ci, inst_id, color,
            )

            if found:
                # View index label top-left
                cv2.putText(ann, f"v{vi}", (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1, cv2.LINE_AA)
                found_views.append(ann)
                if len(found_views) >= views_per_component:
                    break

        if not found_views:
            placeholder = np.full((VIEW_H, VIEW_W, 3), 40, np.uint8)
            cv2.putText(placeholder, "not detected",
                        (10, VIEW_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (120, 120, 120), 1, cv2.LINE_AA)
            found_views = [placeholder]
            print(f"[Viz2D]   {inst_id}: not detected in any view")
        else:
            print(f"[Viz2D]   {inst_id}: {len(found_views)} view(s)")

        all_strips.append(_make_strip(found_views, inst_id, color, views_per_component))

    final    = np.vstack(all_strips)
    out_path = out_dir / "viz_2d.png"
    cv2.imwrite(str(out_path), final)
    print(f"[Viz2D] Saved → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-component 2D instance overlay")
    parser.add_argument("--sample",              type=int, required=True)
    parser.add_argument("--run",                 type=str, default=None,
                        help="Run folder name. Defaults to most recent run_* folder.")
    parser.add_argument("--views-per-component", type=int, default=8,
                        help="Max views to show per component (default 8)")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run)
    print(f"[Viz2D] Using run: {run_dir.name}")
    visualize_2d(args.sample, run_dir, args.views_per_component)
