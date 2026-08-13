"""
Debug tool: visualise the actual kept SAM masks per view (post-filter,
post-banded-keep, PRE 3D-clustering) — one colour per mask.

This is upstream of viz_2d.py: viz_2d.py shows masks AFTER 3D clustering and
manifest assignment, so a mislabelled/collapsed cluster there can make good
per-view SAM masks look wrong. This tool shows exactly what SAM + the filter
pipeline produced per view, before any of that.

Loads from the SAM cache written by a previous pipeline run (`run_sam_on_images`
returns immediately on a cache hit — no model load, no GPU needed) — point it
at any run directory that already processed this sample.

Usage:
    python visualize_sam_masks.py [sample_id] [--run RUN_DIR_NAME] [--max-views N]
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

import config as cfg
from detection.sam_runner import run_sam_on_images
from puml_parser import find_puml, parse_puml

_TILE_SIZE  = 320
_HEADER_H   = 28
_MASK_ALPHA = 0.45


def _find_run_dir(sample_id: int, run: str | None) -> Path:
    if run is not None:
        run_dir = cfg.OUTPUTS_ROOT / run
        if not (run_dir / f"sample_{sample_id}").exists():
            raise FileNotFoundError(f"{run_dir}/sample_{sample_id} not found")
        return run_dir
    candidates = sorted(cfg.OUTPUTS_ROOT.glob(f"run_*/sample_{sample_id}"),
                        key=lambda p: p.parent.name)
    if not candidates:
        raise FileNotFoundError(
            f"No run with sample_{sample_id} found under {cfg.OUTPUTS_ROOT} — "
            f"run the pipeline for this sample first (e.g. ./slow.sh {sample_id})")
    return candidates[-1].parent


def _mask_palette(n: int) -> list[tuple[int, int, int]]:
    colours = []
    for i in range(max(n, 1)):
        hue = int(180 * i / max(n, 1))
        bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colours.append(tuple(int(c) for c in bgr))
    return colours


def _annotate_tile(base_bgr: np.ndarray, masks: list[dict],
                   tile_size: int = _TILE_SIZE) -> np.ndarray:
    overlay = base_bgr.copy()
    palette = _mask_palette(len(masks))
    centroids = []
    for m, colour in zip(masks, palette):
        seg = m["segmentation"]
        overlay[seg] = colour
        ys, xs = np.where(seg)
        centroids.append((int(xs.mean()), int(ys.mean())) if len(ys) else None)

    vis = cv2.addWeighted(overlay, _MASK_ALPHA, base_bgr, 1 - _MASK_ALPHA, 0)
    for i, c in enumerate(centroids):
        if c is None:
            continue
        cv2.putText(vis, str(i), c, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                   (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, str(i), c, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                   (255, 255, 255), 1, cv2.LINE_AA)

    h, w = vis.shape[:2]
    scale = tile_size / max(h, w, 1)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(vis, (nw, nh))
    tile = np.zeros((tile_size, tile_size, 3), np.uint8)
    oy, ox = (tile_size - nh) // 2, (tile_size - nw) // 2
    tile[oy:oy + nh, ox:ox + nw] = resized
    return tile


def _make_grid(tiles: list[np.ndarray], labels: list[str]) -> np.ndarray:
    n = len(tiles)
    cols = min(6, max(1, int(np.ceil(np.sqrt(n)))))
    rows = int(np.ceil(n / cols))
    tile_size = tiles[0].shape[0]
    canvas = np.full((rows * (tile_size + _HEADER_H), cols * tile_size, 3), 40, np.uint8)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        r, c = divmod(i, cols)
        y0, x0 = r * (tile_size + _HEADER_H), c * tile_size
        cv2.putText(canvas, label, (x0 + 4, y0 + _HEADER_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[y0 + _HEADER_H: y0 + _HEADER_H + tile_size, x0: x0 + tile_size] = tile
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sample_id", type=int, nargs="?", default=113)
    p.add_argument("--run", type=str, default=None,
                   help="run dir name (e.g. run_034_20260807_1028); default: latest")
    p.add_argument("--max-views", type=int, default=None)
    args = p.parse_args()

    run_dir    = _find_run_dir(args.sample_id, args.run)
    sample_dir = run_dir / f"sample_{args.sample_id}"
    cropped    = sample_dir / "cropped"
    img_paths  = sorted(cropped.glob("*.jpg")) + sorted(cropped.glob("*.JPG"))
    if not img_paths:
        raise FileNotFoundError(f"No cropped images found in {cropped}")
    if args.max_views:
        img_paths = img_paths[:args.max_views]

    manifest = parse_puml(find_puml(args.sample_id))
    print(f"[Viz] {run_dir.name} / sample_{args.sample_id}  "
          f"({manifest.n_components} manifest components, {len(img_paths)} views)")

    # Cache hit only: run_sam_on_images returns immediately from the pickle
    # written by the original run without touching the SAM model/GPU, as long
    # as n_components / grid_spacing / dt_seeds / min_area / max_area_frac
    # match what that run used (defaults, unless overridden at run time).
    all_masks = run_sam_on_images(
        img_paths, manifest.n_components, sample_dir / "sam",
        weights_path=cfg.SAM_WEIGHTS, model_type=cfg.SAM_MODEL_TYPE, device=cfg.DEVICE,
        points_per_side=cfg.SAM_POINTS_SIDE, iou_thresh=cfg.SAM_IOU_THRESH,
        stability_thresh=cfg.SAM_STABILITY, min_area=cfg.SAM_MIN_AREA,
        max_area_frac=cfg.SAM_MAX_AREA_FRAC, fg_overlap_min=cfg.SAM_FG_OVERLAP_MIN,
        dedup_iou=cfg.SAM_DEDUP_IOU, keep_factor=cfg.SAM_KEEP_FACTOR,
        mode="prompted", grid_spacing=cfg.SAM_GRID_SPACING, dt_seeds=cfg.SAM_DT_SEEDS,
    )

    tiles, labels = [], []
    for img_path, masks in zip(img_paths, all_masks):
        base_bgr = cv2.imread(str(img_path))
        tiles.append(_annotate_tile(base_bgr, masks))
        ious = [m["predicted_iou"] for m in masks]
        mean_iou = f"{np.mean(ious):.2f}" if ious else "-"
        labels.append(f"{img_path.stem}  n={len(masks)} iou~{mean_iou}")
        print(f"[Viz]   {img_path.name}: {len(masks)} masks, mean predicted_iou={mean_iou}")

    grid_img = _make_grid(tiles, labels)
    out_dir  = sample_dir / "sam_masks_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "masks_overlay.png"
    cv2.imwrite(str(out_path), grid_img)
    print(f"[Viz] Saved → {out_path}")


if __name__ == "__main__":
    main()
