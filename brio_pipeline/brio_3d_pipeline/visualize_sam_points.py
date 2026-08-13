"""
Debug tool: visualise where SAM's prompt points (uniform grid + edge-aware
DT seeds) land, without running SAM or DUSt3R.

Reuses the exact preprocessing chain `_run_sam_prompted` uses before calling
`_fg_grid_points` (foreground mask -> tight crop -> black background -> CLAHE),
so the points shown are exactly what the real pipeline would prompt SAM with.

Usage:
    python visualize_sam_points.py [sample_id] [--grid-spacing N] [--no-dt-seeds] [--max-views N]
"""
import argparse
import cv2
import numpy as np
from pathlib import Path

import config as cfg
from detection.sam_runner import _foreground_mask, _tight_fg_crop, _clahe_enhance, _fg_grid_points

_TILE_SIZE = 320 # size of each tile in the final grid
_HEADER_H  = 28
_COLOUR_GRID = (255, 255, 0)   # BGR cyan
_COLOUR_DT   = (0, 0, 255)     # BGR red

# fetch the sample's multi-view images (fixed-half crops if available, else raw images)
def _find_cached_crops(sample_id: int) -> Path | None:
    """Most recent run's fixed-half crop dir for this sample, if one exists."""
    candidates = sorted(cfg.OUTPUTS_ROOT.glob(f"run_*/sample_{sample_id}/cropped"))
    return candidates[-1] if candidates else None


def _load_view_images(sample_id: int) -> list[Path]:
    cropped_dir = _find_cached_crops(sample_id)
    if cropped_dir is not None:
        imgs = sorted(cropped_dir.glob("*.jpg")) + sorted(cropped_dir.glob("*.JPG"))
        if imgs:
            print(f"[Viz] Using cached fixed-half crops from {cropped_dir}")
            return imgs

    # Fallback: raw multi-view images at their native resolution. Points still
    # preview correctly (grid spacing is a fixed pixel value regardless of
    # image scale), but this isn't the exact fixed-half scale a real pipeline
    # run would crop to — no cached run exists yet to read that scale from.
    from pipeline import collect_images
    print(f"[Viz] No cached crop found for sample {sample_id} — using raw "
          f"multi-view images (preview only, not fixed-half scale)")
    return collect_images(sample_id)

# replay the preprocessing chain 
def _points_for_view(img_path: Path, grid_spacing: int, dt_seeds: bool):
    image_bgr = cv2.imread(str(img_path))
    fg = _foreground_mask(image_bgr)
    crop_bgr, crop_fg, _, _ = _tight_fg_crop(image_bgr, fg)
    crop_bgr[~crop_fg] = 0
    crop_bgr = _clahe_enhance(crop_bgr)
    pts, is_dt = _fg_grid_points(crop_fg, grid_spacing, dt_seeds=dt_seeds,
                                 bgr=crop_bgr, return_source=True)
    return crop_bgr, pts, is_dt


def _annotate_tile(crop_bgr: np.ndarray, pts: np.ndarray, is_dt: np.ndarray,
                   tile_size: int = _TILE_SIZE) -> np.ndarray:
    vis = crop_bgr.copy()
    for (x, y), dt in zip(pts.astype(int), is_dt):
        colour = _COLOUR_DT if dt else _COLOUR_GRID
        cv2.circle(vis, (int(x), int(y)), 4 if dt else 3, colour, -1, cv2.LINE_AA)

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
    p.add_argument("--grid-spacing", type=int, default=None,
                   help="override config.SAM_GRID_SPACING")
    p.add_argument("--no-dt-seeds", action="store_true",
                   help="disable edge-aware DT seeds (grid points only)")
    p.add_argument("--max-views", type=int, default=None,
                   help="limit number of views shown (default: all)")
    args = p.parse_args()

    grid_spacing = args.grid_spacing or cfg.SAM_GRID_SPACING
    dt_seeds = cfg.SAM_DT_SEEDS and not args.no_dt_seeds

    img_paths = _load_view_images(args.sample_id)
    if args.max_views:
        img_paths = img_paths[:args.max_views]

    tiles, labels = [], []
    n_grid_total = n_dt_total = 0
    for img_path in img_paths:
        crop_bgr, pts, is_dt = _points_for_view(img_path, grid_spacing, dt_seeds)
        n_dt   = int(is_dt.sum())
        n_grid = len(pts) - n_dt
        n_grid_total += n_grid
        n_dt_total   += n_dt
        tiles.append(_annotate_tile(crop_bgr, pts, is_dt))
        labels.append(f"{img_path.stem}  grid={n_grid} dt={n_dt}")
        print(f"[Viz]   {img_path.name}: grid={n_grid}  dt={n_dt}  total={len(pts)}")

    grid_img = _make_grid(tiles, labels)
    out_dir = cfg.OUTPUTS_ROOT / f"sample_{args.sample_id}" / "sam_points_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"points_sp{grid_spacing}_dt{int(dt_seeds)}.png"
    cv2.imwrite(str(out_path), grid_img)

    print(f"[Viz] {len(img_paths)} views  —  grid={n_grid_total}  dt={n_dt_total}  "
          f"total={n_grid_total + n_dt_total}  (spacing={grid_spacing}, dt_seeds={dt_seeds})")
    print(f"[Viz] cyan = uniform grid point, red = edge-aware DT seed")
    print(f"[Viz] Saved → {out_path}")


if __name__ == "__main__":
    main()
