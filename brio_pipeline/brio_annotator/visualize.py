#!/usr/bin/env python3
"""
Visualize brio_annotator annotation results across multiple views.

Rendering style mirrors brio_3d_pipeline/visualize_2d.py:
  - Each tile is CROPPED to the construction bounding box (white background removed).
  - Primary view: actual SAM pixel masks drawn as semi-transparent coloured fills
    with a bright contour outline and label at the mask centroid.
  - Propagated views: semi-transparent filled bounding rectangle (no mask available).
  - All drawing happens at full image resolution; tiles are resized only at the end.

Two output images per sample:

  viz_multiview.png
      Grid of all labelled images.  Every annotation shown in its class colour.

  viz_per_component.png
      One horizontal strip per PUML component.  Target component highlighted in
      full colour; all others dimmed to grey for spatial context.

Usage
─────
    python visualize.py --sample 113
    python visualize.py --sample 113 --run run_002_20260616_2305 --cols 6
"""
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ann_config as cfg

BG_THRESHOLD = 245    # pixels brighter than this on all channels = background


# ── Layout constants ──────────────────────────────────────────────────────────

VIEW_W   = 480    # tile width  (after crop-and-resize)
VIEW_H   = 360    # tile height
HEADER_H = 32     # height of the per-strip label bar
CROP_PAD = 25     # padding (px) around construction crop

ALPHA_HI = 0.45   # fill alpha for highlighted component
ALPHA_LO = 0.18   # fill alpha for grey-dimmed context components


# ── Colour palette (BGR) ──────────────────────────────────────────────────────

_PALETTE = [
    (  0, 114, 189), (217,  83,  25), (237, 177,  32),
    (126,  47, 142), (119, 172,  48), ( 77, 190, 238),
    (162,  20,  47), (  0, 168, 107), (255, 127,   0),
    ( 77,  77, 255), (189,  62, 189), ( 25, 133, 161),
]

def _color(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


# ── Foreground crop ───────────────────────────────────────────────────────────

def _crop_to_fg(img_bgr: np.ndarray,
                pad: int = CROP_PAD,
                ) -> tuple[np.ndarray, int, int]:
    """
    Crop img_bgr to the bounding box of the construction (non-white region).
    Returns (cropped_img, x_off, y_off).  x_off / y_off are the top-left corner
    of the crop in original image coordinates — used to translate masks and labels.
    """
    H, W = img_bgr.shape[:2]
    fg   = np.any(img_bgr < BG_THRESHOLD, axis=2)
    ys, xs = np.where(fg)
    if len(ys) == 0:
        return img_bgr, 0, 0
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + pad)
    return img_bgr[y0:y1, x0:x1].copy(), x0, y0


# ── Run-dir resolver ──────────────────────────────────────────────────────────

def _resolve_run(run_arg: str | None) -> Path:
    if run_arg:
        d = cfg.OUTPUTS_ROOT / run_arg
        if d.exists():
            return d
        raise FileNotFoundError(f"Run dir not found: {d}")
    dirs = sorted(
        [d for d in cfg.OUTPUTS_ROOT.iterdir()
         if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.stat().st_mtime,
    )
    if not dirs:
        raise FileNotFoundError(f"No run_* dirs in {cfg.OUTPUTS_ROOT}")
    return dirs[-1]


# ── YOLO label parsing ────────────────────────────────────────────────────────

def _load_labels(label_path: Path, img_w: int,
                  img_h: int) -> list[tuple[str, int, int, int, int]]:
    """Parse a YOLO .txt → list of (class_name, x0, y0, x1, y1) in pixel coords."""
    if not label_path.exists():
        return []
    entries = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        ci = int(parts[0])
        cx, cy, w, h = (float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]))
        cls = cfg.CLASSES[ci] if ci < len(cfg.CLASSES) else f"cls{ci}"
        x0  = int((cx - w / 2) * img_w)
        y0  = int((cy - h / 2) * img_h)
        x1  = int((cx + w / 2) * img_w)
        y1  = int((cy + h / 2) * img_h)
        entries.append((cls, x0, y0, x1, y1))
    return entries


def _translate_labels(labels: list[tuple[str, int, int, int, int]],
                       x_off: int, y_off: int,
                       ) -> list[tuple[str, int, int, int, int]]:
    """Shift label pixel coords by -x_off, -y_off (for construction crop)."""
    return [(cls, x0 - x_off, y0 - y_off, x1 - x_off, y1 - y_off)
            for cls, x0, y0, x1, y1 in labels]


# ── Core drawing primitives ───────────────────────────────────────────────────

def _draw_mask_overlay(canvas: np.ndarray,
                        mask: np.ndarray,
                        color: tuple[int, int, int],
                        alpha: float) -> np.ndarray:
    """Semi-transparent coloured fill over mask pixels + contour outline."""
    H, W = canvas.shape[:2]
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    color_f = np.array(color, np.float32)
    flat    = canvas.astype(np.float32)
    flat[mask] = flat[mask] * (1 - alpha) + color_f * alpha
    out = flat.astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def _draw_bbox_overlay(canvas: np.ndarray,
                        bbox: tuple[int, int, int, int],
                        color: tuple[int, int, int],
                        alpha: float) -> np.ndarray:
    """Semi-transparent filled rectangle for propagated-view annotations."""
    x0, y0, x1, y1 = bbox
    H, W = canvas.shape[:2]
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    if x1c <= x0c or y1c <= y0c:
        return canvas
    out     = canvas.copy().astype(np.float32)
    color_f = np.array(color, np.float32)
    out[y0c:y1c, x0c:x1c] = (out[y0c:y1c, x0c:x1c] * (1 - alpha)
                               + color_f * alpha)
    out = out.astype(np.uint8)
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    return out


def _grey_dim_mask(canvas: np.ndarray, mask: np.ndarray) -> np.ndarray:
    H, W = canvas.shape[:2]
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    grey = np.array([60, 60, 60], np.float32)
    flat = canvas.astype(np.float32)
    flat[mask] = flat[mask] * (1 - ALPHA_LO) + grey * ALPHA_LO
    out = flat.astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (80, 80, 80), 1)
    return out


def _grey_dim_bbox(canvas: np.ndarray,
                    bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    H, W = canvas.shape[:2]
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    if x1c <= x0c or y1c <= y0c:
        return canvas
    out  = canvas.copy().astype(np.float32)
    grey = np.array([60, 60, 60], np.float32)
    out[y0c:y1c, x0c:x1c] = out[y0c:y1c, x0c:x1c] * (1 - ALPHA_LO) + grey * ALPHA_LO
    out = out.astype(np.uint8)
    cv2.rectangle(out, (x0, y0), (x1, y1), (80, 80, 80), 1)
    return out


def _label_at_centroid(canvas: np.ndarray, mask_or_bbox, text: str,
                        color: tuple[int, int, int]) -> np.ndarray:
    if isinstance(mask_or_bbox, np.ndarray):
        ys, xs = np.where(mask_or_bbox)
        if len(xs) == 0:
            return canvas
        cx, cy = int(xs.mean()), int(ys.mean())
    else:
        x0, y0, x1, y1 = mask_or_bbox
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

    H, W = canvas.shape[:2]
    fs, th = 0.50, 1
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    lx = max(0, min(W - tw - 4, cx - tw // 2))
    ly = min(H - 4, max(14, cy + 4))
    cv2.rectangle(canvas, (lx - 2, ly - 14), (lx + tw + 2, ly + 2),
                  (20, 20, 20), -1)
    cv2.putText(canvas, text, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return canvas


# ── Per-view renderers ────────────────────────────────────────────────────────

def _render_all_components(img_bgr     : np.ndarray,
                             annotations : list[dict],
                             masks       : dict[str, np.ndarray],
                             color_map   : dict[str, tuple],
                             primary_stem: str,
                             labels      : list[tuple[str, int, int, int, int]],
                             img_stem    : str,
                             ) -> np.ndarray:
    """Draw all components on one image tile, cropped to construction region."""
    crop, x_off, y_off = _crop_to_fg(img_bgr)
    cH, cW = crop.shape[:2]
    canvas     = crop.copy()
    is_primary = (img_stem == primary_stem)

    if is_primary and masks:
        for ann in annotations:
            iid   = ann["instance_id"]
            color = color_map[ann["cls"]]
            mask  = masks.get(iid)
            if mask is None:
                continue
            mask_crop = mask[y_off:y_off + cH, x_off:x_off + cW].astype(bool)
            if not mask_crop.any():
                continue
            canvas = _draw_mask_overlay(canvas, mask_crop, color, ALPHA_HI)
            canvas = _label_at_centroid(canvas, mask_crop, iid, color)
    else:
        lbls_local = _translate_labels(labels, x_off, y_off)
        for cls, x0, y0, x1, y1 in lbls_local:
            color  = color_map.get(cls, (160, 160, 160))
            canvas = _draw_bbox_overlay(canvas, (x0, y0, x1, y1), color, ALPHA_HI)
        for cls, x0, y0, x1, y1 in lbls_local:
            color  = color_map.get(cls, (160, 160, 160))
            canvas = _label_at_centroid(canvas, (x0, y0, x1, y1), cls, color)

    return canvas


def _render_component_view(img_bgr     : np.ndarray,
                             annotations : list[dict],
                             masks       : dict[str, np.ndarray],
                             color_map   : dict[str, tuple],
                             primary_stem: str,
                             labels      : list[tuple[str, int, int, int, int]],
                             img_stem    : str,
                             target_cls  : str,
                             target_iid  : str,
                             color       : tuple,
                             ) -> tuple[np.ndarray, bool]:
    """Draw one tile for the per-component strip, cropped to construction region."""
    crop, x_off, y_off = _crop_to_fg(img_bgr)
    cH, cW = crop.shape[:2]
    canvas     = crop.copy()
    is_primary = (img_stem == primary_stem)
    found      = False

    if is_primary and masks:
        # First pass: grey-dim all non-target masks
        for ann in annotations:
            if ann["cls"] == target_cls:
                continue
            mask = masks.get(ann["instance_id"])
            if mask is None:
                continue
            mc = mask[y_off:y_off + cH, x_off:x_off + cW].astype(bool)
            if mc.any():
                canvas = _grey_dim_mask(canvas, mc)

        # Second pass: highlight target mask
        target_mask_full = masks.get(target_iid)
        if target_mask_full is not None:
            mc = target_mask_full[y_off:y_off + cH, x_off:x_off + cW].astype(bool)
            if mc.any():
                canvas = _draw_mask_overlay(canvas, mc, color, ALPHA_HI)
                canvas = _label_at_centroid(canvas, mc, target_iid, color)
                found = True
    else:
        lbls_local  = _translate_labels(labels, x_off, y_off)
        target_bbox = None
        for cls, x0, y0, x1, y1 in lbls_local:
            if cls == target_cls:
                target_bbox = (x0, y0, x1, y1)
            else:
                canvas = _grey_dim_bbox(canvas, (x0, y0, x1, y1))
        if target_bbox is not None:
            canvas = _draw_bbox_overlay(canvas, target_bbox, color, ALPHA_HI)
            canvas = _label_at_centroid(canvas, target_bbox, target_iid, color)
            found = True

    return canvas, found


# ── Strip builder ─────────────────────────────────────────────────────────────

def _make_strip(tiles: list[np.ndarray], inst_id: str,
                 cls: str, color: tuple, n_cols: int) -> np.ndarray:
    blank = np.zeros((VIEW_H, VIEW_W, 3), np.uint8)
    row   = [cv2.resize(t, (VIEW_W, VIEW_H)) for t in tiles[:n_cols]]
    while len(row) < n_cols:
        row.append(blank.copy())
    strip_w = VIEW_W * n_cols
    header  = np.full((HEADER_H, strip_w, 3), 35, np.uint8)
    cv2.rectangle(header, (6, 4), (22, HEADER_H - 4), color, -1)
    cv2.putText(header, f"{inst_id}  ({cls})", (30, HEADER_H - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([header, np.hstack(row)])


# ── Main visualiser ───────────────────────────────────────────────────────────

def visualize(sample_id: int, run_dir: Path, cols: int = 6) -> tuple[Path, Path]:
    out_dir      = run_dir / f"sample_{sample_id}"
    imgs_dir     = out_dir / "images"
    labels_dir   = out_dir / "labels"
    summary_path = out_dir / "annotation_summary.json"
    masks_path   = out_dir / "annotation_masks.npz"
    primary_file = out_dir / "primary_stem.txt"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"annotation_summary.json not found for sample {sample_id}. "
            "Run annotator.py first.")

    summary     = json.loads(summary_path.read_text())
    annotations = summary.get("annotations", [])

    masks: dict[str, np.ndarray] = {}
    if masks_path.exists():
        data  = np.load(str(masks_path))
        masks = {k: data[k] for k in data.files}
    else:
        print("[Viz] WARNING: annotation_masks.npz not found — "
              "re-run annotator.py to regenerate with mask data")

    primary_stem = primary_file.read_text().strip() if primary_file.exists() else ""

    unique_cls = list(dict.fromkeys(a["cls"] for a in annotations))
    color_map  = {cls: _color(i) for i, cls in enumerate(unique_cls)}

    all_imgs = sorted(imgs_dir.glob("*.jpg")) + sorted(imgs_dir.glob("*.JPG"))
    labeled  = [p for p in all_imgs
                if (labels_dir / p.with_suffix(".txt").name).exists()]
    if not labeled:
        raise FileNotFoundError(f"No labeled images found in {imgs_dir}")

    print(f"[Viz] Sample {sample_id}: {len(labeled)} labeled views, "
          f"{len(annotations)} components  "
          f"({'with' if masks else 'WITHOUT'} SAM masks)")

    # ── Multi-view grid ───────────────────────────────────────────────────────
    tiles: list[np.ndarray] = []
    for img_path in labeled:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W  = img.shape[:2]
        stem  = img_path.stem
        lbl_p = labels_dir / img_path.with_suffix(".txt").name
        lbls  = _load_labels(lbl_p, W, H)

        tile  = _render_all_components(img, annotations, masks, color_map,
                                        primary_stem, lbls, stem)

        elev  = stem.split("_")[0]
        tag   = f"[P] {elev}" if stem == primary_stem else elev
        cv2.putText(tile, tag, (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 50), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(tile, (VIEW_W, VIEW_H)))

    n_rows = max(1, -(-len(tiles) // cols))
    blank  = np.zeros((VIEW_H, VIEW_W, 3), np.uint8)
    while len(tiles) < n_rows * cols:
        tiles.append(blank.copy())

    grid      = np.vstack([np.hstack(tiles[i * cols:(i + 1) * cols])
                           for i in range(n_rows)])
    grid_path = out_dir / "viz_multiview.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"[Viz] Multi-view grid     → {grid_path}")

    # ── Per-component strips ──────────────────────────────────────────────────
    all_strips: list[np.ndarray] = []

    for ann in annotations:
        cls     = ann["cls"]
        inst_id = ann["instance_id"]
        color   = color_map[cls]

        component_tiles: list[np.ndarray] = []
        for img_path in labeled:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W  = img.shape[:2]
            stem  = img_path.stem
            lbl_p = labels_dir / img_path.with_suffix(".txt").name
            lbls  = _load_labels(lbl_p, W, H)

            tile, found = _render_component_view(
                img, annotations, masks, color_map,
                primary_stem, lbls, stem,
                cls, inst_id, color,
            )
            if not found:
                continue

            elev = stem.split("_")[0]
            cv2.putText(tile, elev, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 50), 1, cv2.LINE_AA)
            component_tiles.append(cv2.resize(tile, (VIEW_W, VIEW_H)))
            if len(component_tiles) >= cols:
                break

        if not component_tiles:
            placeholder = np.full((VIEW_H, VIEW_W, 3), 40, np.uint8)
            cv2.putText(placeholder, "not in any view",
                        (8, VIEW_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (90, 90, 90), 1, cv2.LINE_AA)
            component_tiles = [placeholder]
            print(f"[Viz]   {inst_id}: not found in any view")
        else:
            has_mask = masks.get(inst_id) is not None
            print(f"[Viz]   {inst_id}: {len(component_tiles)} view(s)"
                  f"{'  [primary mask]' if has_mask else ''}")

        all_strips.append(_make_strip(component_tiles, inst_id, cls, color, cols))

    comp_path = out_dir / "viz_per_component.png"
    cv2.imwrite(str(comp_path), np.vstack(all_strips))
    print(f"[Viz] Per-component strips → {comp_path}")

    return grid_path, comp_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize brio_annotator results")
    parser.add_argument("--sample", type=int,  required=True)
    parser.add_argument("--run",    type=str,  default=None,
                        help="Run folder name (default: most recent run_*)")
    parser.add_argument("--cols",   type=int,  default=6,
                        help="View tiles per row / per strip (default 6)")
    args = parser.parse_args()

    run_dir = _resolve_run(args.run)
    print(f"[Viz] Using run: {run_dir.name}")
    visualize(args.sample, run_dir, cols=args.cols)
