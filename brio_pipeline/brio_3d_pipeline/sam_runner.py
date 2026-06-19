"""
Run SAM on all images for a sample.

Two modes are available (select via run_sam_on_images(..., mode=...)):

  "auto"     — SamAutomaticMaskGenerator with a uniform 32-pt grid.  The
               original approach.  Grid covers the full fixed-scale crop.

  "prompted" — SamPredictor with foreground-only grid prompts (DEFAULT).
               Before encoding, the image is cropped to the tight construction
               bounding box (white background removed completely) and the
               background is filled with pure black so SAM sees only the
               construction.  Grid points are placed only inside the foreground
               at a spacing tuned to expected component density, so no SAM
               queries are wasted on background or padding pixels.
               Masks are translated back to fixed-scale coordinates so the
               backprojector works unchanged.

Mask post-filtering (both modes):
  1. fg-overlap  — < SAM_FG_OVERLAP_MIN overlap with fg → drop
  2. border      — touching the image border → drop
  3. max-area    — > SAM_MAX_AREA_FRAC of fg → decompose by subtracting smaller masks
  4. dedup       — IoU > SAM_DEDUP_IOU → keep higher-confidence one
  5. container   — mask largely covered by smaller masks → keep only the remainder
  6. blob-split  — disconnected blobs inside one mask → split into candidates
"""
import pickle
import numpy as np
from pathlib import Path
import cv2


try:
    from config import CLAHE_CLIP_LIMIT as _CLAHE_CLIP_LIMIT
except ImportError:
    _CLAHE_CLIP_LIMIT = 3.0
_CLAHE = cv2.createCLAHE(clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))

# Same threshold as preprocessor.BG_THRESHOLD — cropped images have a white
# background (and white padding), so "non-white" is the foreground signal.
BG_THRESHOLD = 245


def _clahe_enhance(bgr: np.ndarray) -> np.ndarray:
    """Boost local contrast in LAB space so white components stand out from white backgrounds."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _foreground_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Largest connected non-white region of the ORIGINAL (pre-CLAHE) image,
    morphologically closed and slightly dilated. Same logic as the
    preprocessor LCC detection, so the foreground matches the crop window.
    """
    mask   = np.any(bgr < BG_THRESHOLD, axis=2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels < 2:
        return mask.astype(bool)   # no components found — keep everything non-white
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg   = (labels == best).astype(np.uint8)
    fg   = cv2.dilate(fg, kernel)   # tolerance for masks slightly past the LCC edge
    return fg.astype(bool)


# ── Tight-crop helpers (prompted mode) ───────────────────────────────────────

def _tight_fg_crop(bgr: np.ndarray, fg: np.ndarray,
                   pad: int = 15) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Crop bgr to the tight bounding box of the foreground mask.
    Returns (crop_bgr, crop_fg, x_off, y_off).
    x_off / y_off are the top-left corner of the crop in the original image.
    Used so SAM encodes only the construction area at full 1024×1024 resolution.
    """
    ys, xs = np.where(fg)
    if len(ys) == 0:
        return bgr.copy(), fg.copy(), 0, 0
    H, W = bgr.shape[:2]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + pad)
    return bgr[y0:y1, x0:x1].copy(), fg[y0:y1, x0:x1].copy(), x0, y0


def _expand_masks(masks_crop: list[dict], orig_H: int, orig_W: int,
                  x_off: int, y_off: int) -> list[dict]:
    """
    Translate masks from tight-crop coordinates back to fixed-scale image space
    so the backprojector can index them against DUSt3R pts3d correctly.
    """
    expanded = []
    for m in masks_crop:
        seg_crop = m["segmentation"]
        ch, cw = seg_crop.shape
        full = np.zeros((orig_H, orig_W), bool)
        ey0, ey1 = y_off, min(orig_H, y_off + ch)
        ex0, ex1 = x_off, min(orig_W, x_off + cw)
        full[ey0:ey1, ex0:ex1] = seg_crop[:ey1 - ey0, :ex1 - ex0]
        expanded.append({**m, "segmentation": full})
    return expanded


def _fg_grid_points(fg_mask: np.ndarray, n_components: int,
                    density: float = 2.5, min_spacing: int = 12,
                    ) -> np.ndarray:
    """
    Place a regular grid of points inside the foreground mask.

    Spacing is chosen so there are approximately density × n_components points
    total — enough to guarantee every component gets at least one prompt while
    keeping the number of SAM decoder calls low.

    Returns (M, 2) float32 array of (x, y) coordinates.
    """
    ys, xs = np.where(fg_mask)
    if len(ys) == 0:
        return np.empty((0, 2), np.float32)

    fg_area = int(len(ys))
    n_target = max(1, int(n_components * density))
    spacing  = max(min_spacing, int((fg_area / n_target) ** 0.5))

    H, W = fg_mask.shape
    gx = np.arange(spacing // 2, W, spacing)
    gy = np.arange(spacing // 2, H, spacing)
    grid_x, grid_y = np.meshgrid(gx, gy)
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(int)

    # Keep only grid points that fall inside the foreground
    valid_x = np.clip(pts[:, 0], 0, W - 1)
    valid_y = np.clip(pts[:, 1], 0, H - 1)
    inside  = fg_mask[valid_y, valid_x]
    return pts[inside].astype(np.float32)


def _filter_masks(masks: list[dict], fg: np.ndarray,
                  max_area_frac: float, fg_overlap_min: float,
                  dedup_iou: float, min_area: int = 100) -> list[dict]:
    """Reject background / duplicate masks; decompose whole-object masks.

    Oversized masks that genuinely lie on the foreground are NOT dropped:
    when one component dominates the silhouette (e.g. a block with a small
    bolt), the "whole-object" mask IS that component plus the small parts.
    Subtracting the smaller kept masks recovers the dominant component.
    """
    fg_area = max(int(fg.sum()), 1)

    normal, oversized = [], []
    for m in masks:
        seg = m["segmentation"]
        # 1. Mostly outside the foreground → background / table
        inside = np.logical_and(seg, fg).sum()
        if inside < fg_overlap_min * m["area"]:
            continue
        # 2. Touching the image border → background / padding
        if seg[0, :].any() or seg[-1, :].any() or seg[:, 0].any() or seg[:, -1].any():
            continue
        # 3. Oversized but on-object → candidate for decomposition
        if m["area"] > max_area_frac * fg_area:
            oversized.append(m)
        else:
            normal.append(m)

    # 4. Deduplicate normal masks, preferring higher predicted IoU
    normal.sort(key=lambda m: m["predicted_iou"], reverse=True)
    final: list[dict] = []
    for m in normal:
        is_dup = False
        for f in final:
            inter = np.logical_and(m["segmentation"], f["segmentation"]).sum()
            union = m["area"] + f["area"] - inter
            if union > 0 and inter / union > dedup_iou:
                is_dup = True
                break
        if not is_dup:
            final.append(m)

    # 4b. Drop/decompose CONTAINER masks at every scale, not just oversized
    #     ones: a mask largely covered by the union of smaller kept masks is
    #     a multi-part union (e.g. "block+nut").  Such masks bridge instances
    #     in the 3D overlap graph and merge different components.  Keep only
    #     finest-granularity masks; a substantial remainder survives as its
    #     own mask.
    final.sort(key=lambda m: m["area"])          # smallest first
    pruned: list[dict] = []
    for m in final:
        if not pruned:
            pruned.append(m)
            continue
        union_smaller = np.zeros_like(m["segmentation"])
        for f in pruned:                          # all strictly smaller (kept) masks
            union_smaller |= f["segmentation"]
        covered = np.logical_and(m["segmentation"], union_smaller).sum() / m["area"]
        if covered <= 0.50:
            pruned.append(m)                      # genuinely new region
            continue
        remainder = np.logical_and(m["segmentation"], ~union_smaller)
        area = int(remainder.sum())
        if area >= min_area:
            pruned.append({**m, "segmentation": remainder, "area": area})
        # else: pure union mask — dropped
    final = pruned

    # 5. Decompose oversized masks: subtract the kept smaller masks; the
    #    remainder is the dominant component (or empty if it really was a
    #    whole-object union — then it disappears naturally).
    if oversized and final:
        union_small = np.zeros_like(fg)
        for f in final:
            union_small |= f["segmentation"]
        for m in sorted(oversized, key=lambda m: m["predicted_iou"], reverse=True):
            remainder = np.logical_and(m["segmentation"], ~union_small)
            area = int(remainder.sum())
            if area < min_area:
                continue
            cand = {**m, "segmentation": remainder, "area": area}
            is_dup = False
            for f in final:
                inter = np.logical_and(remainder, f["segmentation"]).sum()
                union = area + f["area"] - inter
                if union > 0 and inter / union > dedup_iou:
                    is_dup = True
                    break
            if not is_dup:
                final.append(cand)
    elif oversized and not final:
        # No small masks at all — keep the largest oversized mask rather
        # than returning nothing (single-component construction).
        final.append(max(oversized, key=lambda m: m["area"]))

    # 6. Split masks whose segmentation contains multiple disconnected blobs.
    #    A single mask covering two separate regions is almost certainly spanning
    #    two distinct components — split it into one candidate per blob.
    split: list[dict] = []
    for m in final:
        seg = m["segmentation"].astype(np.uint8)
        n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)
        large = [i for i in range(1, n_lbl) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        if len(large) <= 1:
            split.append(m)
        else:
            for i in large:
                blob_seg  = (lbl_map == i)
                blob_area = int(stats[i, cv2.CC_STAT_AREA])
                split.append({**m, "segmentation": blob_seg, "area": blob_area})
    final = split

    final.sort(key=lambda m: m["predicted_iou"], reverse=True)
    return final


def _run_sam_prompted(image_paths: list[Path], n_components: int,
                       output_dir: Path, weights_path: Path,
                       model_type: str = "vit_b", device: str = "cuda",
                       min_area: int = 80, max_area_frac: float = 0.50,
                       fg_overlap_min: float = 0.60, dedup_iou: float = 0.80,
                       keep_factor: int = 2) -> list[list[dict]]:
    """
    SAM in SamPredictor mode with tight-crop + black-bg + fg-grid prompts.

    For each image:
      1. Detect foreground (LCC) on the fixed-scale crop.
      2. Crop to the tight foreground bounding box — SAM encodes only the
         construction, eliminating wasted queries on white background/padding.
      3. Set background to pure black (not grey) for maximum contrast.
      4. CLAHE enhancement on the construction pixels.
      5. Place a regular grid of foreground-only points (density ~2.5 × N).
      6. SamPredictor: one encoder pass + M lightweight decoder passes.
      7. Expand masks back to fixed-scale image coordinates.
      8. Apply the standard filter pipeline (fg-overlap, border, dedup,
         container-decompose, blob-split) and keep top N×keep_factor.

    Amodal note: for rod classes with two exposed ends (occluded middle), the
    two separate SAM masks that result are co-axially merged in backprojector.py
    into a single 3D cloud.  The 3D bounding box of the merged cloud covers
    both ends — projecting it back to 2D per view gives the amodal 2D bbox
    for YOLO training.
    """
    keep_n     = n_components * keep_factor
    cache_file = output_dir / f"sam_masks_top{keep_n}_prompted_filt.pkl"
    if cache_file.exists():
        print(f"[SAM] Loading cached prompted masks from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"[SAM] Prompted mode: {len(image_paths)} images — "
          f"tight-crop + black-bg + fg-grid  (top-{keep_n}/image)")
    output_dir.mkdir(parents=True, exist_ok=True)

    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry[model_type](checkpoint=str(weights_path))
    sam.to(device)
    predictor = SamPredictor(sam)

    import time
    all_masks = []
    n_raw_total, n_filt_total = 0, 0

    for idx, img_path in enumerate(image_paths):
        t0 = time.time()
        image_bgr = cv2.imread(str(img_path))
        H_orig, W_orig = image_bgr.shape[:2]

        # Foreground mask on the fixed-scale crop (same logic as auto mode)
        fg = _foreground_mask(image_bgr)

        # Tight crop: construction fills the entire crop area
        crop_bgr, crop_fg, x_off, y_off = _tight_fg_crop(image_bgr, fg)

        # Pure black background — removes background ambiguity completely
        crop_bgr[~crop_fg] = 0

        # CLAHE contrast enhancement on construction pixels
        crop_bgr = _clahe_enhance(crop_bgr)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        # Foreground-only grid prompts in the crop coordinate system
        points = _fg_grid_points(crop_fg, n_components)

        if len(points) == 0:
            all_masks.append([])
            print(f"[SAM]   {idx+1}/{len(image_paths)}  no foreground — skipping")
            continue

        # One encoder pass for the whole image, N decoder passes (one per prompt)
        predictor.set_image(crop_rgb)
        crop_masks: list[dict] = []

        for (cx, cy) in points:
            input_point = np.array([[cx, cy]], dtype=np.float32)
            input_label = np.array([1])   # 1 = foreground
            masks, scores, _ = predictor.predict(
                point_coords     = input_point,
                point_labels     = input_label,
                multimask_output = True,   # 3 candidates; keep highest confidence
            )
            best = int(scores.argmax())
            crop_masks.append({
                "segmentation":  masks[best].astype(bool),
                "area":          int(masks[best].sum()),
                "predicted_iou": float(scores[best]),
                "image_idx":     idx,
            })

        t_gen = time.time() - t0
        n_raw_total += len(crop_masks)

        # Translate masks from crop space → fixed-scale image space
        full_masks = _expand_masks(crop_masks, H_orig, W_orig, x_off, y_off)

        # Standard filter pipeline (unchanged from auto mode)
        filtered = _filter_masks(full_masks, fg,
                                  max_area_frac=max_area_frac,
                                  fg_overlap_min=fg_overlap_min,
                                  dedup_iou=dedup_iou, min_area=min_area)
        n_filt_total += len(filtered)

        top_masks = filtered[:keep_n]
        all_masks.append(top_masks)

        print(f"[SAM]   {idx+1}/{len(image_paths)}  "
              f"prompts={len(points)}  filt={len(filtered)}  kept={len(top_masks)}"
              f"  t={t_gen:.1f}s", flush=True)

    n_kept = sum(len(m) for m in all_masks)
    print(f"[SAM] Prompted: {n_raw_total} raw → {n_filt_total} filtered → "
          f"{n_kept} kept (top-{keep_n}/image)")

    with open(cache_file, "wb") as f:
        pickle.dump(all_masks, f)
    print(f"[SAM] Cached to {cache_file}")
    return all_masks


def run_sam_on_images(image_paths: list[Path], n_components: int,
                      output_dir: Path, weights_path: Path,
                      model_type: str = "vit_b", device: str = "cuda",
                      points_per_side: int = 32, iou_thresh: float = 0.70,
                      stability_thresh: float = 0.85, min_area: int = 80,
                      max_area_frac: float = 0.50, fg_overlap_min: float = 0.60,
                      dedup_iou: float = 0.80, keep_factor: int = 2,
                      mode: str = "prompted") -> list[list[dict]]:
    """
    Run SAM on each image and return up to keep_factor×N masks per image.

    mode="prompted" (default): tight-crop + black-bg + fg-grid + SamPredictor.
    mode="auto":               SamAutomaticMaskGenerator (original approach).

    Returns list (one per image) of list of mask dicts:
        segmentation : (H, W) bool array  — in fixed-scale image coords
        area         : int
        predicted_iou: float
        image_idx    : int
    """
    if mode == "prompted":
        return _run_sam_prompted(
            image_paths, n_components, output_dir, weights_path,
            model_type=model_type, device=device,
            min_area=min_area, max_area_frac=max_area_frac,
            fg_overlap_min=fg_overlap_min, dedup_iou=dedup_iou,
            keep_factor=keep_factor,
        )

    # ── Auto mode (original) ──────────────────────────────────────────────────
    keep_n = n_components * keep_factor
    cache_file = output_dir / f"sam_masks_top{keep_n}_pts{points_per_side}_filt3.pkl"
    if cache_file.exists():
        print(f"[SAM] Loading cached masks from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"[SAM] Auto mode: {len(image_paths)} images "
          f"(filtered, top-{keep_n} per image)")
    output_dir.mkdir(parents=True, exist_ok=True)

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry[model_type](checkpoint=str(weights_path))
    sam.to(device)

    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=iou_thresh,
        stability_score_thresh=stability_thresh,
        min_mask_region_area=min_area,
    )

    import time
    all_masks = []
    n_raw_total, n_filt_total = 0, 0
    for idx, img_path in enumerate(image_paths):
        t0 = time.time()
        image_bgr = cv2.imread(str(img_path))
        fg        = _foreground_mask(image_bgr)           # on ORIGINAL image
        image_bgr[~fg] = 128                              # grey-fill background before CLAHE+SAM
        image_bgr = _clahe_enhance(image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        masks = mask_gen.generate(image_rgb)
        t_gen = time.time() - t0
        n_raw_total += len(masks)

        filtered = _filter_masks(masks, fg,
                                 max_area_frac=max_area_frac,
                                 fg_overlap_min=fg_overlap_min,
                                 dedup_iou=dedup_iou, min_area=min_area)
        n_filt_total += len(filtered)

        # Already sorted by predicted IoU (descending) inside _filter_masks
        top_masks = filtered[:keep_n]

        # Slim the dict — only keep what we need downstream
        slim = [{
            "segmentation":  m["segmentation"],
            "area":          m["area"],
            "predicted_iou": m["predicted_iou"],
            "image_idx":     idx,
        } for m in top_masks]

        all_masks.append(slim)
        print(f"[SAM]   {idx+1}/{len(image_paths)}  raw={len(masks)} "
              f"filt={len(filtered)} kept={len(slim)}  gen={t_gen:.1f}s",
              flush=True)

    n_kept = sum(len(m) for m in all_masks)
    print(f"[SAM] Mask filtering: {n_raw_total} raw → {n_filt_total} after "
          f"bg/subtract/dedup filters → {n_kept} kept (top-{keep_n}/image)")

    with open(cache_file, "wb") as f:
        pickle.dump(all_masks, f)
    print(f"[SAM] Cached masks to {cache_file}")

    return all_masks


if __name__ == "__main__":
    import sys
    from config import (MULTI_VIEW, IMAGE_ELEVATION, SAM_WEIGHTS, SAM_MODEL_TYPE,
                        SAM_POINTS_SIDE, SAM_IOU_THRESH, SAM_STABILITY,
                        SAM_MIN_AREA, SAM_MAX_AREA_FRAC, SAM_FG_OVERLAP_MIN,
                        SAM_DEDUP_IOU, DEVICE, OUTPUTS_ROOT)
    from puml_parser import find_puml, parse_puml

    sample_id  = int(sys.argv[1]) if len(sys.argv) > 1 else 113
    manifest   = parse_puml(find_puml(sample_id))
    img_dir    = MULTI_VIEW / f"Sample_{sample_id}" / IMAGE_ELEVATION
    img_paths  = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.JPG"))
    out_dir    = OUTPUTS_ROOT / f"sample_{sample_id}" / "sam"

    masks = run_sam_on_images(
        img_paths, manifest.n_components, out_dir,
        weights_path=SAM_WEIGHTS, model_type=SAM_MODEL_TYPE, device=DEVICE,
        points_per_side=SAM_POINTS_SIDE, iou_thresh=SAM_IOU_THRESH,
        stability_thresh=SAM_STABILITY, min_area=SAM_MIN_AREA,
        max_area_frac=SAM_MAX_AREA_FRAC, fg_overlap_min=SAM_FG_OVERLAP_MIN,
        dedup_iou=SAM_DEDUP_IOU,
    )
    print(f"Total images: {len(masks)}, masks per image: {[len(m) for m in masks]}")
