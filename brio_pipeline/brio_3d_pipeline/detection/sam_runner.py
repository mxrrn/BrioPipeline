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


def _fg_grid_points(fg_mask: np.ndarray, spacing: int,
                    dt_seeds: bool = True,
                    bgr: np.ndarray | None = None,
                    return_source: bool = False,
                    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Place a fixed-spacing regular grid of points inside the foreground mask,
    optionally supplemented by edge-aware distance-transform seeds.

    `spacing` is a fixed pixel distance, not derived from component count or
    foreground area. The previous version computed spacing as
    sqrt(fg_area / (n_components * density)) — i.e. "spread ~N points evenly
    over the whole foreground" — which has no relationship to how big any
    individual component is. A large strap covering 30% of the foreground
    would catch several points by sheer area; a small nut/screw covering
    0.5% of the foreground had roughly a 0.5% chance per point of being hit
    at all, so it was a lottery, not a guarantee, despite the old docstring's
    claim. There's no way to truly guarantee one prompt per *semantic*
    component without already knowing where components are (that's what
    we're segmenting for) — the honest fix is a small fixed spacing, dense
    enough that no real fastener-sized object can fit entirely between grid
    points, regardless of how many components the sample has or how large
    the foreground is.

    dt_seeds adds one prompt near the centre of each *edge-bounded region*:
    visible image edges (Canny on `bgr`) are subtracted from the foreground,
    which splits the silhouette into regions roughly corresponding to
    individual visible part surfaces (a screw head bounded by its circular
    outline, a strap segment between joints); the distance-transform maxima
    of those regions are their medial axes / centres. Verified necessary in
    exactly this form: running the DT on the raw foreground instead (no edge
    subtraction) yields the medial axis of the whole connected silhouette —
    dominated by the largest part — and produced zero seeds on any fastener
    in a real test view, because a part sitting ON another part contributes
    nothing to the silhouette. A well-centred prompt matters beyond mere
    coverage, too: a grid point that lands near a part's boundary is exactly
    where SAM is most likely to segment the neighbouring part instead.
    When `bgr` is None, falls back to the (weak) whole-foreground DT.

    Returns (M, 2) float32 array of (x, y) coordinates. If `return_source`
    is True, instead returns (points, is_dt_seed) where is_dt_seed is a
    (M,) bool array — True for edge-aware DT seeds, False for uniform-grid
    points (debugging/visualisation only; the default caller doesn't need
    per-point provenance, so this is opt-in and doesn't change the default
    return type).
    """
    def _out(pts: np.ndarray, is_dt: np.ndarray):
        return (pts, is_dt) if return_source else pts

    ys, xs = np.where(fg_mask)
    if len(ys) == 0:
        empty = np.empty((0, 2), np.float32)
        return _out(empty, np.empty((0,), bool))

    H, W = fg_mask.shape
    gx = np.arange(spacing // 2, W, spacing)
    gy = np.arange(spacing // 2, H, spacing)
    grid_x, grid_y = np.meshgrid(gx, gy)
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(int)

    # Keep only grid points that fall inside the foreground
    valid_x = np.clip(pts[:, 0], 0, W - 1)
    valid_y = np.clip(pts[:, 1], 0, H - 1)
    inside  = fg_mask[valid_y, valid_x]
    grid_pts = pts[inside]

    if not dt_seeds:
        return _out(grid_pts.astype(np.float32),
                    np.zeros(len(grid_pts), bool))

    # Edge-bounded regions: eroded foreground minus dilated Canny edges.
    # The erosion kills the thin ribbon between the foreground-mask boundary
    # and the part's own outline edge — without it, that ribbon's ridge line
    # produced dozens of useless seeds strung along the silhouette (observed
    # directly on a test view before adding the erosion + width floor).
    region = fg_mask
    if bgr is not None:
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        fg_core = cv2.erode(fg_mask.astype(np.uint8),
                            np.ones((5, 5), np.uint8)).astype(bool)
        region = np.logical_and(fg_core, edges == 0)

    # One seed per connected edge-bounded region: the position of that
    # region's own distance-transform maximum (its widest interior point).
    # Per-region argmax rather than windowed non-max suppression — a
    # windowed comparison suppresses a small region's maximum whenever a
    # taller maximum of a large neighbouring region falls inside the
    # window (observed: screw heads next to a wide plate got no seed).
    # dist >= 3.0 requires a region at least ~6px wide — a real part
    # surface qualifies, edge slivers and wood-grain fragments don't.
    from scipy import ndimage
    dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    n_lbl, lbl = cv2.connectedComponents(region.astype(np.uint8),
                                         connectivity=8)
    dt_pts = np.empty((0, 2), int)
    if n_lbl > 1:
        idx      = np.arange(1, n_lbl)
        max_vals = ndimage.maximum(dist, labels=lbl, index=idx)
        keep     = np.asarray(max_vals) >= 3.0
        if keep.any():
            positions = ndimage.maximum_position(dist, labels=lbl,
                                                 index=idx[keep])
            dt_pts = np.array([(x, y) for (y, x) in positions], int)

    # Merge, deduplicating points that fall in the same 4px cell.
    # DT seeds listed first so they win the dedup (np.unique keeps the
    # first occurrence) — a region-centre point is a strictly better prompt
    # than a grid point 2px away from it.
    all_pts = np.vstack([dt_pts, grid_pts])
    is_dt_all = np.concatenate([np.ones(len(dt_pts), bool),
                                np.zeros(len(grid_pts), bool)])
    cells = all_pts // 4
    _, keep_idx = np.unique(cells, axis=0, return_index=True)
    order = np.sort(keep_idx)
    return _out(all_pts[order].astype(np.float32), is_dt_all[order])


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
    #
    #    A dense grid can tile almost the entire surface of a single large
    #    flat component with dozens of small, individually-under-threshold
    #    sub-part masks — verified directly: the union of just the "normal"
    #    (non-oversized) masks covered 99.3% of the foreground on one view,
    #    all belonging to what is physically one plate. Testing "did
    #    decomposition produce a remainder ≥ min_area" is the wrong check
    #    here — most oversized candidates for that plate still cleared it
    #    with a ~130-150px edge sliver, which is not a decomposition
    #    success, it just means min_area is small enough that SOME leftover
    #    edge pixel always exists. So don't gate on decomposition succeeding
    #    at all: always additionally keep the single best-confidence
    #    oversized mask undecomposed, alongside whatever real small
    #    remainders survive. The large component then always has one
    #    coherent whole-object candidate to fall back on; downstream dedup
    #    and 3D clustering — which have more context than a single 2D
    #    view — decide what to do with the redundancy.
    if oversized:
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

        best_whole = max(oversized, key=lambda m: m["predicted_iou"])
        is_dup = False
        for f in final:
            inter = np.logical_and(best_whole["segmentation"], f["segmentation"]).sum()
            union = best_whole["area"] + f["area"] - inter
            if union > 0 and inter / union > dedup_iou:
                is_dup = True
                break
        if not is_dup:
            final.append(best_whole)

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


# Area bands for the keep policy, as fractions of the foreground area.
# small = fastener scale (nuts/screw heads), large = whole-object candidates.
_BAND_SMALL_FRAC = 0.02
_BAND_LARGE_FRAC = 0.35   # matches SAM_MAX_AREA_FRAC's "oversized" notion
# Slot allocation across (small, mid, large) as fractions of keep_n.
_BAND_SLOTS = (0.5, 0.3, 0.2)


def _select_masks_banded(masks: list[dict], fg: np.ndarray,
                         keep_n: int) -> list[dict]:
    """
    Area-banded, coverage-greedy top-N selection.

    Replaces the plain IoU-sorted `filtered[:keep_n]` cut. Measured on
    run_030 (sample 114): nearly all filtered masks score predicted_iou
    ≈0.96 regardless of size, so an IoU sort makes the cut effectively
    random among equal scorers — which small masks survive then varies
    arbitrarily view to view, undermining the cross-view consistency the
    3D grouping depends on. (The intuitive alternative hypothesis — that
    IoU sorting is *biased against* small masks — was tested on the same
    data and is false: small kept masks scored the same 0.959 mean as
    large ones.)

    Policy:
      * Split candidates into three area bands (fastener-scale /
        component-scale / whole-object-scale, relative to foreground area)
        with fixed slot allocations, so small masks can never be crowded
        out by an abundance of equally-scoring mid-size masks.
      * Within a band, select greedily by most newly-covered foreground
        (IoU order as the tiebreak, since candidates arrive IoU-sorted).
        Within one band containment is rare, so coverage-greedy spreads
        the kept masks spatially instead of stacking near-duplicates.
        (Coverage-greedy across *all* masks at once would not work: the
        whole-object mask covers everything, so after picking it every
        small mask adds zero new coverage.)
      * Unused slots from an under-populated band spill to the others.
    """
    if len(masks) <= keep_n:
        return masks

    fg_area = max(int(fg.sum()), 1)
    bands: tuple[list[dict], ...] = ([], [], [])
    for m in masks:   # masks arrive sorted by predicted_iou desc
        frac = m["area"] / fg_area
        if frac < _BAND_SMALL_FRAC:
            bands[0].append(m)
        elif frac < _BAND_LARGE_FRAC:
            bands[1].append(m)
        else:
            bands[2].append(m)

    slots = [int(round(f * keep_n)) for f in _BAND_SLOTS]
    slots[0] += keep_n - sum(slots)   # rounding remainder → small band

    # Spill unused slots (band has fewer masks than slots) to other bands
    for i in range(3):
        spare = slots[i] - len(bands[i])
        if spare > 0:
            slots[i] = len(bands[i])
            j = (i + 1) % 3
            slots[j] += spare

    selected: list[dict] = []
    for band, n_slots in zip(bands, slots):
        n_slots = min(n_slots, len(band))
        if n_slots <= 0:
            continue
        covered = np.zeros_like(fg, bool)
        remaining = list(band)
        while remaining and n_slots > 0:
            best_i, best_gain = 0, -1
            for i, m in enumerate(remaining):
                gain = int(np.logical_and(m["segmentation"], ~covered).sum())
                if gain > best_gain:
                    best_gain, best_i = gain, i
            m = remaining.pop(best_i)
            selected.append(m)
            covered |= m["segmentation"]
            n_slots -= 1

    selected.sort(key=lambda m: m["predicted_iou"], reverse=True)
    return selected


def _run_sam_prompted(image_paths: list[Path], n_components: int,
                       output_dir: Path, weights_path: Path,
                       model_type: str = "vit_b", device: str = "cuda",
                       min_area: int = 80, max_area_frac: float = 0.50,
                       fg_overlap_min: float = 0.60, dedup_iou: float = 0.80,
                       keep_factor: int = 2, grid_spacing: int = 15,
                       dt_seeds: bool = True) -> list[list[dict]]:
    """
    SAM in SamPredictor mode with tight-crop + black-bg + fg-grid prompts.

    For each image:
      1. Detect foreground (LCC) on the fixed-scale crop.
      2. Crop to the tight foreground bounding box — SAM encodes only the
         construction, eliminating wasted queries on white background/padding.
      3. Set background to pure black (not grey) for maximum contrast.
      4. CLAHE enhancement on the construction pixels.
      5. Place a fixed-spacing grid of foreground-only points (see
         _fg_grid_points — spacing is a fixed pixel value, not derived from
         component count, so small components can't fall in the gaps).
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
    cache_file = output_dir / (
        f"sam_masks_top{keep_n}_prompted_sp{grid_spacing}_dt{int(dt_seeds)}"
        f"_ma{min_area}_mf{max_area_frac}_allgran_v4_filt.pkl"
    )
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

        # Foreground-only grid prompts in the crop coordinate system.
        # crop_bgr here is the CLAHE-enhanced black-background version —
        # good input for the edge detector inside (boosted local contrast).
        points = _fg_grid_points(crop_fg, grid_spacing, dt_seeds=dt_seeds,
                                  bgr=crop_bgr)

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
                multimask_output = True,   # whole/part/subpart granularities
            )
            # Collapsing to argmax(score) here was silently discarding the
            # subpart hypothesis: SAM's own confidence is biased toward large,
            # clean, well-bounded regions, so a point clicked on a small
            # embedded fastener (nut/screw) reliably scored "the plate it's
            # attached to" higher than "just the fastener" — meaning a
            # correct small-part mask was thrown away before it ever reached
            # the filter pipeline, on every single prompt. Commented out in
            # favor of keeping all 3 granularities as separate candidates;
            # the filter pipeline's dedup/container/decompose passes already
            # handle the resulting duplicates.
            # best = int(scores.argmax())
            # crop_masks.append({
            #     "segmentation":  masks[best].astype(bool),
            #     "area":          int(masks[best].sum()),
            #     "predicted_iou": float(scores[best]),
            #     "image_idx":     idx,
            # })
            for m, s in zip(masks, scores):
                crop_masks.append({
                    "segmentation":  m.astype(bool),
                    "area":          int(m.sum()),
                    "predicted_iou": float(s),
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

        top_masks = _select_masks_banded(filtered, fg, keep_n)
        all_masks.append(top_masks)

        # Foreground-coverage metric: fraction of foreground pixels inside
        # the union of kept masks. Direct per-view proxy for "did every
        # component get a mask" — uncovered foreground is a missed part.
        cov_union = np.zeros_like(fg, bool)
        for m in top_masks:
            cov_union |= m["segmentation"]
        cov = np.logical_and(cov_union, fg).sum() / max(int(fg.sum()), 1)

        print(f"[SAM]   {idx+1}/{len(image_paths)}  "
              f"prompts={len(points)}  filt={len(filtered)}  kept={len(top_masks)}"
              f"  cov={cov:.2f}  t={t_gen:.1f}s", flush=True)

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
                      mode: str = "prompted", grid_spacing: int = 15,
                      dt_seeds: bool = True) -> list[list[dict]]:
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
            keep_factor=keep_factor, grid_spacing=grid_spacing,
            dt_seeds=dt_seeds,
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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import (MULTI_VIEW, IMAGE_ELEVATION, SAM_WEIGHTS, SAM_MODEL_TYPE,
                        SAM_POINTS_SIDE, SAM_IOU_THRESH, SAM_STABILITY,
                        SAM_MIN_AREA, SAM_MAX_AREA_FRAC, SAM_FG_OVERLAP_MIN,
                        SAM_DEDUP_IOU, SAM_GRID_SPACING, SAM_DT_SEEDS,
                        DEVICE, OUTPUTS_ROOT)
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
        dedup_iou=SAM_DEDUP_IOU, grid_spacing=SAM_GRID_SPACING,
        dt_seeds=SAM_DT_SEEDS,
    )
    print(f"Total images: {len(masks)}, masks per image: {[len(m) for m in masks]}")
