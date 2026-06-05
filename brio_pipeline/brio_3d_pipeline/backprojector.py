"""
Back-project SAM 2D masks → 3D instance point clouds using DUSt3R pts3d,
then cluster all per-view mask clouds into N instance groups.

Improvements over v1
────────────────────
* pts3d used directly — no manual K/pose/depth back-projection.
* Feature vector = [3D median centroid (3) + mean HSV colour (3)] with
  independent normalisation so colour and geometry contribute equally.
* Agglomerative clustering (Ward, n_clusters=N) — globally optimal.
* DBSCAN cleanup per cluster: adaptive eps from bbox diagonal, discards
  isolated noise points without assuming a Gaussian distribution.
* Optional ComponentClassifier: predicts the class of each SAM mask crop,
  then majority-votes per cluster to produce a visual class label.
  All crops are batched into a single GPU forward pass for speed.
"""
import pickle
import numpy as np
import cv2
from pathlib import Path
from collections import Counter
from sklearn.cluster import AgglomerativeClustering, DBSCAN

# Same CLAHE settings as sam_runner — keeps classifier crops visually consistent
# with what SAM saw when generating the masks.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _clahe_enhance(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

_CLOUD_MAX_PTS = 5_000   # cap per-mask cloud to bound downstream memory


# ── Point cloud helpers ───────────────────────────────────────────────────────

def _extract_mask_cloud(pts3d_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """World-space 3D points under mask from one DUSt3R frame (H×W×3)."""
    if mask.shape != pts3d_frame.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (pts3d_frame.shape[1], pts3d_frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    pts   = pts3d_frame[mask]
    valid = np.linalg.norm(pts, axis=1) > 1e-6
    pts   = pts[valid].astype(np.float32)
    if len(pts) > _CLOUD_MAX_PTS:
        idx = np.random.choice(len(pts), _CLOUD_MAX_PTS, replace=False)
        pts = pts[idx]
    return pts


def _median_centroid(pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.array([np.inf, np.inf, np.inf], np.float32)
    return np.median(pts, axis=0).astype(np.float32)


def _mean_hsv(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean HSV of mask pixels, normalised to [0,1]. Returns zeros if empty."""
    if mask.shape != img_rgb.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    pixels = img_rgb[mask]
    if len(pixels) == 0:
        return np.zeros(3, np.float32)
    hsv  = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0]
    mean = hsv.mean(axis=0)
    return (mean / np.array([180., 255., 255.])).astype(np.float32)


def _dbscan_cleanup(pts: np.ndarray,
                    eps_frac: float = 0.05,
                    min_samples: int = 5) -> np.ndarray:
    """
    Remove outlier points using DBSCAN: keep only the largest dense cluster.

    eps is set adaptively as eps_frac × bounding-box diagonal so the threshold
    auto-scales with the physical size of the construction, removing the need to
    tune an absolute distance.  Points labelled -1 (noise) are discarded.
    If DBSCAN marks everything as noise the original cloud is returned unchanged.
    """
    if len(pts) < min_samples * 2:
        return pts
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    eps  = max(diag * eps_frac, 1e-4)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    core_mask = labels != -1
    if not core_mask.any():
        return pts  # all labelled noise — nothing to remove
    unique, counts = np.unique(labels[core_mask], return_counts=True)
    main_label = unique[counts.argmax()]
    kept = pts[labels == main_label].astype(np.float32)
    return kept if len(kept) >= min_samples else pts


# ── Main API ──────────────────────────────────────────────────────────────────

def compute_proposals(dust3r_result: dict,
                      sam_masks_per_image: list[list[dict]],
                      n_components: int,
                      output_dir: Path,
                      image_paths: list[Path] | None = None,
                      classifier=None,
                      conf_thresh: float = 0.40,
                      ) -> tuple[list[np.ndarray], list[str | None], list[np.ndarray]]:
    """
    SAM masks × DUSt3R pts3d → N merged + cleaned 3D instance point clouds.

    Args:
        dust3r_result      : output of dust3r_runner.run_dust3r()
        sam_masks_per_image: output of sam_runner.run_sam_on_images()
        n_components       : N from PUML manifest
        output_dir         : cache directory
        image_paths        : original (cropped) image paths for colour extraction
        classifier         : optional ComponentClassifier for visual voting
        conf_thresh        : minimum confidence to accept a visual prediction

    Returns (clouds, visual_cls, cluster_colours):
        clouds          : list of N (M_i, 3) world-space float32 arrays
        visual_cls      : list of N dominant visual class codes (or None per cluster)
        cluster_colours : list of N mean HSV (3,) float32 arrays in [0,1]
    """
    use_visual = classifier is not None
    cache_suffix = "_vis" if use_visual else ""
    cache_file = output_dir / f"proposals_N{n_components}_v5{cache_suffix}.pkl"
    if cache_file.exists():
        print(f"[Backproject] Loading cached proposals from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    pts3d_all = dust3r_result["pts3d"]   # list of (H, W, 3)
    use_colour = image_paths is not None
    print(f"[Backproject] {len(sam_masks_per_image)} views × top-{n_components} masks  "
          f"colour={'yes' if use_colour else 'no'}  visual={'yes' if use_visual else 'no'}")

    from component_classifier import crop_from_mask

    all_clouds    : list[np.ndarray]  = []
    all_centroids : list[np.ndarray]  = []   # 3D centroid (3,)
    all_colours   : list[np.ndarray]  = []   # mean HSV   (3,)
    all_visual_cls: list[str | None]  = []   # per-mask visual prediction (filled after batch)

    # Crop collection for batched classifier inference
    pending_crops : list[np.ndarray] = []   # BGR crops awaiting prediction
    pending_idx   : list[int]        = []   # index into all_clouds for each crop

    for img_idx, (masks, p3d) in enumerate(zip(sam_masks_per_image, pts3d_all)):
        # Load image for colour and visual extraction (once per image).
        bgr, img_rgb = None, None
        if use_colour and img_idx < len(image_paths):
            bgr = cv2.imread(str(image_paths[img_idx]))
            if bgr is not None:
                bgr = _clahe_enhance(bgr)
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        for mask_dict in masks:
            seg = mask_dict["segmentation"]
            pts = _extract_mask_cloud(p3d, seg)
            if len(pts) < 20:
                continue

            cloud_idx = len(all_clouds)
            all_clouds.append(pts)
            all_centroids.append(_median_centroid(pts))
            colour = _mean_hsv(img_rgb, seg) if img_rgb is not None else np.zeros(3, np.float32)
            all_colours.append(colour)
            all_visual_cls.append(None)  # placeholder — filled below

            # Collect crop for batched inference.
            # Skip if crop too small (<32 px shortest side).
            if use_visual and bgr is not None:
                crop = crop_from_mask(bgr, seg)
                if crop is not None and min(crop.shape[:2]) >= 32:
                    pending_crops.append(crop)
                    pending_idx.append(cloud_idx)

    # ── Batched visual inference ──────────────────────────────────────────
    if use_visual and pending_crops:
        print(f"[Backproject] Running classifier on {len(pending_crops)} crops (batched)")
        batch_preds = classifier.predict_batch(pending_crops)
        for cloud_idx, (pred_cls, pred_conf) in zip(pending_idx, batch_preds):
            if pred_conf >= conf_thresh:
                all_visual_cls[cloud_idx] = pred_cls

    if len(all_clouds) < n_components:
        print(f"[Backproject] WARNING: only {len(all_clouds)} valid clouds for "
              f"{n_components} components — padding")
        while len(all_clouds) < n_components:
            all_clouds.append(np.empty((0, 3), np.float32))
            all_centroids.append(np.zeros(3, np.float32))
            all_colours.append(np.zeros(3, np.float32))
            all_visual_cls.append(None)

    centroids_arr = np.array(all_centroids, np.float32)   # (K, 3)
    colours_arr   = np.array(all_colours,   np.float32)   # (K, 3)

    # ── Normalise each feature block to unit std ──────────────────────────
    c_std = centroids_arr.std(axis=0).clip(1e-6)
    col_std = colours_arr.std(axis=0).clip(1e-6)

    feat = np.concatenate([
        centroids_arr / c_std,
        colours_arr   / col_std,
    ], axis=1)   # (K, 6)

    print(f"[Backproject] Clustering {len(all_clouds)} clouds into {n_components} groups "
          f"using geometry + colour features")

    # ── Agglomerative clustering ──────────────────────────────────────────
    labels = AgglomerativeClustering(
        n_clusters=n_components, metric="euclidean", linkage="ward"
    ).fit_predict(feat)

    # ── Merge + DBSCAN cleanup + visual majority vote ─────────────────────
    merged: list[np.ndarray] = []
    visual_cls_per_cluster: list[str | None] = []
    cluster_colours: list[np.ndarray] = []

    for cid in range(n_components):
        idx   = np.where(labels == cid)[0]
        raw   = np.concatenate([all_clouds[i] for i in idx], axis=0) \
                if len(idx) else np.empty((0, 3), np.float32)
        clean = _dbscan_cleanup(raw)
        merged.append(clean)

        # Per-cluster mean observed HSV (used by classifier for colour matching)
        if len(idx):
            cluster_colours.append(colours_arr[idx].mean(axis=0))
        else:
            cluster_colours.append(np.zeros(3, np.float32))

        # Majority visual vote for this cluster
        votes = [all_visual_cls[i] for i in idx if all_visual_cls[i] is not None]
        dominant = Counter(votes).most_common(1)[0][0] if votes else None
        visual_cls_per_cluster.append(dominant)

    print(f"[Backproject] Results after DBSCAN cleanup:")
    for i, (pts, vcls) in enumerate(zip(merged, visual_cls_per_cluster)):
        c = _median_centroid(pts)
        print(f"  Cluster {i}: {len(pts):>7d} pts | "
              f"centroid ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})"
              + (f"  visual={vcls}" if vcls else ""))

    result = (merged, visual_cls_per_cluster, cluster_colours)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"[Backproject] Cached to {cache_file}")
    return result
