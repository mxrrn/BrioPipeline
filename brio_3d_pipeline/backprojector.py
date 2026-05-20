"""
Back-project SAM 2D masks → 3D instance point clouds using DUSt3R pts3d,
then cluster all per-view mask clouds into N instance groups.

Improvements over v1
────────────────────
* pts3d used directly — no manual K/pose/depth back-projection.
* Feature vector = [3D median centroid (3) + mean HSV colour (3)] with
  independent normalisation so colour and geometry contribute equally.
* Agglomerative clustering (Ward, n_clusters=N) — globally optimal.
* DBSCAN cleanup per cluster removes noisy outlier background points.
"""
import pickle
import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering, DBSCAN


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
    return pts[valid].astype(np.float32)


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
                    eps_frac: float = 0.08,
                    min_samples: int = 30) -> np.ndarray:
    """
    Keep only the largest dense cluster; discard outliers and background bleed.
    eps is computed as a fraction of the cluster's own bounding-box diameter.
    """
    if len(pts) < min_samples * 3:
        return pts
    diam = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    eps  = max(diam * eps_frac, 1e-5)
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(pts)
    good = labels >= 0
    if good.sum() == 0:
        return pts
    best = np.argmax(np.bincount(labels[good]))
    return pts[labels == best].astype(np.float32)


# ── Main API ──────────────────────────────────────────────────────────────────

def compute_proposals(dust3r_result: dict,
                      sam_masks_per_image: list[list[dict]],
                      n_components: int,
                      output_dir: Path,
                      image_paths: list[Path] | None = None) -> list[np.ndarray]:
    """
    SAM masks × DUSt3R pts3d → N merged + cleaned 3D instance point clouds.

    Args:
        dust3r_result      : output of dust3r_runner.run_dust3r()
        sam_masks_per_image: output of sam_runner.run_sam_on_images()
        n_components       : N from PUML manifest
        output_dir         : cache directory
        image_paths        : original (cropped) image paths for colour extraction

    Returns list of N (M_i, 3) world-space float32 arrays.
    """
    cache_file = output_dir / f"proposals_N{n_components}_v2.pkl"
    if cache_file.exists():
        print(f"[Backproject] Loading cached proposals from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    pts3d_all = dust3r_result["pts3d"]   # list of (H, W, 3)
    use_colour = image_paths is not None
    print(f"[Backproject] {len(sam_masks_per_image)} views × top-{n_components} masks"
          f"  colour={'yes' if use_colour else 'no'}")

    all_clouds    : list[np.ndarray]  = []
    all_centroids : list[np.ndarray]  = []   # 3D centroid (3,)
    all_colours   : list[np.ndarray]  = []   # mean HSV   (3,)

    for img_idx, (masks, p3d) in enumerate(zip(sam_masks_per_image, pts3d_all)):
        # Load image for colour extraction (once per image)
        img_rgb = None
        if use_colour and img_idx < len(image_paths):
            bgr = cv2.imread(str(image_paths[img_idx]))
            if bgr is not None:
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        for mask_dict in masks:
            seg = mask_dict["segmentation"]
            pts = _extract_mask_cloud(p3d, seg)
            if len(pts) < 20:
                continue

            all_clouds.append(pts)
            all_centroids.append(_median_centroid(pts))
            colour = _mean_hsv(img_rgb, seg) if img_rgb is not None else np.zeros(3, np.float32)
            all_colours.append(colour)

    if len(all_clouds) < n_components:
        print(f"[Backproject] WARNING: only {len(all_clouds)} valid clouds for "
              f"{n_components} components — padding")
        while len(all_clouds) < n_components:
            all_clouds.append(np.empty((0, 3), np.float32))
            all_centroids.append(np.zeros(3, np.float32))
            all_colours.append(np.zeros(3, np.float32))

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

    # ── Merge + DBSCAN cleanup ────────────────────────────────────────────
    merged = []
    for cid in range(n_components):
        idx    = np.where(labels == cid)[0]
        raw    = np.concatenate([all_clouds[i] for i in idx], axis=0) \
                 if len(idx) else np.empty((0, 3), np.float32)
        clean  = _dbscan_cleanup(raw) if len(raw) >= 90 else raw
        merged.append(clean)

    print(f"[Backproject] Results after DBSCAN cleanup:")
    for i, pts in enumerate(merged):
        c = _median_centroid(pts)
        print(f"  Cluster {i}: {len(pts):>7d} pts | "
              f"centroid ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})")

    with open(cache_file, "wb") as f:
        pickle.dump(merged, f)
    print(f"[Backproject] Cached to {cache_file}")
    return merged
