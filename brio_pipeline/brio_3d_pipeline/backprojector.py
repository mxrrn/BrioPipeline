"""
Back-project SAM 2D masks → 3D instance point clouds using DUSt3R pts3d,
then group all per-view mask clouds into N instance groups.

Grouping (v7)
─────────────
* pts3d used directly — no manual K/pose/depth back-projection; points below
  the DUSt3R confidence threshold are dropped.
* 3D-overlap graph: two masks (from any views) belong to the same instance
  iff their 3D point clouds occupy the same voxels.  Union-find over the
  overlap graph yields the groups; groups are merged down / padded to N.
  This replaces Ward clustering on [centroid, HSV] features, which collapsed
  whenever masks of different parts were closer in feature space than the
  same part across views.
* DBSCAN cleanup per group: adaptive eps from bbox diagonal, discards
  isolated noise points without assuming a Gaussian distribution.
* Geometric co-axial merge re-joins rod fragments split by occlusion.
* Optional ComponentClassifier: predicts the class of each SAM mask crop,
  then majority-votes per group to produce a visual class label.
"""
import pickle
import numpy as np
import cv2
from pathlib import Path
from collections import Counter
from sklearn.cluster import DBSCAN

# Same CLAHE settings as sam_runner — keeps classifier crops visually consistent
# with what SAM saw when generating the masks.
try:
    from config import CLAHE_CLIP_LIMIT as _CLAHE_CLIP_LIMIT
except ImportError:
    _CLAHE_CLIP_LIMIT = 3.0
_CLAHE = cv2.createCLAHE(clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))

def _clahe_enhance(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

_CLOUD_MAX_PTS = 5_000   # cap per-mask cloud to bound downstream memory


# ── Point cloud helpers ───────────────────────────────────────────────────────

def _extract_mask_cloud(pts3d_frame: np.ndarray, mask: np.ndarray,
                        conf: np.ndarray | None = None,
                        conf_thresh: float = 0.0) -> np.ndarray:
    """World-space 3D points under mask from one DUSt3R frame (H×W×3).

    If a DUSt3R confidence map is given, pixels below `conf_thresh` are
    dropped — these are typically textureless background or misaligned
    regions that produce floating outlier points.
    """
    if mask.shape != pts3d_frame.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (pts3d_frame.shape[1], pts3d_frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    if conf is not None:
        mask = np.logical_and(mask, conf >= conf_thresh)
    pts   = pts3d_frame[mask]
    valid = np.linalg.norm(pts, axis=1) > 1e-6
    pts   = pts[valid].astype(np.float32)
    if len(pts) > _CLOUD_MAX_PTS:
        idx = np.random.choice(len(pts), _CLOUD_MAX_PTS, replace=False)
        pts = pts[idx]
    return pts


def _compute_elongation(pts: np.ndarray) -> float:
    """PCA elongation σ₁/σ₂ of a point cloud (≥1; 1 = isotropic, high = rod-like)."""
    if len(pts) < 10:
        return 1.0
    centered = pts - pts.mean(0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    return float(s[0] / max(s[1], 1e-9))


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


_DBSCAN_MAX_PTS = 20_000   # subsample cap to keep DBSCAN memory O(n), not O(n²)


def _dbscan_cleanup(pts: np.ndarray,
                    eps_frac: float = 0.05,
                    min_samples: int = 5) -> np.ndarray:
    """
    Remove outlier points using DBSCAN: keep only the largest dense cluster.

    eps = eps_frac × bounding-box diagonal (auto-scales with construction size).
    If the merged cloud is large, DBSCAN runs on a random subsample and the
    resulting main-cluster centroid + radius is used to filter the full cloud —
    keeping memory O(n) rather than O(n²) for dense neighbourhoods.
    """
    if len(pts) < min_samples * 2:
        return pts

    # Subsample for DBSCAN to bound neighbour-list memory.
    # Fixed seed ensures the same subsample on every run (reproducibility).
    if len(pts) > _DBSCAN_MAX_PTS:
        rng     = np.random.default_rng(seed=42)
        sub_idx = rng.choice(len(pts), _DBSCAN_MAX_PTS, replace=False)
        sub = pts[sub_idx]
    else:
        sub_idx = None
        sub = pts

    diag = float(np.linalg.norm(sub.max(axis=0) - sub.min(axis=0)))
    eps  = max(diag * eps_frac, 1e-4)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(sub)

    core_mask = labels != -1
    if not core_mask.any():
        return pts  # all noise — return original

    unique, counts = np.unique(labels[core_mask], return_counts=True)
    main_label = unique[counts.argmax()]
    main_sub   = sub[labels == main_label]

    if sub_idx is None:
        # Small cloud — labels apply directly
        kept = pts[labels == main_label].astype(np.float32)
    else:
        # Large cloud — use the subsample's main cluster to define a radius
        # filter on the full cloud so outlier removal generalises beyond the sub.
        centre = np.median(main_sub, axis=0)
        dists  = np.linalg.norm(pts - centre, axis=1)
        radius = np.percentile(np.linalg.norm(main_sub - centre, axis=1), 95) * 2.0
        kept   = pts[dists <= radius].astype(np.float32)

    return kept if len(kept) >= min_samples else pts


def _overlap_group(clouds: list[np.ndarray], n_target: int,
                   overlap_min: float = 0.20, voxel_div: int = 40,
                   colours: np.ndarray | None = None,
                   color_weight: float = 0.0) -> np.ndarray:
    """
    Group per-view mask clouds by 3D voxel overlap.

    Pairwise affinity:
        overlap_ij = |voxels_i ∩ voxels_j| / min(|voxels_i|, |voxels_j|)
    (clipped to 0 below overlap_min).  Groups come from AVERAGE-LINKAGE
    agglomerative clustering on distance 1 − overlap with n_clusters = N.
    Average linkage is essential: connected components / single linkage
    chain through shared boundary voxels and collapse the whole assembly
    into one blob (observed on sample 113: 200 clouds → 2 components).

    Returns an int label array (len == len(clouds), values in [0, n_target)).
    """
    from sklearn.cluster import AgglomerativeClustering

    K = len(clouds)
    labels = np.zeros(K, np.int64)
    valid = [i for i in range(K) if len(clouds[i]) > 0]
    if not valid:
        return labels

    all_pts = np.concatenate([clouds[i] for i in valid], axis=0)
    diag  = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)))
    voxel = max(diag / voxel_div, 1e-5)

    keys: list[set] = [
        set(map(tuple, np.floor(c / voxel).astype(np.int64))) if len(c) else set()
        for c in clouds
    ]

    # ── Pairwise overlap matrix among valid clouds ────────────────────────
    nv = len(valid)
    overlap = np.zeros((nv, nv), np.float32)
    for a in range(nv):
        ka = keys[valid[a]]
        for b in range(a + 1, nv):
            kb = keys[valid[b]]
            inter = len(ka & kb)
            if inter:
                ov = inter / min(len(ka), len(kb))
                if ov >= overlap_min:
                    overlap[a, b] = overlap[b, a] = ov

    n_groups = min(n_target, nv)
    dist = 1.0 - overlap
    np.fill_diagonal(dist, 0.0)

    # Optional HSV colour repulsion: penalises merging differently-coloured clouds.
    # L2 in normalised HSV ∈ [0,1]³; max distance = sqrt(3) ≈ 1.732.
    if colours is not None and color_weight > 0 and nv > 1:
        hsv_v = colours[np.array(valid)]   # (nv, 3)
        for a in range(nv):
            for b in range(a + 1, nv):
                cd = float(np.linalg.norm(hsv_v[a] - hsv_v[b])) / 1.732
                dist[a, b] += color_weight * cd
                dist[b, a] += color_weight * cd

    lab_valid = AgglomerativeClustering(
        n_clusters=n_groups, metric="precomputed", linkage="average"
    ).fit_predict(dist)

    # Stable ordering: biggest group (by points) first
    sizes = np.zeros(n_groups)
    for a, i in enumerate(valid):
        sizes[lab_valid[a]] += len(clouds[i])
    order = np.argsort(-sizes)
    remap = {int(old): new for new, old in enumerate(order)}
    for a, i in enumerate(valid):
        labels[i] = remap[int(lab_valid[a])]

    linked = int((overlap > 0).sum() // 2)
    print(f"[Backproject] Overlap clustering: {nv} clouds, {linked} links "
          f"→ {n_groups} groups (avg-linkage, voxel={voxel*1000:.1f} mm)")
    return labels


def _coaxial_merge(
    clouds      : list[np.ndarray],
    visual_cls  : list[str | None],
    max_gap_m   : float,
    angle_deg   : float,
    max_offset_m: float = 0.012,
    min_elong   : float = 2.5,
) -> tuple[list[np.ndarray], list[str | None]]:
    """
    Merge pairs of clusters that are co-axial fragments of the same rod.

    A rod passing through a connector is occluded in the middle; the two
    visible ends reconstruct as disjoint lobes.  Purely geometric test —
    two clusters are fragments of one rod iff:
      1. both are elongated (PCA σ1/σ2 > min_elong) — only rods qualify
      2. their principal axes are within `angle_deg` of each other
      3. the lateral offset between the two axes is < max_offset_m
         (rejects distinct parallel rods lying side by side)
      4. the gap between their 1-D projections onto the axis is < max_gap_m
         AND less than 50 % of the combined span

    The smaller cloud is absorbed into the larger one; its entry is zeroed out
    (it still occupies a slot so the Hungarian assignment keeps N clusters).
    """
    n = len(clouds)

    def _pca(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (mean, principal axis, elongation σ1/σ2)."""
        mean = pts.mean(axis=0)
        _, S, Vt = np.linalg.svd(pts - mean, full_matrices=False)
        return mean, Vt[0], float(S[0] / max(S[1], 1e-9))

    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = clouds[i], clouds[j]
            if len(pi) < 30 or len(pj) < 30:
                continue

            ci, ai, ei = _pca(pi)
            cj, aj, ej = _pca(pj)
            if ei < min_elong or ej < min_elong:
                continue
            if abs(float(np.dot(ai, aj))) < np.cos(np.radians(angle_deg)):
                continue

            # Lateral offset: distance between the two axes (component of the
            # centroid difference perpendicular to the shared axis)
            axis = ai
            d    = cj - ci
            lat  = float(np.linalg.norm(d - np.dot(d, axis) * axis))
            if lat > max_offset_m:
                continue

            proj_i = pi @ axis
            proj_j = pj @ axis
            lo_all = min(proj_i.min(), proj_j.min())
            hi_all = max(proj_i.max(), proj_j.max())
            span   = hi_all - lo_all
            gap    = max(0.0, max(proj_i.min(), proj_j.min())
                             - min(proj_i.max(), proj_j.max()))

            if gap > max_gap_m or gap > 0.5 * span:
                continue

            # Absorb smaller cloud into larger; zero out smaller slot
            src, dst = (j, i) if len(pj) <= len(pi) else (i, j)
            clouds[dst] = np.concatenate([clouds[dst], clouds[src]], axis=0)
            clouds[src] = np.empty((0, 3), np.float32)
            if visual_cls[dst] is None:
                visual_cls[dst] = visual_cls[src]
            visual_cls[src] = None
            print(f"[Backproject] Co-axial merge: cluster {src} → {dst}  "
                  f"elong=({ei:.1f},{ej:.1f})  lat={lat*1000:.1f} mm  "
                  f"gap={gap:.3f} m  span={span:.3f} m")

    return clouds, visual_cls


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

    Returns (clouds, visual_cls, cluster_colours, cluster_elongations):
        clouds               : list of N (M_i, 3) world-space float32 arrays
        visual_cls           : list of N dominant visual class codes (or None per cluster)
        cluster_colours      : list of N mean HSV (3,) float32 arrays in [0,1]
        cluster_elongations  : list of N PCA elongation values (σ₁/σ₂, ≥1)

    DBSCAN subsampling uses a fixed seed (42) so the output is reproducible;
    without a seed, bbox-based features change across cache misses.
    """
    use_visual = classifier is not None
    cache_suffix = "_vis" if use_visual else ""
    cache_file = output_dir / f"proposals_N{n_components}_v12{cache_suffix}.pkl"
    if cache_file.exists():
        print(f"[Backproject] Loading cached proposals from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    pts3d_all = dust3r_result["pts3d"]   # list of (H, W, 3)
    confs_all = dust3r_result.get("confs")   # list of (H, W) or None (old cache)
    import config as _cfg
    conf_thresh = _cfg.DUST3R_CONF_THRESH if confs_all is not None else 0.0
    use_colour = image_paths is not None
    print(f"[Backproject] {len(sam_masks_per_image)} views × top-{n_components} masks  "
          f"colour={'yes' if use_colour else 'no'}  visual={'yes' if use_visual else 'no'}  "
          f"conf_filter={'>=%.1f' % conf_thresh if confs_all is not None else 'off'}")

    from component_classifier import crop_from_mask

    all_clouds    : list[np.ndarray]  = []
    all_centroids : list[np.ndarray]  = []   # 3D centroid (3,)
    all_colours   : list[np.ndarray]  = []   # mean HSV   (3,)
    all_visual_cls: list[str | None]  = []   # per-mask visual prediction (filled after batch)

    # Crop collection for batched classifier inference
    pending_crops : list[np.ndarray] = []   # BGR crops awaiting prediction
    pending_idx   : list[int]        = []   # index into all_clouds for each crop

    for img_idx, (masks, p3d) in enumerate(zip(sam_masks_per_image, pts3d_all)):
        conf = confs_all[img_idx] if confs_all is not None else None
        # Load image for colour and visual extraction (once per image).
        # Colour features come from the RAW image so they match the class
        # prototypes (built from raw reference images); CLAHE is only applied
        # to classifier crops for consistency with what SAM saw.
        bgr, img_rgb = None, None
        if use_colour and img_idx < len(image_paths):
            bgr_raw = cv2.imread(str(image_paths[img_idx]))
            if bgr_raw is not None:
                bgr     = _clahe_enhance(bgr_raw)
                img_rgb = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2RGB)

        for mask_dict in masks:
            seg = mask_dict["segmentation"]
            pts = _extract_mask_cloud(p3d, seg, conf=conf, conf_thresh=conf_thresh)
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

    colours_arr = np.array(all_colours, np.float32)   # (K, 3)

    print(f"[Backproject] Grouping {len(all_clouds)} clouds into ≤{n_components} "
          f"instances via 3D voxel overlap")

    # ── 3D-overlap graph grouping ─────────────────────────────────────────
    labels = _overlap_group(all_clouds, n_components,
                            overlap_min=_cfg.OVERLAP_MIN,
                            voxel_div=_cfg.OVERLAP_VOXEL_DIV,
                            colours=colours_arr,
                            color_weight=_cfg.COLOR_DIST_WEIGHT)

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

    # ── Co-axial rod fragment merge ───────────────────────────────────────
    if _cfg.MAX_ROD_GAP_M > 0:
        merged, visual_cls_per_cluster = _coaxial_merge(
            merged, visual_cls_per_cluster,
            max_gap_m=_cfg.MAX_ROD_GAP_M,
            angle_deg=_cfg.COAXIAL_ANGLE_DEG,
            max_offset_m=_cfg.COAXIAL_MAX_OFFSET_M,
            min_elong=_cfg.COAXIAL_MIN_ELONG,
        )

    print(f"[Backproject] Results after DBSCAN cleanup + co-axial merge:")
    for i, (pts, vcls) in enumerate(zip(merged, visual_cls_per_cluster)):
        c = _median_centroid(pts)
        print(f"  Cluster {i}: {len(pts):>7d} pts | "
              f"centroid ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})"
              + (f"  visual={vcls}" if vcls else ""))

    cluster_elongations = [_compute_elongation(c) for c in merged]
    for i, (e, pts) in enumerate(zip(cluster_elongations, merged)):
        print(f"  Cluster {i}: elong={e:.2f}")

    result = (merged, visual_cls_per_cluster, cluster_colours, cluster_elongations)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"[Backproject] Cached to {cache_file}")
    return result
