"""
SAM in SamPredictor (prompted) mode.

Region proposals: K-means on foreground pixel coordinates gives N centroid
positions (one per PUML component).  Each centroid is used as a foreground
POINT prompt, and a tight bounding box (±component_radius around the centroid)
is provided as a spatial constraint.

Combining a point prompt (confirms foreground at the centroid) with a tight box
(constrains the search region) gives SAM unambiguous localisation per component.
This avoids the key failure of pure bbox prompts with K-means cluster boxes:
those boxes are Voronoi-partitioned regions that span large parts of the image
and often contain multiple components, causing SAM to segment wrong regions.
"""
import cv2
import numpy as np
from pathlib import Path

BG_THRESHOLD = 245


# ── Foreground utilities ──────────────────────────────────────────────────────

def foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of the largest connected foreground component."""
    mask   = np.any(img_bgr < BG_THRESHOLD, axis=2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return mask.astype(bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(bool)


def grey_fill(img_bgr: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Return a copy of img_bgr with non-foreground pixels set to neutral grey."""
    out = img_bgr.copy()
    out[~fg] = 128
    return out


# ── Region proposals via K-means ─────────────────────────────────────────────

def kmeans_centroids(fg_mask: np.ndarray, n: int,
                     min_area: int = 80,
                     padding_factor: float = 0.80,
                     ) -> tuple[list[list[float]], list[list[int]]]:
    """
    Cluster foreground pixels into N groups via K-means.

    Returns (centroids, boxes) where:
      centroids : [[cx, cy], ...]  — K-means cluster centres
      boxes     : [[x0,y0,x1,y1], ...] — tight bbox ± component_radius

    The radius is derived from expected component area (fg_area / N), so boxes
    are proportional to component density rather than fixed pixel sizes.
    """
    from sklearn.cluster import KMeans

    ys, xs = np.where(fg_mask)
    if len(ys) == 0:
        return [], []

    H, W  = fg_mask.shape
    k     = min(n, len(ys))
    pts   = np.column_stack([xs, ys]).astype(np.float32)

    # Expected pixel side-length per component
    fg_area = int(len(ys))
    radius  = max(20, int((fg_area / n) ** 0.5 * padding_factor))

    if k == 1:
        cx, cy = float(xs.mean()), float(ys.mean())
        box    = [max(0, int(cx) - radius), max(0, int(cy) - radius),
                  min(W - 1, int(cx) + radius), min(H - 1, int(cy) + radius)]
        return [[cx, cy]], [box]

    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(pts)

    centroids: list[list[float]] = []
    boxes:     list[list[int]]   = []
    for label_id in range(k):
        cluster_pts = pts[km.labels_ == label_id]
        if len(cluster_pts) < min_area:
            continue
        cx = float(cluster_pts[:, 0].mean())
        cy = float(cluster_pts[:, 1].mean())
        centroids.append([cx, cy])
        box = [max(0,     int(cx) - radius),
               max(0,     int(cy) - radius),
               min(W - 1, int(cx) + radius),
               min(H - 1, int(cy) + radius)]
        boxes.append(box)

    return centroids, boxes


# ── SAM predictor wrappers ────────────────────────────────────────────────────

def load_predictor(weights_path: Path, model_type: str = "vit_b",
                   device: str = "cuda"):
    """Load SAM and return a SamPredictor (not the automatic generator)."""
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[model_type](checkpoint=str(weights_path))
    sam.to(device)
    return SamPredictor(sam)


def predict_from_prompts(predictor,
                          image_rgb  : np.ndarray,
                          centroids  : list[list[float]],
                          boxes      : list[list[int]],
                          ) -> list[np.ndarray]:
    """
    Run SamPredictor with combined point + box prompts (one per component).

      point = K-means centroid  → confirms a foreground object at this location
      box   = tight bbox        → constrains SAM's search region

    Combining both gives unambiguous localisation: the box prevents SAM from
    growing into the whole scene, and the point prevents it from returning an
    empty or background mask inside the box.

    One encoder pass is shared across all N decoder invocations.
    Returns one boolean mask per (centroid, box) pair.
    """
    predictor.set_image(image_rgb)
    masks_out: list[np.ndarray] = []
    for (cx, cy), box in zip(centroids, boxes):
        x0, y0, x1, y1 = box
        input_box   = np.array([[x0, y0, x1, y1]], dtype=np.float32)
        input_point = np.array([[cx, cy]],          dtype=np.float32)
        input_label = np.array([1])                  # 1 = foreground

        masks, scores, _ = predictor.predict(
            point_coords   = input_point,
            point_labels   = input_label,
            box            = input_box,
            multimask_output = True,   # 3 candidates; keep highest confidence
        )
        best = int(scores.argmax())
        masks_out.append(masks[best].astype(bool))
    return masks_out


def segment_image(img_bgr: np.ndarray, n_components: int,
                   predictor, min_area: int = 80,
                   ) -> tuple[list[np.ndarray], list[list[int]]]:
    """
    Full single-image annotation flow:
      1. Compute foreground mask and grey-fill background.
      2. K-means → N centroid positions + tight bbox proposals.
      3. SamPredictor (point + box prompts) → N refined masks.

    Returns (masks, boxes).
    """
    fg         = foreground_mask(img_bgr)
    img_filled = grey_fill(img_bgr, fg)
    image_rgb  = cv2.cvtColor(img_filled, cv2.COLOR_BGR2RGB)

    centroids, boxes = kmeans_centroids(fg, n_components, min_area=min_area)
    if not centroids:
        return [], []

    masks = predict_from_prompts(predictor, image_rgb, centroids, boxes)
    return masks, boxes
