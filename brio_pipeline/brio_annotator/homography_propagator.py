"""
Propagate YOLO bounding box annotations from a source image to target images
via ORB feature matching + RANSAC homography.

For each target image:
  1. Compute homography H: source → target using ORB descriptors.
  2. Warp all four corners of each source bbox through H.
  3. Gate: discard warped bboxes whose overlap with the target foreground mask
     falls below min_fg_overlap — these components are occluded or out-of-frame
     in the target view, so injecting a wrong label is worse than no label.

Limitation: homography assumes a planar scene approximation.  It is accurate
within a single elevation ring (azimuth variation only) and degrades for large
elevation changes.  Use per-elevation-ring calls when elevation diversity matters.
"""
import cv2
import numpy as np
from pathlib import Path
from dataclasses import replace

from amodal_bbox import Annotation

BG_THRESHOLD = 245


# ── Foreground mask ───────────────────────────────────────────────────────────

def _foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
    mask   = np.any(img_bgr < BG_THRESHOLD, axis=2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


# ── Homography estimation ─────────────────────────────────────────────────────

def _compute_homography(src_bgr: np.ndarray, dst_bgr: np.ndarray,
                         min_inliers: int = 8) -> np.ndarray | None:
    """
    Compute homography from src → dst using ORB + BFMatcher + RANSAC.
    Returns None when fewer than min_inliers inliers survive RANSAC.
    """
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(src_bgr, None)
    kp2, des2 = orb.detectAndCompute(dst_bgr, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return H if (H is not None and n_inliers >= min_inliers) else None


# ── Bbox warping ──────────────────────────────────────────────────────────────

def _warp_bbox(bbox_xyxy: tuple[int, int, int, int],
               H: np.ndarray) -> tuple[int, int, int, int]:
    """Warp all four corners through H; return the enclosing axis-aligned bbox."""
    x0, y0, x1, y1 = bbox_xyxy
    corners = np.float32([[x0, y0], [x1, y0],
                           [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    warped  = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    return (int(warped[:, 0].min()), int(warped[:, 1].min()),
            int(warped[:, 0].max()), int(warped[:, 1].max()))


# ── Public API ────────────────────────────────────────────────────────────────

def propagate_to_image(source_bgr  : np.ndarray,
                        annotations : list[Annotation],
                        target_bgr  : np.ndarray,
                        min_matches : int   = 8,
                        min_fg_overlap: float = 0.30,
                        ) -> list[tuple[Annotation, bool]]:
    """
    Warp source annotations into the target image coordinate frame.

    Returns a list of (annotation, valid) pairs:
      valid=True  → warped bbox has sufficient foreground overlap; export this label.
      valid=False → component is likely occluded or out-of-frame; skip this label.

    The returned annotations have bbox_xyxy in target image coordinates and an
    empty mask (masks are not propagated — bboxes only).
    """
    H_tgt, W_tgt = target_bgr.shape[:2]
    target_fg    = _foreground_mask(target_bgr)
    H_mat        = _compute_homography(source_bgr, target_bgr, min_matches)

    if H_mat is None:
        print(f"  [Propagate] Homography failed (<{min_matches} matches) — skipping")
        return []

    results: list[tuple[Annotation, bool]] = []
    for ann in annotations:
        if ann.bbox_xyxy == (0, 0, 0, 0):
            results.append((ann, False))
            continue

        x0, y0, x1, y1 = _warp_bbox(ann.bbox_xyxy, H_mat)

        # Clip to image bounds
        x0c = max(0, x0);    y0c = max(0, y0)
        x1c = min(W_tgt - 1, x1);  y1c = min(H_tgt - 1, y1)

        if x1c <= x0c or y1c <= y0c:
            results.append((ann, False))
            continue

        # Foreground overlap gate
        roi_fg   = target_fg[y0c:y1c, x0c:x1c]
        roi_area = max((x1c - x0c) * (y1c - y0c), 1)
        valid    = float(roi_fg.sum()) / roi_area >= min_fg_overlap

        warped_ann = Annotation(
            cls         = ann.cls,
            instance_id = ann.instance_id,
            bbox_xyxy   = (x0c, y0c, x1c, y1c),
            mask        = np.zeros((H_tgt, W_tgt), bool),   # not propagated
            amodal      = ann.amodal,
        )
        results.append((warped_ann, valid))

    n_valid = sum(1 for _, v in results if v)
    print(f"  [Propagate] {n_valid}/{len(annotations)} boxes passed fg-overlap gate")
    return results
