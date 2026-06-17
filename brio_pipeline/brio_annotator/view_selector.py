"""
Select the best multi-view images for annotation.

Scoring criteria (higher = better view for annotation):
  - fg_fraction  : fraction of image pixels that are foreground (more = more of the
                   construction is visible)
  - lcc_fraction : fraction of foreground pixels in the single largest connected
                   component (high = compact, unambiguous construction silhouette)
  - n_blobs      : penalise highly fragmented views where the construction is broken
                   into many disconnected regions by occlusion or perspective

The best view is the one with the most coherent and complete foreground silhouette.
"""
import cv2
import numpy as np
from pathlib import Path


BG_THRESHOLD = 245


def _foreground_stats(img_bgr: np.ndarray) -> dict:
    mask   = np.any(img_bgr < BG_THRESHOLD, axis=2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    total      = max(mask.size, 1)
    fg_pixels  = int(mask.sum())
    fg_fraction = fg_pixels / total

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    n_blobs = max(n_labels - 1, 0)

    if n_labels >= 2:
        lcc_area     = int(stats[1:, cv2.CC_STAT_AREA].max())
        lcc_fraction = lcc_area / max(fg_pixels, 1)
    else:
        lcc_fraction = 0.0

    return {"fg_fraction": fg_fraction, "n_blobs": n_blobs,
            "lcc_fraction": lcc_fraction}


def _score(stats: dict) -> float:
    return stats["fg_fraction"] * stats["lcc_fraction"] - 0.05 * stats["n_blobs"]


def collect_candidate_images(sample_id: int, multi_view_root: Path,
                              elevations: list[str], stride: int) -> list[Path]:
    """Return all sampled multi-view images for one sample across all elevations."""
    base = multi_view_root / f"Sample_{sample_id}"
    imgs: list[Path] = []
    for elev in elevations:
        folder = base / elev
        if not folder.exists():
            continue
        ring = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.JPG"))
        imgs.extend(ring[::stride])
    return imgs


def select_best_views(image_paths: list[Path], top_k: int = 3) -> list[Path]:
    """
    Return up to top_k image paths ranked by foreground coverage and coherence.

    Images are scored by how much of the construction is visible and how
    compact the visible silhouette is.  The highest-scoring image becomes the
    primary annotation view; others are used as fallbacks for occluded components.
    """
    scored: list[tuple[float, Path]] = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        scored.append((_score(_foreground_stats(img)), p))
    scored.sort(reverse=True)
    return [p for _, p in scored[:top_k]]
