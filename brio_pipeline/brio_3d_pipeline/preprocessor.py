"""
Auto-crop multi-view images to a tight bounding box around the BRIO construction,
at a FIXED scale consistent across all samples.

Fixed-scale logic
─────────────────
1. For each sample, detect the largest connected foreground component (LCC) and
   record its bounding box centroid and half-size.
2. Across all samples being processed, find the maximum half-size (the largest
   construction in the batch sets the scale).
3. Every sample is cropped to a square of that fixed size centred on its own LCC,
   with white-pixel padding if the crop exceeds the image boundary.

This guarantees that 1 pixel = the same physical length for every sample, which
keeps DUSt3R depth estimates and SAM mask sizes consistent across the dataset.
"""
import cv2
import numpy as np
from pathlib import Path

BG_THRESHOLD = 245   # pixels above this in all channels = background


# ── Internal helpers ─────────────────────────────────────────────────────────

def _lcc_info(img: np.ndarray) -> dict | None:
    """
    Find the largest connected foreground component and return its centroid
    and tight bounding box.  Returns None if no foreground found.
    """
    mask = np.any(img < BG_THRESHOLD, axis=2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if n_labels < 2:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    best  = int(np.argmax(areas)) + 1

    cx = float(centroids[best, 0])
    cy = float(centroids[best, 1])
    x0 = int(stats[best, cv2.CC_STAT_LEFT])
    y0 = int(stats[best, cv2.CC_STAT_TOP])
    x1 = x0 + int(stats[best, cv2.CC_STAT_WIDTH])
    y1 = y0 + int(stats[best, cv2.CC_STAT_HEIGHT])
    half_w = max(cx - x0, x1 - cx)
    half_h = max(cy - y0, y1 - cy)
    return {"cx": cx, "cy": cy, "half_w": half_w, "half_h": half_h}


def _sample_lcc_halfsize(image_paths: list[Path]) -> float:
    """Return the maximum half-size (pixels) of the LCC across all images."""
    maxhalf = 0.0
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        info = _lcc_info(img)
        if info:
            maxhalf = max(maxhalf, info["half_w"], info["half_h"])
    return maxhalf


def _crop_image_fixed(img: np.ndarray, cx: float, cy: float,
                      half: int) -> np.ndarray:
    """
    Crop a square of side 2*half centred at (cx, cy).
    Regions outside the image are filled with white (255).
    """
    h_img, w_img = img.shape[:2]
    size = 2 * half
    out  = np.full((size, size, 3), 255, dtype=np.uint8)

    # Source window in original image
    src_x0 = int(round(cx)) - half
    src_y0 = int(round(cy)) - half
    src_x1 = src_x0 + size
    src_y1 = src_y0 + size

    # Intersection with image bounds
    ix0 = max(0, src_x0); iy0 = max(0, src_y0)
    ix1 = min(w_img, src_x1); iy1 = min(h_img, src_y1)

    # Destination in output canvas
    dx0 = ix0 - src_x0; dy0 = iy0 - src_y0
    dx1 = dx0 + (ix1 - ix0); dy1 = dy0 + (iy1 - iy0)

    out[dy0:dy1, dx0:dx1] = img[iy0:iy1, ix0:ix1]
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def compute_global_halfsize(all_sample_image_paths: list[list[Path]],
                             padding: float = 0.20) -> int:
    """
    Compute the fixed crop half-size (pixels) from the largest LCC across
    all samples.  Add proportional padding and return as an integer.

    Call once before processing any sample; pass the result to crop_images().
    """
    max_half = 0.0
    for img_paths in all_sample_image_paths:
        h = _sample_lcc_halfsize(img_paths)
        max_half = max(max_half, h)

    padded = int(np.ceil(max_half * (1 + padding)))
    print(f"[Crop] Global half-size: {padded} px "
          f"(raw max {max_half:.1f} + {int(padding*100)}% padding) "
          f"→ {2*padded}×{2*padded} px crops")
    return padded


def crop_images(image_paths: list[Path], output_dir: Path,
                fixed_half: int) -> list[Path]:
    """
    Crop all images to a fixed square of side 2*fixed_half centred on each
    image's own LCC centroid, then downscale to half resolution.  Cached.

    Output filename includes the source folder name as a prefix to avoid
    collisions when different elevation folders share the same filename.

    Args:
        image_paths : original image paths
        output_dir  : where cropped images are saved
        fixed_half  : half-side from compute_global_halfsize()

    Returns list of cropped image paths (same order as input).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # marker name encodes both the crop size and the half-res flag so that
    # switching from an old full-res cache forces a clean regeneration
    marker = output_dir / f"half_{fixed_half}_r50.txt"

    # prefix each output filename with its source elevation folder to prevent
    # collisions (e.g. Images90/IMG_001.jpg → Images90_IMG_001.jpg)
    cropped_paths = [output_dir / f"{p.parent.name}_{p.name}" for p in image_paths]
    if marker.exists() and all(cp.exists() for cp in cropped_paths):
        out_side = fixed_half  # saved at half resolution → side = fixed_half
        print(f"[Crop] Using cached {out_side}×{out_side} crops from {output_dir}")
        return cropped_paths

    # Clear stale crops from a different half-size or resolution setting
    for f in output_dir.glob("*.jpg"):
        f.unlink()
    for f in output_dir.glob("*.JPG"):
        f.unlink()
    for f in output_dir.glob("half_*.txt"):
        f.unlink()

    out_side = fixed_half  # after halving: 2*fixed_half / 2 = fixed_half
    print(f"[Crop] Cropping {len(image_paths)} images → {out_side}×{out_side} px (half-res)")

    for p, out_path in zip(image_paths, cropped_paths):
        img  = cv2.imread(str(p))
        if img is None:
            continue
        info = _lcc_info(img)
        if info:
            cx, cy = info["cx"], info["cy"]
        else:
            cx, cy = img.shape[1] / 2, img.shape[0] / 2
        cropped = _crop_image_fixed(img, cx, cy, fixed_half)
        # Downscale to half resolution for faster inference
        h, w = cropped.shape[:2]
        cropped = cv2.resize(cropped, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path), cropped)

    marker.write_text(str(fixed_half))
    print(f"[Crop] Done → {output_dir}")
    return cropped_paths
