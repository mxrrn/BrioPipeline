"""
Maps component image folder names → PUML class codes, and extracts per-class
color/texture prototypes from the reference images in 02-resources/data/component_images/.

Prototypes are cached so extraction only runs once.
"""
import cv2
import numpy as np
import pickle
from pathlib import Path

# Folder name → PUML class code
FOLDER_TO_CLASS = {
    "blockwood11":   "blwo11",
    "blockwood21":   "blwo21",
    "bolt":          "bo",
    "nose":          "no",
    "nut":           "nu",
    "plateplastic53":"plpl53",
    "platewood21":   "plwo21",
    "platewood31":   "plwo31",
    "platewood33":   "plwo33",
    "platewood53":   "plwo53",
    "plug":          "pl",
    "rodlong":       "rolo",
    "rodmedium":     "rome",
    "rodsmall":      "rosm",
    "screwlong":     "sclo",
    "screwmedium":   "scme",
    "screwsmall":    "scsm",
    "sleeve":        "sl",
    "strapplastic5": "stpl5",
    "strapwood3":    "stwo3",
    "strapwood4":    "stwo4",
    "strapwood5":    "stwo5",
    "strapwood6":    "stwo6",
    "strapwood7":    "stwo7",
    "strapwood9":    "stwo9",
    "tire":          "ti",
    "washer":        "wa",
    "wheelred":      "whre",
    "wheelwhite":    "whwh",
}
CLASS_TO_FOLDER = {v: k for k, v in FOLDER_TO_CLASS.items()}

# Background threshold — same as preprocessor
BG_THRESHOLD = 245


def _mean_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """Return mean HSV of non-background pixels, normalised to [0,1]."""
    fg_mask = np.any(image_bgr < BG_THRESHOLD, axis=2)
    if fg_mask.sum() < 10:
        return np.zeros(3, np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[fg_mask]          # (M, 3)
    mean   = pixels.mean(axis=0)   # H ∈ [0,180], S ∈ [0,255], V ∈ [0,255]
    return (mean / np.array([180., 255., 255.])).astype(np.float32)


def build_color_prototypes(component_images_dir: Path,
                            elevation: str = "Images45",
                            cache_path: Path | None = None) -> dict[str, np.ndarray]:
    """
    Extract mean-HSV color prototype for each component class from reference images.

    Returns dict: class_code → (3,) float32 normalised HSV [0,1].
    """
    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    prototypes = {}
    for folder, cls in FOLDER_TO_CLASS.items():
        img_dir = component_images_dir / folder / elevation
        if not img_dir.exists():
            continue
        img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.JPG"))
        if not img_paths:
            continue

        # Average HSV across all reference views
        hsv_samples = []
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is not None:
                hsv_samples.append(_mean_hsv(img))

        if hsv_samples:
            prototypes[cls] = np.stack(hsv_samples).mean(axis=0)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(prototypes, f)

    print(f"[ComponentMap] Built color prototypes for {len(prototypes)} classes")
    return prototypes


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import DATA_ROOT, OUTPUTS_ROOT
    protos = build_color_prototypes(
        DATA_ROOT / "component_images",
        cache_path=OUTPUTS_ROOT / "color_prototypes.pkl"
    )
    for cls, hsv in sorted(protos.items()):
        print(f"  {cls:<12}  H={hsv[0]:.3f}  S={hsv[1]:.3f}  V={hsv[2]:.3f}")
