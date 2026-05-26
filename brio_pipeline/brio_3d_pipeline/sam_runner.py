"""
Run SAM automatic mask generation on all images for a sample.
Keeps exactly the top-N masks per image where N = number of components from PUML.
Results are cached per-image so SAM only runs once.
"""
import json
import pickle
import numpy as np
from pathlib import Path
import cv2


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def _clahe_enhance(bgr: np.ndarray) -> np.ndarray:
    """Boost local contrast in LAB space so white components stand out from white backgrounds."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def run_sam_on_images(image_paths: list[Path], n_components: int,
                      output_dir: Path, weights_path: Path,
                      model_type: str = "vit_b", device: str = "cuda",
                      points_per_side: int = 16, iou_thresh: float = 0.80,
                      stability_thresh: float = 0.90, min_area: int = 200) -> list[list[dict]]:
    """
    Run SAM on each image and return top-N masks sorted by predicted IoU.

    Returns list (one per image) of list of mask dicts, each containing:
        segmentation : (H, W) bool array
        area         : int
        predicted_iou: float
        image_idx    : int
    """
    cache_file = output_dir / f"sam_masks_top{n_components}.pkl"
    if cache_file.exists():
        print(f"[SAM] Loading cached masks from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"[SAM] Generating masks for {len(image_paths)} images (top-{n_components} per image)")
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

    all_masks = []
    for idx, img_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(img_path))
        image_bgr = _clahe_enhance(image_bgr)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        masks = mask_gen.generate(image_rgb)

        # Sort by predicted IoU descending, keep top N
        masks.sort(key=lambda m: m["predicted_iou"], reverse=True)
        top_masks = masks[:n_components]

        # Slim the dict — only keep what we need downstream
        slim = [{
            "segmentation":  m["segmentation"],
            "area":          m["area"],
            "predicted_iou": m["predicted_iou"],
            "image_idx":     idx,
        } for m in top_masks]

        all_masks.append(slim)
        if (idx + 1) % 5 == 0 or idx == len(image_paths) - 1:
            print(f"[SAM]   {idx+1}/{len(image_paths)} images processed")

    with open(cache_file, "wb") as f:
        pickle.dump(all_masks, f)
    print(f"[SAM] Cached masks to {cache_file}")

    return all_masks


if __name__ == "__main__":
    import sys
    from config import (MULTI_VIEW, IMAGE_ELEVATION, SAM_WEIGHTS, SAM_MODEL_TYPE,
                        SAM_POINTS_SIDE, SAM_IOU_THRESH, SAM_STABILITY,
                        SAM_MIN_AREA, DEVICE, OUTPUTS_ROOT)
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
    )
    print(f"Total images: {len(masks)}, masks per image: {[len(m) for m in masks]}")
