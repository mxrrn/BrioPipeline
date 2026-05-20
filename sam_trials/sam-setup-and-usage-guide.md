# SAM Setup and Usage Guide — BRIO Component Annotation

**Purpose:** End-to-end guide for installing SAM, running it on the BRIO construction
dataset, performing multi-component annotation, and exporting labels for CNN training.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Environment Setup](#2-environment-setup)
3. [Installing SAM and Dependencies](#3-installing-sam-and-dependencies)
4. [Downloading Model Weights](#4-downloading-model-weights)
5. [Dataset Paths and File Layout](#5-dataset-paths-and-file-layout)
6. [How Many Components Per Image — Single vs Multi-Mask](#6-how-many-components-per-image--single-vs-multi-mask)
7. [Inference Modes: Automatic vs Prompted](#7-inference-modes-automatic-vs-prompted)
8. [Prompt Engineering for BRIO Parts](#8-prompt-engineering-for-brio-parts)
9. [Parsing the PUML Ground Truth](#9-parsing-the-puml-ground-truth)
10. [Full Annotation Workflow (Step by Step)](#10-full-annotation-workflow-step-by-step)
11. [Exporting Labels](#11-exporting-labels)
12. [Output Format Reference](#12-output-format-reference)
13. [Extending to Multi-View Images](#13-extending-to-multi-view-images)
14. [Annotation Validation Against PUML](#14-annotation-validation-against-puml)
15. [Common Issues and Fixes](#15-common-issues-and-fixes)

---

## 1. System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8 | 3.10 |
| RAM | 8 GB | 16 GB |
| Disk (weights) | 2.5 GB | 2.5 GB |
| GPU (VRAM) | None (CPU fallback) | 8 GB+ (NVIDIA) |
| CUDA | Not required | 11.7+ |

SAM runs on CPU. The ViT-H encoder takes ~5–10 seconds per image on CPU. For the
annotation phase on 150 `Construction.jpg` images, total runtime is ~15–25 minutes on
CPU — acceptable since annotation is a one-time offline task.

---

## 2. Environment Setup

Create a dedicated conda environment to keep SAM isolated from the `petr` environment:

```bash
conda create -n sam python=3.10 -y
conda activate sam
```

All commands in this guide assume the `sam` environment is active.

---

## 3. Installing SAM and Dependencies

```bash
# Install PyTorch (CPU version — sufficient for annotation)
pip install torch torchvision

# If you have an NVIDIA GPU and want faster encoding:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install SAM itself
pip install git+https://github.com/facebookresearch/segment-anything.git

# Required utilities
pip install opencv-python pycocotools matplotlib numpy Pillow tqdm
```

Verify the installation:

```python
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
print("SAM installed OK")
```

---

## 4. Downloading Model Weights

Three model sizes are available. For annotation work, ViT-H gives the best mask quality.

```bash
# Create a weights directory
mkdir -p /mnt/c/BA/03-code/sam_weights

# Download ViT-H (recommended — best quality, ~2.4 GB)
wget -P /mnt/c/BA/03-code/sam_weights \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Alternative: ViT-B (smaller, ~375 MB — faster on CPU if ViT-H is too slow)
wget -P /mnt/c/BA/03-code/sam_weights \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Loading the model:

```python
import torch
from segment_anything import sam_model_registry

CHECKPOINT = "/mnt/c/BA/03-code/sam_weights/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"   # use "vit_b" if you downloaded the smaller variant

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=device)
```

---

## 5. Dataset Paths and File Layout

The dataset is split across two directory trees. Key paths:

```
/mnt/c/BA/02-resources/data/
├── constructions/
│   ├── Sample_1_to_31/
│   │   └── Sample_N_InstanceDiagram/
│   │       ├── Construction.jpeg     ← main photo (note: .jpeg not .jpg in this batch)
│   │       ├── InstanceDiagramSN.puml
│   │       └── Mapping.png
│   ├── Sample_32_to_51/
│   │   └── Sample_N_InstanceDiagram/
│   │       ├── Construction.jpg      ← main photo (.jpg in this batch)
│   │       ├── InstanceDiagramSN.puml
│   │       └── Mapping.png
│   ├── Sample_52_to_81/  ...
│   ├── Sample_82_to_111/ ...
│   └── Sample_112_to_150/ ...
└── multi_view_images/
    └── Sample_N/
        ├── Images30/   (24 images — azimuth sweep at 30° elevation)
        ├── Images45/   (24 images — azimuth sweep at 45° elevation)
        ├── Images60/   (24 images — azimuth sweep at 60° elevation)
        └── Images90/   (6 images — near-overhead)
```

**Note:** Samples 1–31 use `Construction.jpeg`; samples 32–150 use `Construction.jpg`.
The helper function below handles this difference.

```python
import os
from pathlib import Path

CONSTRUCTIONS_ROOT = Path("/mnt/c/BA/02-resources/data/constructions")
MULTIVIEW_ROOT     = Path("/mnt/c/BA/02-resources/data/multi_view_images")

BATCH_DIRS = {
    range(1,  32):  "Sample_1_to_31",
    range(32, 52):  "Sample_32_to_51",
    range(52, 82):  "Sample_52_to_81",
    range(82, 112): "Sample_82_to_111",
    range(112, 151):"Sample_112_to_150",
}

def get_sample_dir(n: int) -> Path:
    for r, batch in BATCH_DIRS.items():
        if n in r:
            return CONSTRUCTIONS_ROOT / batch / f"Sample_{n}_InstanceDiagram"
    raise ValueError(f"Sample {n} not found in any batch")

def get_construction_image(n: int) -> Path:
    d = get_sample_dir(n)
    for ext in ("Construction.jpeg", "Construction.jpg"):
        p = d / ext
        if p.exists():
            return p
    raise FileNotFoundError(f"No construction image for Sample {n}")

def get_puml_path(n: int) -> Path:
    d = get_sample_dir(n)
    return d / f"InstanceDiagramS{n}.puml"

def get_multiview_images(n: int, elevation: int = 45) -> list[Path]:
    """elevation: 30, 45, 60, or 90"""
    folder = MULTIVIEW_ROOT / f"Sample_{n}" / f"Images{elevation}"
    return sorted(folder.glob("*.jpg"))
```

---

## 6. How Many Components Per Image — Single vs Multi-Mask

**SAM can segment multiple components in a single image pass. There is no one-component
limit.**

Both inference modes return all segmented regions at once:

- **Automatic mode:** SAM tiles the image with a grid of prompt points and returns a
  mask for every distinct region it finds — typically 10–50+ masks per image depending
  on complexity and settings. All BRIO components in a photo are returned together.

- **Prompted mode:** Each call to `predictor.predict()` returns up to 3 masks for the
  prompted location. You call it once per component to get that component's mask, but you
  can do this in a loop over all components without re-encoding the image (the image
  encoding is cached after `predictor.set_image()`).

For annotation purposes, **automatic mode is the right choice** — one function call
returns candidate masks for every component in the image simultaneously.

---

## 7. Inference Modes: Automatic vs Prompted

### 7.1 Automatic Mask Generator

Finds all segments in an image with no human input. Best for initial annotation.

```python
from segment_anything import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,          # grid density — higher = more masks found, slower
    pred_iou_thresh=0.88,        # discard masks with predicted IoU < this
    stability_score_thresh=0.95, # discard unstable masks
    crop_n_layers=1,             # repeat at 2 zoom levels (helps find small parts)
    crop_n_points_downscale_factor=2,
    min_mask_region_area=500,    # discard tiny noise regions (pixels²)
)

import cv2
image = cv2.imread(str(get_construction_image(10)))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(image)

# Each mask is a dict:
# {
#   "segmentation":      np.ndarray (H, W) bool — the pixel mask
#   "bbox":              [x, y, w, h]  in pixels
#   "area":              int (pixel count)
#   "predicted_iou":     float (SAM's confidence in mask quality)
#   "stability_score":   float (mask consistency across threshold changes)
#   "point_coords":      [[x, y]]  the prompt point that generated this mask
#   "crop_box":          [x, y, w, h]  the crop window used
# }

print(f"Found {len(masks)} masks")
```

### 7.2 Prompted Predictor

Use this when you know roughly where a component is (e.g., from Mapping.drawio or
from a bounding-box estimate). This gives you a single component's mask on demand.

```python
from segment_anything import SamPredictor
import numpy as np

predictor = SamPredictor(sam)

image = cv2.imread(str(get_construction_image(10)))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor.set_image(image)    # encode once — ~5–10s on CPU, cached thereafter

# --- Point prompt: click a point inside the component ---
# point_coords: array of (x, y) points
# point_labels: 1 = foreground (inside component), 0 = background (outside)
masks, scores, logits = predictor.predict(
    point_coords=np.array([[320, 240]]),   # pixel (x, y) of the component center
    point_labels=np.array([1]),
    multimask_output=True                  # returns 3 candidate masks
)
# masks[np.argmax(scores)] is the best mask

# --- Box prompt: draw a bounding box around the component ---
masks, scores, logits = predictor.predict(
    box=np.array([100, 80, 400, 300]),     # [x_min, y_min, x_max, y_max]
    multimask_output=False
)

# --- Combining point + box (most precise) ---
masks, scores, logits = predictor.predict(
    point_coords=np.array([[200, 180], [50, 50]]),
    point_labels=np.array([1, 0]),   # 1=fg point, 0=bg exclusion point
    box=np.array([100, 80, 400, 300]),
    multimask_output=False
)
```

---

## 8. Prompt Engineering for BRIO Parts

### 8.1 Automatic Mode — Tuning Parameters

The default settings work for natural images. BRIO components are small,
colorful, and often touching. Adjust:

| Parameter | Default | Recommended for BRIO | Reason |
|-----------|---------|----------------------|--------|
| `points_per_side` | 32 | **32–64** | Small parts need dense sampling |
| `pred_iou_thresh` | 0.88 | **0.85** | Lower to recover smaller components |
| `stability_score_thresh` | 0.95 | **0.92** | Slightly lower for thin parts |
| `min_mask_region_area` | 100 | **300–600** | Filter noise but keep small bolts |
| `crop_n_layers` | 0 | **1–2** | Multi-scale helps small bolts/screws |
| `box_nms_thresh` | 0.7 | **0.5** | Tighter NMS to prevent merging adjacent parts |

```python
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=300,
    box_nms_thresh=0.5,
)
```

### 8.2 Prompted Mode — Choosing Good Points

When using `SamPredictor` for a specific component:

1. **One foreground point per component** — place it near the visual center of the
   part, not on its edge. For cylindrical parts (bolts, screws), click the cylinder body.

2. **Add background exclusion points** to prevent SAM from expanding into a neighbour.
   If bolt `bo_1` is touching plate `plwo31_1`, add a point labeled `0` on the plate:
   ```python
   point_coords = np.array([[cx_bolt, cy_bolt], [cx_plate, cy_plate]])
   point_labels = np.array([1, 0])   # bolt=fg, plate=bg
   ```

3. **Use a bounding box from Mapping.drawio** if available — it constrains SAM to the
   right region and prevents mask drift. Box + point together is the most reliable:
   ```python
   masks, scores, _ = predictor.predict(
       point_coords=np.array([[cx, cy]]),
       point_labels=np.array([1]),
       box=np.array([x1, y1, x2, y2]),
       multimask_output=False
   )
   ```

4. **Use `multimask_output=True`** and inspect all 3 candidates when ambiguous.
   `scores[np.argmax(scores)]` gives the highest-confidence one. For parts that share
   a joint (bolt through a washer), the three candidates may give you: just the bolt,
   bolt + washer together, and bolt + washer + sleeve. Pick the one matching the PUML.

5. **Iterative refinement** — if the first mask is wrong, pass its logits back:
   ```python
   masks, scores, logits = predictor.predict(
       point_coords=np.array([[cx, cy]]),
       point_labels=np.array([1]),
       multimask_output=False
   )
   # Pass logits as a mask hint for the next prompt
   masks, scores, logits = predictor.predict(
       point_coords=np.array([[cx, cy], [bad_x, bad_y]]),
       point_labels=np.array([1, 0]),
       mask_input=logits,            # feed previous prediction back in
       multimask_output=False
   )
   ```

### 8.3 Background Separation Strategy

BRIO constructions are typically photographed on a **plain white or light-colored surface**.
This means the background is largely uniform and SAM will naturally assign it to one large
low-confidence mask. Filter it out by area:

```python
image_area = image.shape[0] * image.shape[1]

component_masks = [
    m for m in masks
    if m["area"] < image_area * 0.5      # reject background (>50% of image)
    and m["area"] > 300                   # reject noise
]
```

---

## 9. Parsing the PUML Ground Truth

The `InstanceDiagramSN.puml` file tells you exactly which components are in each
construction and how many instances of each type exist. Parse this before annotating
so you have the ground-truth component manifest to match against.

```python
import re
from collections import defaultdict

# Component abbreviations from the thesis (30 classes)
COMPONENT_CLASSES = [
    "bo", "nu", "pl", "sl", "wa", "ti", "no", "whwh", "whre",
    "ro_lo", "rome", "rosm", "sclo", "scme", "scsm",
    "stwo3", "stwo4", "stwo5", "stwo6", "stwo7", "stwo8", "stwo9",
    "blwo11", "plwo21", "plwo31", "plwo41", "plwo42", "plwo43",
    "plwo53", "plpl53", "stpl5"
]
# Map abbreviation → integer class ID for YOLO
CLASS_ID = {name: i for i, name in enumerate(COMPONENT_CLASSES)}

def parse_puml_components(puml_path: Path) -> dict[str, int]:
    """
    Returns {component_abbreviation: instance_count} for a PUML file.
    Example: {"bo": 2, "wa": 1, "plwo31": 1}
    """
    text = puml_path.read_text()
    # Match lines like: object "bo_1 : Component" as bo1
    pattern = re.compile(r'object\s+"([^"]+)\s*:\s*Component"')
    counts = defaultdict(int)
    for match in pattern.finditer(text):
        label = match.group(1).strip()   # e.g. "bo_1"
        abbr  = label.rsplit("_", 1)[0]  # e.g. "bo"
        counts[abbr] += 1
    return dict(counts)

# Example
manifest = parse_puml_components(get_puml_path(10))
# {"no": 1, "pl": 1, "plwo31": 1}
print(manifest)
```

---

## 10. Full Annotation Workflow (Step by Step)

This section walks through annotating a single sample end-to-end, then
shows how to batch it across all 150 samples.

### Step 1 — Initialize SAM

```python
import torch, cv2, json, numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

CHECKPOINT = "/mnt/c/BA/03-code/sam_weights/sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT)
sam.to(device=device)
```

### Step 2 — Load Image and Generate Automatic Masks

```python
def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    min_mask_region_area=300,
    box_nms_thresh=0.5,
)

sample_n = 10
image = load_rgb(get_construction_image(sample_n))
H, W = image.shape[:2]

masks = mask_generator.generate(image)

# Filter out background and noise
image_area = H * W
masks = [m for m in masks if 300 < m["area"] < image_area * 0.5]
masks.sort(key=lambda m: m["area"])   # smallest first (easier to review)

print(f"Sample {sample_n}: {len(masks)} candidate masks")
```

### Step 3 — Get Component Manifest from PUML

```python
manifest = parse_puml_components(get_puml_path(sample_n))
# e.g. {"no": 1, "pl": 1, "plwo31": 1}

total_components = sum(manifest.values())
print(f"PUML says {total_components} components: {manifest}")
# Quick sanity check
if len(masks) < total_components:
    print("WARNING: fewer masks than components — consider lowering pred_iou_thresh")
elif len(masks) > total_components * 3:
    print("WARNING: many more masks than components — some are background/noise")
```

### Step 4 — Visualize Masks for Labeling

Run this to produce a side-by-side visualization you can use to label each mask:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_masks_for_labeling(image, masks, title=""):
    fig, axes = plt.subplots(1, len(masks) + 1,
                             figsize=(4 * (len(masks) + 1), 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, m in enumerate(masks):
        overlay = image.copy()
        overlay[m["segmentation"]] = [0, 200, 100]   # green tint
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(
            f"Mask {i}\n"
            f"area={m['area']}\n"
            f"iou={m['predicted_iou']:.2f}"
        )
        x, y, bw, bh = m["bbox"]
        rect = patches.Rectangle((x, y), bw, bh,
                                  linewidth=1, edgecolor="red", facecolor="none")
        axes[i + 1].add_patch(rect)
        axes[i + 1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"/mnt/c/BA/04-experiments/sam_vis_sample_{sample_n}.png", dpi=100)
    plt.show()

show_masks_for_labeling(image, masks,
    title=f"Sample {sample_n} | Components: {manifest}")
```

Open the saved PNG. For each numbered mask, assign a component class from the manifest.
Write down (or enter in the script below) the assignments as a dict:

```python
# You fill this in after looking at the visualization:
assignments = {
    0: "no",      # mask index → component abbreviation
    1: "pl",
    2: "plwo31",
}
```

### Step 5 — Interactive Refinement for Merged Masks

If a mask covers two merged components (common when parts touch), switch to
prompted mode to split them:

```python
predictor = SamPredictor(sam)
predictor.set_image(image)   # encode once — reuse for all prompt calls

# Suppose mask 1 merges a bolt and a washer.
# Click inside the bolt only, with an exclusion point on the washer:
bolt_masks, bolt_scores, _ = predictor.predict(
    point_coords=np.array([[bolt_cx, bolt_cy], [washer_cx, washer_cy]]),
    point_labels=np.array([1, 0]),
    multimask_output=False
)
washer_masks, washer_scores, _ = predictor.predict(
    point_coords=np.array([[washer_cx, washer_cy], [bolt_cx, bolt_cy]]),
    point_labels=np.array([1, 0]),
    multimask_output=False
)
# Replace the merged mask with two separate ones:
masks.pop(1)
masks.insert(1, {"segmentation": bolt_masks[0],   "bbox": mask_to_bbox(bolt_masks[0]),
                  "area": bolt_masks[0].sum(),      "predicted_iou": bolt_scores[0]})
masks.insert(2, {"segmentation": washer_masks[0],  "bbox": mask_to_bbox(washer_masks[0]),
                  "area": washer_masks[0].sum(),    "predicted_iou": washer_scores[0]})

def mask_to_bbox(mask):
    rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
```

### Step 6 — Record Labeled Annotations

```python
annotations = []
for mask_idx, class_abbr in assignments.items():
    m = masks[mask_idx]
    x, y, bw, bh = m["bbox"]
    annotations.append({
        "sample":     sample_n,
        "class_abbr": class_abbr,
        "class_id":   CLASS_ID[class_abbr],
        "bbox_xywh":  [x, y, bw, bh],
        "mask_rle":   mask_to_rle(m["segmentation"]),   # see Section 11
        "area":       m["area"],
        "iou":        m["predicted_iou"],
    })
```

---

## 11. Exporting Labels

### 11.1 YOLO Format (for object detection training)

YOLO expects one `.txt` file per image, same filename stem as the image.
Each line: `class_id x_center y_center width height` (all normalized 0–1).

```python
def export_yolo(annotations: list, image_path: Path, image_w: int, image_h: int,
                out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_dir / (image_path.stem + ".txt")
    lines = []
    for ann in annotations:
        x, y, bw, bh = ann["bbox_xywh"]
        cx = (x + bw / 2) / image_w
        cy = (y + bh / 2) / image_h
        nw = bw / image_w
        nh = bh / image_h
        lines.append(f"{ann['class_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines))

# Usage
LABELS_DIR = Path("/mnt/c/BA/04-experiments/sam_labels/yolo")
export_yolo(annotations, get_construction_image(10), W, H, LABELS_DIR)
```

Also copy the image to the YOLO images directory:

```python
import shutil
IMAGES_DIR = Path("/mnt/c/BA/04-experiments/sam_labels/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy(get_construction_image(10), IMAGES_DIR)
```

### 11.2 COCO JSON Format (for instance segmentation training)

COCO stores all annotations in a single JSON with RLE-encoded masks.

```python
from pycocotools import mask as mask_utils

def mask_to_rle(bool_mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(bool_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def build_coco_dataset(all_annotations: list, image_metas: list) -> dict:
    categories = [{"id": CLASS_ID[c], "name": c} for c in COMPONENT_CLASSES]
    images, annotations_out = [], []
    ann_id = 1

    for img_meta in image_metas:
        images.append({
            "id": img_meta["id"], "file_name": img_meta["file_name"],
            "width": img_meta["width"], "height": img_meta["height"],
        })

    for ann in all_annotations:
        x, y, bw, bh = ann["bbox_xywh"]
        annotations_out.append({
            "id":           ann_id,
            "image_id":     ann["image_id"],
            "category_id":  ann["class_id"],
            "segmentation": ann["mask_rle"],
            "bbox":         [x, y, bw, bh],
            "area":         ann["area"],
            "iscrowd":      0,
        })
        ann_id += 1

    return {"images": images, "annotations": annotations_out, "categories": categories}
```

### 11.3 YOLO Dataset YAML (required to train with Ultralytics YOLOv8)

Create this file at `/mnt/c/BA/04-experiments/sam_labels/dataset.yaml`:

```yaml
path: /mnt/c/BA/04-experiments/sam_labels
train: images/train
val: images/val

nc: 31   # number of classes (adjust to actual count after annotation)
names:
  0:  bo
  1:  nu
  2:  pl
  3:  sl
  4:  wa
  5:  ti
  6:  no
  7:  whwh
  8:  whre
  9:  ro_lo
  10: rome
  11: rosm
  12: sclo
  13: scme
  14: scsm
  15: stwo3
  16: stwo4
  17: stwo5
  18: stwo6
  19: stwo7
  20: stwo8
  21: stwo9
  22: blwo11
  23: plwo21
  24: plwo31
  25: plwo41
  26: plwo42
  27: plwo43
  28: plwo53
  29: plpl53
  30: stpl5
```

---

## 12. Output Format Reference

After running the annotation workflow for sample `N`, you will have produced:

```
/mnt/c/BA/04-experiments/sam_labels/
├── images/
│   └── Construction_N.jpg          ← copy of the construction photo
├── yolo/
│   └── Construction_N.txt          ← YOLO labels (one bbox per line)
├── coco/
│   └── annotations.json            ← cumulative COCO JSON (all samples)
└── masks/
    └── Sample_N/
        └── component_abbr_instNum.png  ← binary mask PNGs (optional, for debugging)
```

Each YOLO label line:
```
class_id  x_center  y_center  width  height
      3   0.412500  0.583333  0.125  0.208333
```

All values are **relative to image dimensions** (0 to 1).

---

## 13. Extending to Multi-View Images

Once construction images are labeled, you want the same component labels applied to
the multi-view images. Two strategies:

### Strategy A — Run SAM on Multi-View Images Independently

Repeat Steps 2–6 for each multi-view image. Since the component manifest (from the PUML)
is the same for all views of the same construction, you can reuse `manifest` directly
and only need to re-run the matching step.

```python
from tqdm import tqdm

for elevation in [30, 45, 60, 90]:
    mv_images = get_multiview_images(sample_n, elevation)
    for img_path in tqdm(mv_images, desc=f"Sample {sample_n} @ {elevation}°"):
        image = load_rgb(img_path)
        masks = mask_generator.generate(image)
        # ... same pipeline as Steps 2–6
        export_yolo(annotations, img_path, image.shape[1], image.shape[0], LABELS_DIR)
```

This is the most accurate approach but requires reviewing masks for every view.
For 150 samples × 78 views = 11,700 images, full manual review is impractical.
**Annotate Images45 fully** (best visual coverage at 45° elevation) and treat
the other views as augmentation via copy-paste or label propagation.

### Strategy B — Label Propagation via Template Matching

If you have a labeled mask for a component in one view, attempt to find it in
adjacent views by searching for a similar patch:

```python
import cv2

def propagate_label_to_view(source_mask, source_image, target_image):
    # Extract source patch from the bounding box of the mask
    x, y, bw, bh = mask_to_bbox(source_mask)
    template = source_image[y:y+bh, x:x+bw]

    # Match template in target image
    result = cv2.matchTemplate(
        cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(template,     cv2.COLOR_RGB2GRAY),
        cv2.TM_CCOEFF_NORMED
    )
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val > 0.7:   # confidence threshold — tune per component type
        tx, ty = max_loc
        return [tx, ty, bw, bh], max_val
    return None, max_val  # not found confidently
```

Template matching is reliable for views with small azimuth differences (~15°)
but degrades for large rotations. Use it to bootstrap labels; verify spot-checks manually.

---

## 14. Annotation Validation Against PUML

After labeling all masks for a sample, verify the annotation is consistent with
the PUML ground truth before exporting:

```python
from collections import Counter

def validate_annotations(annotations: list, manifest: dict, sample_n: int):
    detected = Counter(a["class_abbr"] for a in annotations)
    errors = []

    for abbr, expected_count in manifest.items():
        found = detected.get(abbr, 0)
        if found != expected_count:
            errors.append(
                f"  {abbr}: expected {expected_count}, got {found}"
            )

    for abbr, found in detected.items():
        if abbr not in manifest:
            errors.append(f"  {abbr}: labeled but not in PUML (spurious)")

    if errors:
        print(f"Sample {sample_n} — VALIDATION FAILED:")
        for e in errors: print(e)
        return False

    print(f"Sample {sample_n} — OK ({sum(manifest.values())} components)")
    return True

validate_annotations(annotations, manifest, sample_n)
```

Common failure modes and what to do:

| Failure | Cause | Fix |
|---------|-------|-----|
| `bo: expected 1, got 0` | SAM missed the bolt (too small) | Lower `pred_iou_thresh`; add a manual point prompt |
| `plwo31: expected 1, got 2` | SAM split the plate into two regions | Merge the two masks: `combined = m1["segmentation"] \| m2["segmentation"]` |
| `wa: expected 1, got 0` but total masks match | Washer labeled as wrong type | Re-check the visualization — washer may have been merged into adjacent bolt mask |

---

## 15. Common Issues and Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| SAM out of memory | `RuntimeError: CUDA out of memory` | Use ViT-B instead of ViT-H, or run on CPU |
| Too few masks for small components | Bolts / screws not segmented | Increase `points_per_side` to 64; add `crop_n_layers=2`; lower `min_mask_region_area` to 200 |
| Masks span entire construction | Large merged mask | Lower `box_nms_thresh` to 0.4; add a background exclusion box |
| YOLO labels outside [0,1] | Bbox calculation error | Verify image dimensions match what was passed to `export_yolo` |
| `Class X not in CLASS_ID` | New component abbreviation | Add the abbreviation to `COMPONENT_CLASSES` list and regenerate `CLASS_ID` |
| Image reads as `None` | Wrong path or `.jpeg` vs `.jpg` | Use `get_construction_image(n)` which handles both extensions |
| PUML parser returns empty dict | Unusual PUML formatting | Print the raw PUML and test the regex on a few lines manually |

---

## Quick-Start Summary

```bash
# 1. Create environment
conda create -n sam python=3.10 -y && conda activate sam

# 2. Install
pip install torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib numpy Pillow tqdm

# 3. Download weights
wget -P /mnt/c/BA/03-code/sam_weights \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 4. Run annotation script (put the code from Sections 9–11 in a .py file)
python /mnt/c/BA/00-project/annotate_with_sam.py --sample 10
```

The annotation script for Section 10 should be saved to `00-project/annotate_with_sam.py`
since all new code belongs in `00-project/` (not in `03-code/`).
