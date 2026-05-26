# BRIO Pipeline — Documentation

This directory contains two pipelines and a set of launcher scripts:

| Pipeline | Purpose | Runtime |
|---|---|---|
| `brio_3d_pipeline/` | Slow annotation pipeline. Produces 3D ground-truth labels and training data. | ~35 min/sample (cold), ~2 min (cached) |
| `brio_fast_pipeline/` | Fast inference. Trained YOLOv8 + fixed-rig triangulation. | < 2 seconds/sample |

The slow pipeline runs once per sample to produce training labels. The fast pipeline trains on those labels and then runs on unseen samples.

**Logs are written automatically** on every run — no manual redirection needed. Both pipelines write timestamped `.log` files to their own `logs/` folder.

---

## Quick-start

All phases are launched from `brio_pipeline/` using the shell scripts below. Run them from that directory:

```bash
cd /mnt/c/BA/00-project/brio_pipeline
```

### Phase 0 — Train the component visual classifier (once)
```bash
./train_classifier.sh
```
Trains MobileNetV3-small on the isolated component dataset (~40 epochs, ~10 min on GPU). Weights are saved to `brio_3d_pipeline/component_classifier.pth` and loaded automatically by the slow pipeline. Only needs to run once; retrain if the component dataset changes.

### Phase 1 — Annotate samples (slow pipeline)
```bash
./slow.sh 113 114 115 116 117
```
Runs CLAHE → DUSt3R → SAM → clustering → visual classification on the listed samples. Each run creates a new timestamped folder `outputs/run_NNN_YYYYMMDD_HHMM/`. Logs to `brio_3d_pipeline/logs/`.

### Phase 2 — Export training labels
```bash
./labels.sh
```
Converts all completed slow-pipeline outputs into YOLO-format labels. Logs to `brio_fast_pipeline/logs/`.

To export only specific samples:
```bash
./labels.sh 113 114 115
```

### Phase 3 — Calibrate camera rig (once)
```bash
./calibrate.sh 113
```
Builds the fixed camera rig model from one or more DUSt3R caches. Only needs to run once. Add more reference samples for a more stable calibration:
```bash
./calibrate.sh 113 114 115
```

### Phase 4 — Train YOLOv8
```bash
./train.sh
```
Trains YOLOv8n on the exported dataset. Requires at least ~20 annotated samples for meaningful results. Logs to `brio_fast_pipeline/logs/`.

Optional overrides:
```bash
./train.sh --epochs 150 --batch 8
```

### Phase 5 — Run fast inference
```bash
./infer.sh 120
```
Runs the full fast pipeline on sample 120 and writes `predicted.puml` + `results.json` to `brio_fast_pipeline/outputs/sample_120/`. Logs to `brio_fast_pipeline/logs/`.

With a ground-truth PUML (tells the model the expected per-class instance counts):
```bash
./infer.sh 120 --puml /mnt/c/BA/02-resources/data/constructions/Sample_120_InstanceDiagram/InstanceDiagramS120.puml
```

### Visualise slow-pipeline results
```bash
./visualize.sh 113          # auto-selects latest run
./visualize.sh 113 run_002_20260527_0012   # specific run
```
Produces `viz_3d.png` and `viz_2d.png` in the run's sample folder.

---

## Launcher scripts reference

All scripts live in `brio_pipeline/` and change into the correct subdirectory automatically.

| Script | Arguments | What it does |
|--------|-----------|--------------|
| `slow.sh` | `<sample_ids...> [--device cpu]` | Slow pipeline: annotate samples |
| `train_classifier.sh` | `[--epochs N] [--batch N] [--lr F]` | Train the component visual classifier |
| `visualize.sh` | `<sample_id> [<run_name>]` | Plot 3D/2D instance clouds for one sample |
| `labels.sh` | `[<sample_ids...>]` | Export YOLO labels from completed samples |
| `calibrate.sh` | `<sample_ids...>` | Build fixed camera rig calibration |
| `train.sh` | `[--epochs N] [--batch N]` | Train YOLOv8n on exported dataset |
| `infer.sh` | `<sample_id> [--puml <path>]` | Fast inference on one sample |

---

## Directory layout

```
00-project/
│
└── brio_pipeline/              # Both pipelines + launchers
    ├── slow.sh
    ├── train_classifier.sh     # Train component visual classifier
    ├── visualize.sh
    ├── labels.sh
    ├── calibrate.sh
    ├── train.sh
    ├── infer.sh
    ├── README.md               # This file
    ├── CHANGELOG.md            # Change log (reverse chronological)
    ├── 260520-brio-3d-pipeline-setup-guide.md
    │
    ├── brio_3d_pipeline/       # Slow pipeline
    │   ├── pipeline.py         # Entry point
    │   ├── config.py           # Paths and settings
    │   ├── logger.py           # Automatic logging
    │   ├── puml_parser.py
    │   ├── preprocessor.py
    │   ├── dust3r_runner.py
    │   ├── sam_runner.py       # CLAHE pre-processing + SAM
    │   ├── backprojector.py    # Back-projection, clustering, visual votes
    │   ├── classifier.py       # Hungarian assignment (visual + colour)
    │   ├── component_classifier.py  # MobileNetV3-small classifier
    │   ├── component_map.py    # Folder→code mapping + HSV prototypes
    │   ├── component_classifier.pth  # Trained weights (git-ignored)
    │   ├── visualize.py
    │   ├── visualize_2d.py
    │   ├── logs/               # Auto-created; one .log per run
    │   └── outputs/
    │       └── run_NNN_YYYYMMDD_HHMM/   # Timestamped run folder
    │           └── sample_N/
    │               ├── image_order.json
    │               ├── cropped/
    │               ├── dust3r/dust3r_cache.pkl
    │               ├── sam/sam_masks_topN.pkl
    │               ├── proposals/proposals_NX_v4.pkl
    │               ├── results.json
    │               ├── viz_3d.png
    │               └── viz_2d.png
    │
    └── brio_fast_pipeline/     # Fast pipeline
        ├── config.py
        ├── logger.py
        ├── label_exporter.py
        ├── calibrator.py
        ├── detector.py
        ├── triangulator.py
        ├── connector.py
        ├── puml_generator.py
        ├── train.py
        ├── infer.py
        ├── logs/
        ├── calibration/rig_poses.pkl
        ├── dataset/            # YOLO-format training data
        └── outputs/
            └── sample_N/
                ├── predicted.puml
                └── results.json
```

---

## Logging

Every entry-point script calls `setup_logging()` at startup. This redirects all `print()` output and stderr to a timestamped file while keeping terminal output intact.

Log file naming: `YYYYMMDD_HHMMSS_<label>.log`

To follow a run live:
```bash
tail -f brio_3d_pipeline/logs/20260527_001200_samples_113_114.log
```

---

## Table of Contents — Slow Pipeline

1. [Problem Statement](#1-problem-statement)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Stage-by-Stage Walkthrough](#3-stage-by-stage-walkthrough)
   - 3.1 [PUML Parsing](#31-puml-parsing)
   - 3.2 [Image Collection](#32-image-collection)
   - 3.3 [Auto-Crop Preprocessing](#33-auto-crop-preprocessing)
   - 3.4 [DUSt3R — 3D Reconstruction](#34-dust3r--3d-reconstruction)
   - 3.5 [SAM — 2D Segmentation](#35-sam--2d-segmentation)
   - 3.6 [Back-Projection + Clustering](#36-back-projection--clustering)
   - 3.7 [Component Visual Classifier](#37-component-visual-classifier)
   - 3.8 [Classification — Hungarian Assignment](#38-classification--hungarian-assignment)
4. [Output Format](#4-output-format)
5. [What to Expect from Inference](#5-what-to-expect-from-inference)
6. [Known Limitations](#6-known-limitations)
7. [Caching](#7-caching)
8. [Configuration Reference](#8-configuration-reference)
9. [Component Class Vocabulary](#9-component-class-vocabulary)
10. [Environment Setup](#10-environment-setup)
11. [Fast Pipeline — Option C](#11-fast-pipeline--option-c)

---

## 1. Problem Statement

Each BRIO construction sample is a physical arrangement of toy building parts (wooden blocks, rods, plastic plates, nuts, bolts, etc.). For each sample the dataset provides:

- ~78 photographs taken at multiple elevation rings (30°, 45°, 60°, 90°) and 24 azimuth positions per ring.
- A **PlantUML instance diagram** (`.puml`) that declares which component classes are present and how many instances of each, e.g. `object "blwo11_1 : Component"`.

The pipeline's job is to take those photographs and produce a labelled 3D point cloud — one sub-cloud per declared component instance — so that downstream code can verify the reconstruction against the PUML ground truth.

---

## 2. Pipeline Overview

```
PlantUML file
     │
     ▼
[1] PUML Parser ──────────► N components + class list
     │
     ▼
[2] Image Collection ──────► ~20 images per sample
     │                       (all 4 elevation rings, every 4th image)
     ▼
[3] Auto-Crop ─────────────► Fixed-scale half-resolution square crops
     │                       (largest construction in batch sets scale)
     ▼
[4] DUSt3R ────────────────► pts3d[H×W×3] per image (world-space 3D)
     │                       + camera poses + intrinsics
     ▼
[5] SAM ───────────────────► Top-N binary masks per image
     │  (CLAHE contrast boost applied before mask generation)
     ▼
[6] Back-Projection ────────► ~20N raw 3D point clouds
     │  Ward Agglomerative Clustering (6D geometry+colour)
     │  Sigma cleanup (O(n) outlier removal)
     │  Visual classifier → per-mask class prediction
     │  Majority vote per cluster → visual_cls per cluster
     └──────────────────────► N merged, cleaned instance clouds
     │                        + dominant visual class per cluster
     ▼
[7] Visual Classifier ──────► MobileNetV3-small predicts class
     │  (trained on isolated component images, 93.3% val acc)
     ▼
[8] Hungarian Assignment ───► visual mismatch penalty + colour cost
     │                        → class label per cloud
     ▼
results.json  (instance_id, cls, n_points, centroid, bbox_size)
```

---

## 3. Stage-by-Stage Walkthrough

### 3.1 PUML Parsing

**File:** `puml_parser.py`

The PlantUML file for each sample declares what components are physically present:

```
object "blwo11_1 : Component" as blwo111
object "nu_1 : Component" as nu1
object "nu_2 : Component" as nu2
```

`parse_puml()` extracts the class code and instance number from every `object` line via regex, producing a `SampleManifest` with:

- `n_components` — how many physical parts are in this sample (N).
- `components` — ordered list of `Component(cls, instance_id)`.

This manifest drives every subsequent stage: N controls how many SAM masks to keep, how many clusters to produce, and how many assignment slots exist.

---

### 3.2 Image Collection

**File:** `pipeline.py` → `collect_images()`

Every 4th image is taken from each available elevation ring:

| Ring | Typical count | Geometric value |
|------|--------------|-----------------|
| Images30 (30° elevation) | ~6 | Wide-angle side views; good for tall components |
| Images45 (45° elevation) | ~6 | Diagonal views; best overall depth cues |
| Images60 (60° elevation) | ~6 | Near-top views; good for flat components |
| Images90 (top-down) | ~6 | Pure top-down; best XY separation |

**Total: ~20 images per sample** (missing rings are silently skipped). This gives ~190 DUSt3R pairs and provides uniform azimuth coverage at all elevations — critical for correctly reconstructing non-flat constructions.

Always run the full intended sample set together — the pre-pass computes a global pixel scale from all samples, so adding or removing samples changes the crop.

---

### 3.3 Auto-Crop Preprocessing

**File:** `preprocessor.py`

**Step 1 — Largest Connected Component (LCC) detection:**
Each image is binarised (pixels below 245 in all channels = foreground). Morphological closing (15×15 ellipse kernel) fills gaps inside parts. `cv2.connectedComponentsWithStats` finds the largest foreground blob.

**Step 2 — Fixed global scale:**
The pre-pass scans every image across all samples in the batch and finds the largest LCC half-size. All samples are cropped to that same square with 20% padding.

**Step 3 — Half-resolution resize:**
After cropping, each image is downsampled by 50% using `cv2.INTER_AREA`. This halves disk I/O and pre-processing time while preserving enough detail for SAM and DUSt3R.

**Step 4 — Filename collision prevention:**
Images from different elevation folders (e.g. `Images90/IMG_001.jpg` and `Images45/IMG_001.jpg`) share filenames. The output filename is prefixed with the source folder: `Images90_IMG_001.jpg`, `Images45_IMG_001.jpg`.

**Step 5 — Image order preservation:**
The ordered list of cropped paths is written to `image_order.json` so that the visualiser uses exactly the same sequence as SAM and DUSt3R.

This guarantees **1 pixel = the same physical length** across every sample in the batch.

---

### 3.4 DUSt3R — 3D Reconstruction

**File:** `dust3r_runner.py`  
**Model:** `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` (~1.1 GB, downloaded automatically on first use)

DUSt3R is an uncalibrated multi-view stereo model. It predicts dense 3D geometry from image pairs without requiring known camera intrinsics.

**What it produces:**

- **`pts3d`** — the most important output: `(H, W, 3)` per image, each pixel containing its estimated 3D world-space position.
- `poses` — `(N, 4, 4)` camera-to-world matrices.
- `intrinsics` — `(N, 3, 3)` estimated focal lengths and principal points.
- `depths` — `(H, W)` depth maps per image.

**How it works:**
1. All `N*(N-1)/2` image pairs are processed through the pairwise encoder-decoder.
2. `global_aligner` (100 iterations) jointly optimises all poses and depth maps so overlapping regions agree. The loss converges by iteration ~100; additional iterations give negligible improvement.

**VRAM usage:** ViT-L encoder + ~20 images × 190 pairs. `batch_size=1` keeps it within 8 GB.

**Runtime:** ~41 seconds per sample (optimised from ~25 min with 300 iterations). Subsequent runs load from `dust3r_cache.pkl`.

**Key note:** DUSt3R operates in an arbitrary scale and coordinate system. Centroids are in unitless world space, not millimetres. Relative values within one sample are meaningful; absolute values and cross-sample comparisons are not.

---

### 3.5 SAM — 2D Segmentation

**File:** `sam_runner.py`  
**Model:** SAM ViT-B (`sam_vit_b_01ec64.pth`, ~375 MB VRAM)

**CLAHE pre-processing:**
Before SAM sees each image, CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied in LAB colour space: only the L (lightness) channel is equalised (clipLimit=2.0, tile 8×8). This boosts local contrast for white components against white backgrounds without altering colour information.

SAM generates candidate 2D binary masks in **automatic mode**: a 16×16 grid of prompt points is placed over each image and the decoder produces one mask per prompt.

**Filtering:**
- `pred_iou_thresh = 0.80` — minimum model confidence
- `stability_score_thresh = 0.90` — minimum stability under perturbation
- `min_mask_region_area = 200 px` — discard tiny fragments

**Top-N selection:**
After filtering, only the **top N** masks (by predicted IoU) are kept, where N comes from the PUML manifest.

**Runtime:** ~115 seconds per image on GPU. With ~20 images: ~38 minutes — the dominant bottleneck. Results are cached in `sam_masks_topN.pkl`.

---

### 3.6 Back-Projection + Clustering

**File:** `backprojector.py`

**Back-projection:**
For each image and each of its N SAM masks, the mask directly indexes into `pts3d`:

```python
pts = pts3d_frame[mask]   # shape: (M, 3)
```

No camera matrix multiply, no depth conversion — the mask selects pre-computed world-space points. Each cloud is capped at 5,000 points at extraction time.

**Feature vector per cloud:**
```
feature = [centroid_x, centroid_y, centroid_z,   ← 3D median centroid
           H_mean, S_mean, V_mean]                ← mean HSV of mask pixels
```
Both blocks are independently normalised to unit standard deviation.

**Ward Agglomerative Clustering:**
All `~20N` clouds are clustered into exactly **N** groups using `AgglomerativeClustering(linkage="ward")`. Ward linkage minimises within-cluster variance at each merge.

**Sigma cleanup:**
After merging, outlier points are removed: points farther than `mean + 2.5σ` from the cluster median are discarded.

**Visual prediction per mask:**
If a trained `ComponentClassifier` is available, each SAM mask crop is extracted from the CLAHE-enhanced image and classified. Crops smaller than 32×32 px are skipped (too pixelated). Predictions below the confidence threshold (default 0.40) are discarded.

**Majority vote per cluster:**
After clustering, the most common confident visual prediction among all masks in each cluster becomes that cluster's `visual_cls`. This is passed to the assignment stage.

**Output:** N clean float32 arrays `(M_i, 3)` + N visual class codes (or None).

---

### 3.7 Component Visual Classifier

**File:** `component_classifier.py`  
**Dataset:** `new_structure_Component/` — 1,804 isolated component images, 29 classes, 4 elevation sub-folders per class  
**Architecture:** MobileNetV3-small (ImageNet pretrained, fine-tuned)  
**Accuracy:** 93.3% validation accuracy (29-class, held-out 15% split)

The classifier is trained once with `./train_classifier.sh` and its weights (`component_classifier.pth`) are loaded automatically by the pipeline at startup.

**Training details:**
- Heavy augmentation: RandomResizedCrop, horizontal/vertical flip, ColorJitter, label smoothing
- AdamW optimiser, CosineAnnealingLR over 40 epochs
- Input: 224×224 RGB, ImageNet normalisation

**Inference:** Given a BGR crop of a SAM mask region, returns `(cls_code, confidence)`. Used by the back-projector to vote class labels per cluster.

**Small component behaviour:** Components like nuts, washers, and screws are often small in-scene. When the crop is below 32×32 px or confidence is below 0.40, no visual vote is cast — Hungarian assignment handles these via colour prototype fallback.

---

### 3.8 Classification — Hungarian Assignment

**File:** `classifier.py`

The N proposal clouds are matched to the N PUML-declared components in a one-to-one optimal assignment.

**Cost matrix:**

```
cost[i, j] = colour_weight × colour_cost + visual_mismatch_penalty
```

- **Colour cost** — L2 distance between HSV colour prototypes (from `component_map.py`). Prototypes are extracted from the component image dataset.
- **Visual mismatch penalty** — if the visual classifier predicted a class for cluster `i` and it disagrees with manifest entry `j`, a penalty of 8.0 is added. This is the dominant signal when visual predictions are available.

**`scipy.linear_sum_assignment`** (Hungarian algorithm) minimises total cost across all N pairs simultaneously. Assignment decisions are printed per cluster for inspection.

**Output:** N `InstanceResult` objects with `instance_id`, `cls`, `n_points`, `centroid`, `bbox_size`.

---

## 4. Output Format

Each run creates `outputs/run_NNN_YYYYMMDD_HHMM/sample_N/results.json`:

```json
{
  "sample_id": 113,
  "n_components": 5,
  "crop_half": 535,
  "instances": [
    {
      "instance_id": "blwo11_1",
      "cls": "blwo11",
      "n_points": 417105,
      "centroid": [-0.027, 0.016, 0.245],
      "bbox_size": [0.334, 0.395, 0.193]
    }
  ]
}
```

| Field | Meaning |
|-------|---------|
| `instance_id` | Class code + instance number, matching PUML declaration |
| `cls` | Component class code |
| `n_points` | Points in the cleaned 3D cloud |
| `centroid` | Median (x, y, z) in DUSt3R world space |
| `bbox_size` | Axis-aligned bounding box extents (dx, dy, dz) |

Run directories are numbered sequentially (`run_001`, `run_002`, …). The visualiser auto-selects the most recent run when no `--run` argument is given.

---

## 5. What to Expect from Inference

### Typical run (cold — no cache)

| Stage | Time (GPU, ~20 images) |
|-------|-----------------------|
| Pre-pass crop | ~2 min |
| DUSt3R | ~41 sec |
| SAM | ~38 min |
| Back-projection + clustering + visual classification | ~2 min |
| Hungarian assignment | < 1 sec |
| **Total per sample** | **~43 min** |

### With cache

DUSt3R and SAM caches are reused on re-runs. A cached run (back-projector + classifier only) takes under 2 minutes per sample.

### Result quality

**Good separation** (geometrically distinct components like plates + rods + nuts):
- Each instance cloud has a clearly distinct centroid position.
- `n_points` varies by size — a large plate might have 2–3M points; a small nut 100–200k.
- Visual classifier confirms the assignment with high confidence for large/distinctive parts.

**Difficult cases:**
- Components of the same class stacked or touching — Ward clustering may merge them.
- Small components (nuts, washers, screws < 32 px crop) — visual classifier abstains; colour fallback applies.
- Components with very similar colour — colour cost contributes little; visual vote dominates.
- White components on white backgrounds — mitigated by CLAHE, but may still produce weak SAM masks.

### What the 3D positions mean

Centroid coordinates are in DUSt3R's arbitrary world-space (not metres). Do not compare centroid values across samples.

---

## 6. Known Limitations

| Limitation | Effect | Status |
|------------|--------|--------|
| Single-sample scale | Crop scale differs from batch run | Always run the full intended sample set together |
| No metric scale | Centroid/bbox units are DUSt3R arbitrary | Cannot directly compare sizes across samples |
| SAM top-N assumption | Assumes the N best SAM masks correspond to the N physical components | Fails if one component produces no confident mask |
| Small component visual classification | Crops < 32 px skip classifier; confidence < 0.40 casts no vote | Fallback to colour prototype; generally correct for fasteners since they're often unique in the manifest |
| Domain gap (classifier) | Trained on isolated white-background images; inference on scene crops | Mitigated by CLAHE and heavy augmentation; 93.3% val acc in practice |

---

## 7. Caching

Every expensive stage writes a cache file. The pipeline checks for these at startup and skips the stage if they exist.

| Cache file | Stage | Notes |
|------------|-------|-------|
| `cropped/half_N_r50.txt` | Preprocessor | Regenerate if sample batch or resolution changes |
| `dust3r/dust3r_cache.pkl` | DUSt3R | Tied to the cropped images; delete if crops change |
| `sam/sam_masks_topN.pkl` | SAM | N baked into filename; component count change triggers re-run |
| `proposals/proposals_NX_v4.pkl` | Back-projection (no visual) | Delete to force re-cluster |
| `proposals/proposals_NX_v4_vis.pkl` | Back-projection (with visual) | Separate cache when classifier is active |

To clear all caches for a sample:
```bash
rm -rf brio_3d_pipeline/outputs/run_NNN_YYYYMMDD_HHMM/sample_114/
```

---

## 8. Configuration Reference

`brio_3d_pipeline/config.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `DUST3R_SIZE` | `512` | Max image dimension fed to DUSt3R encoder |
| `DUST3R_NITER` | `100` | Global alignment iterations (loss converges by ~100) |
| `DUST3R_BATCH` | `1` | Pairs per forward pass — keep at 1 for 8 GB VRAM |
| `SAM_MODEL_TYPE` | `"vit_b"` | SAM encoder size |
| `SAM_POINTS_SIDE` | `16` | Grid density for automatic mask generation |
| `SAM_IOU_THRESH` | `0.80` | Minimum SAM confidence to keep a mask |
| `SAM_STABILITY` | `0.90` | Minimum stability score |
| `SAM_MIN_AREA` | `200` | Minimum mask area in pixels |
| `AUTO_CROP_PADDING` | `0.20` | Padding around the LCC bounding box |
| `DEVICE` | `"cuda"` | Set to `"cpu"` as fallback |
| `COMPONENT_DATASET` | *(OneDrive path)* | Root of the isolated component image dataset |
| `CLASSIFIER_WEIGHTS` | `component_classifier.pth` | MobileNetV3-small checkpoint |
| `CLASSIFIER_CONF_THRESH` | `0.40` | Minimum confidence to trust a visual prediction |

`brio_fast_pipeline/config.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `YOLO_BASE_MODEL` | `"yolov8n.pt"` | Pretrained COCO weights to fine-tune from |
| `YOLO_EPOCHS` | `100` | Default training epochs |
| `YOLO_BATCH` | `16` | Training batch size |
| `YOLO_IMGSZ` | `640` | Training image size |
| `TRAIN_VAL_SPLIT` | `0.85` | Fraction of samples used for training |
| `CONF_THRESHOLD` | `0.25` | Minimum YOLO confidence for a detection |
| `MIN_VIEWS` | `2` | Minimum views required to triangulate an instance |
| `CONNECTION_DIST_THRESH` | `0.12` | Max 3D distance to consider two components connected |

---

## 9. Component Class Vocabulary

29 physical component types:

| Folder name | Class code | Description |
|-------------|-----------|-------------|
| `blockwood11` | `blwo11` | Wooden block 1×1 |
| `blockwood21` | `blwo21` | Wooden block 2×1 |
| `bolt` | `bo` | Metal bolt |
| `nose` | `no` | Nose connector |
| `nut` | `nu` | Metal nut |
| `plateplastic53` | `plpl53` | Plastic plate 5×3 |
| `platewood21` | `plwo21` | Wooden plate 2×1 |
| `platewood31` | `plwo31` | Wooden plate 3×1 |
| `platewood33` | `plwo33` | Wooden plate 3×3 |
| `platewood53` | `plwo53` | Wooden plate 5×3 |
| `plug` | `pl` | Plug connector |
| `rodlong` | `rolo` | Long rod |
| `rodmedium` | `rome` | Medium rod |
| `rodsmall` | `rosm` | Short rod |
| `screwlong` | `sclo` | Long screw |
| `screwmedium` | `scme` | Medium screw |
| `screwsmall` | `scsm` | Small screw |
| `sleeve` | `sl` | Sleeve connector |
| `strapplastic5` | `stpl5` | Plastic strap length 5 |
| `strapwood3` | `stwo3` | Wooden strap length 3 |
| `strapwood4` | `stwo4` | Wooden strap length 4 |
| `strapwood5` | `stwo5` | Wooden strap length 5 |
| `strapwood6` | `stwo6` | Wooden strap length 6 |
| `strapwood7` | `stwo7` | Wooden strap length 7 |
| `strapwood9` | `stwo9` | Wooden strap length 9 |
| `tire` | `ti` | Rubber tire |
| `washer` | `wa` | Metal washer |
| `wheelred` | `whre` | Red wheel |
| `wheelwhite` | `whwh` | White wheel |

---

## 10. Environment Setup

```bash
conda create -n brio-3d python=3.10 -y
conda activate brio-3d

# PyTorch — use cu124, not cu130 (driver reports CUDA 12.7 but cu130 fails)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# DUSt3R dependencies
pip install -r /mnt/c/BA/07-dust3r/requirements.txt

# SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Core dependencies
pip install opencv-python scikit-learn scipy matplotlib pillow

# Fast pipeline (YOLO)
pip install ultralytics

# Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**SAM weights** must be downloaded separately:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O /mnt/c/BA/03-code/sam_weights/sam_vit_b_01ec64.pth
```

**DUSt3R model weights** download automatically from HuggingFace on first use (~1.1 GB).

**Component classifier weights** are produced by running `./train_classifier.sh` once (~10 min).

**Always use the direct Python binary** — not `conda run`. The launcher scripts do this automatically. If running Python directly:
```bash
/home/mxrn/miniconda3/envs/brio-3d/bin/python
```

---

## 11. Fast Pipeline — Option C

The fast pipeline trains a YOLOv8 detector on the slow pipeline's YOLO labels and performs sub-second inference using a fixed camera rig calibration for 3D triangulation.

### Architecture

```
Training time (runs once):
  labels.sh     →  brio_fast_pipeline/dataset/   (YOLO format)
  train.sh      →  brio_fast_pipeline/runs/detect/.../best.pt

Calibration time (runs once):
  calibrate.sh  →  brio_fast_pipeline/calibration/rig_poses.pkl

Inference time (< 2 seconds per sample):
  ~20 images
    ↓  detector.py     — YOLOv8 detection (~700ms)
    ↓  triangulator.py — DLT triangulation (~5ms)
    ↓  connector.py    — proximity + slot rules (~1ms)
    ↓  puml_generator.py (~1ms)
  brio_fast_pipeline/outputs/sample_N/predicted.puml
  brio_fast_pipeline/outputs/sample_N/results.json
```

### File reference

| File | Role |
|------|------|
| `config.py` | All paths, class vocabulary, slot compatibility rules |
| `logger.py` | Automatic timestamped logging |
| `label_exporter.py` | Converts slow pipeline outputs → YOLO training labels |
| `calibrator.py` | Builds fixed camera rig from DUSt3R caches |
| `detector.py` | YOLOv8 inference wrapper |
| `train.py` | YOLOv8n training entry point |
| `triangulator.py` | DLT triangulation: YOLO 2D boxes × rig poses → 3D |
| `connector.py` | 3D proximity + slot compatibility rules → connection edges |
| `puml_generator.py` | Instance list + edges → PlantUML source |
| `infer.py` | End-to-end fast inference entry point |

### How occlusion is handled

A component occluded in most views is triangulated from the views where it is visible. The DLT solver uses all views that detect the class and finds the least-squares best 3D position. Even two non-adjacent views give an accurate triangulation because the camera geometry is known precisely from the fixed rig.

This is fundamentally stronger than monocular depth estimation (Option B): depth from a single image is ambiguous, especially for the flat-background BRIO setup.

### Switching to Option B later

Options B and C share the same YOLO detector — no re-training or re-labelling required to evaluate Option B. To try monocular depth lifting:
1. `pip install depth-anything-v2`
2. Create `triangulator_mono.py`
3. Compare 3D accuracy against the slow pipeline's ground-truth centroids
