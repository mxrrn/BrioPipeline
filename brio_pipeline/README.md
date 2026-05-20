# BRIO Pipeline — Documentation

This directory contains two pipelines and a set of launcher scripts:

| Pipeline | Purpose | Runtime |
|---|---|---|
| `brio_3d_pipeline/` | Slow annotation pipeline. Produces 3D ground-truth labels and training data. | ~35 min/sample (cold) |
| `brio_fast_pipeline/` | Fast inference. Trained YOLOv8 + fixed-rig triangulation. | < 2 seconds/sample |

The slow pipeline runs once per sample to produce training labels. The fast pipeline trains on those labels and then runs on unseen samples.

**Logs are written automatically** on every run — no manual redirection needed. Both pipelines write timestamped `.log` files to their own `logs/` folder.

---

## Quick-start

All phases are launched from `brio_pipeline/` using the shell scripts below. Run them from that directory:

```bash
cd /mnt/c/BA/00-project/brio_pipeline
```

### Phase 1 — Annotate samples (slow pipeline)
```bash
./slow.sh 113 114 115 116 117
```
Runs DUSt3R + SAM + clustering on the listed samples. Logs to `brio_3d_pipeline/logs/`.

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
./visualize.sh 113
```
Produces `brio_3d_pipeline/outputs/sample_113/viz_3d.png`.

---

## Launcher scripts reference

All scripts live in `brio_pipeline/` and change into the correct subdirectory automatically.

| Script | Arguments | What it does |
|--------|-----------|--------------|
| `slow.sh` | `<sample_ids...> [--device cpu]` | Slow pipeline: annotate samples |
| `visualize.sh` | `<sample_id>` | Plot 3D instance clouds for one sample |
| `labels.sh` | `[<sample_ids...>]` | Export YOLO labels from completed samples |
| `calibrate.sh` | `<sample_ids...>` | Build fixed camera rig calibration |
| `train.sh` | `[--epochs N] [--batch N]` | Train YOLOv8n on exported dataset |
| `infer.sh` | `<sample_id> [--puml <path>]` | Fast inference on one sample |

---

## Directory layout

```
00-project/
│
├── brio_3d_pipeline/   # Slow pipeline (WSL-accessible at this level)
│   ├── pipeline.py     # Entry point
│   ├── config.py       # Paths and settings
│   ├── logger.py       # Automatic logging
│   ├── puml_parser.py
│   ├── preprocessor.py
│   ├── dust3r_runner.py
│   ├── sam_runner.py
│   ├── backprojector.py
│   ├── classifier.py
│   ├── component_map.py
│   ├── visualize.py
│   ├── logs/           # Auto-created; one .log per run
│   └── outputs/
│       └── sample_N/
│           ├── cropped/
│           ├── dust3r/dust3r_cache.pkl
│           ├── sam/sam_masks_topN.pkl
│           ├── proposals/proposals_NX_v3.pkl
│           ├── results.json
│           └── viz_3d.png
│
└── brio_pipeline/      # Fast pipeline + launchers
    ├── slow.sh             # Launcher: slow annotation pipeline
    ├── visualize.sh        # Launcher: 3D visualisation
    ├── labels.sh           # Launcher: YOLO label export
    ├── calibrate.sh        # Launcher: camera rig calibration
    ├── train.sh            # Launcher: YOLOv8 training
    ├── infer.sh            # Launcher: fast inference
    ├── README.md           # This file
    ├── 260520-brio-3d-pipeline-setup-guide.md
    │
    └── brio_fast_pipeline/ # Fast pipeline
    ├── config.py
    ├── logger.py       # Automatic logging
    ├── label_exporter.py
    ├── calibrator.py
    ├── detector.py
    ├── triangulator.py
    ├── connector.py
    ├── puml_generator.py
    ├── train.py
    ├── infer.py
    ├── logs/           # Auto-created; one .log per run
    ├── calibration/rig_poses.pkl
    ├── dataset/        # YOLO-format training data
    └── outputs/
        └── sample_N/
            ├── predicted.puml
            └── results.json
```

---

## Logging

Every entry-point script (`pipeline.py`, `label_exporter.py`, `calibrator.py`, `train.py`, `infer.py`) calls `setup_logging()` at startup. This redirects all `print()` output and stderr to a timestamped file while keeping terminal output intact.

Log file naming: `YYYYMMDD_HHMMSS_<label>.log`

Examples:
```
brio_3d_pipeline/logs/20260520_142301_samples_113_114_115.log
brio_fast_pipeline/logs/20260520_222902_calibrate.log
brio_fast_pipeline/logs/20260521_093012_infer_sample120.log
```

To follow a run live:
```bash
tail -f brio_3d_pipeline/logs/20260520_142301_samples_113_114_115.log
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
   - 3.7 [Classification — Hungarian Assignment](#37-classification--hungarian-assignment)
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
[1] PUML Parser ──────► N components + class list
     │
     ▼
[2] Image Collection ──► 14 images per sample
     │                   (8 × Images90 + 6 × Images45)
     ▼
[3] Auto-Crop ──────────► Fixed-scale 2N×2N px square crops
     │                    (largest construction in batch sets scale)
     ▼
[4] DUSt3R ─────────────► pts3d[H×W×3] per image (world-space 3D)
     │                    + camera poses + intrinsics
     ▼
[5] SAM ────────────────► Top-N binary masks per image
     │
     ▼
[6] Back-Projection ────► 14×N raw 3D point clouds
     │  Ward Agglomerative Clustering (6D geometry+colour)
     │  Sigma cleanup (O(n) outlier removal)
     └──────────────────► N merged, cleaned instance clouds
     │
     ▼
[7] Classifier ─────────► Hungarian assignment → class label per cloud
     │
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
...
```

`parse_puml()` extracts the class code and instance number from every `object` line via regex, producing a `SampleManifest` with:

- `n_components` — how many physical parts are in this sample (N).
- `components` — ordered list of `Component(cls, instance_id)`.

This manifest drives every subsequent stage: N controls how many SAM masks to keep, how many clusters to produce, and how many assignment slots exist.

**What can go wrong:** If the PUML uses a class code not in the vocabulary (e.g. `stwo3` vs `stwo3-9`), the colour lookup in the classifier will return zeros for that class. The geometry assignment still works; only colour-guided cost is weakened.

---

### 3.2 Image Collection

**File:** `pipeline.py` → `collect_images()`

| Ring | Images used | Reason |
|------|-------------|--------|
| Images90 (top-down) | All 8 available | Best XY separation — sees the footprint of every component from above |
| Images45 (45° elevation) | Every 4th → ~6 images | Adds Z/height information; full 360° azimuth coverage with minimal overlap |

**Total: ~14 images per sample.** Using all 78 images would produce 78×77/2 = 3,003 DUSt3R pairs and take hours. 14 images gives 91 pairs and runs in ~4 minutes.

---

### 3.3 Auto-Crop Preprocessing

**File:** `preprocessor.py`

**Step 1 — Largest Connected Component (LCC) detection:**
Each image is binarised (pixels below 245 in all channels = foreground). Morphological closing (15×15 ellipse kernel) fills gaps inside parts. `cv2.connectedComponentsWithStats` finds the largest foreground blob.

**Step 2 — Fixed global scale:**
The pre-pass scans every image across all samples in the batch and finds the largest LCC half-size. All samples are cropped to that same square with 20% padding.

This guarantees **1 pixel = the same physical length** across every sample in the batch. Without it, a small 3-part construction would appear at 3× the zoom of a large 7-part one, making DUSt3R depth estimates and SAM mask areas incomparable.

**Always run the full intended sample set together** — running a single sample alone gives a different pixel scale from a batch run.

**Output:** `cropped/half_N.txt` marker + one JPG per image at `2N × 2N` px.

---

### 3.4 DUSt3R — 3D Reconstruction

**File:** `dust3r_runner.py`  
**Model:** `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` (~1.1 GB, downloaded automatically on first use)

DUSt3R is an uncalibrated multi-view stereo model. It predicts dense 3D geometry from image pairs without requiring known camera intrinsics.

**What it produces:**

- **`pts3d`** — the most important output: `(H, W, 3)` per image, each pixel containing its estimated 3D world-space position. A dense point cloud aligned across all views.
- `poses` — `(N, 4, 4)` camera-to-world matrices.
- `intrinsics` — `(N, 3, 3)` estimated focal lengths and principal points.
- `depths` — `(H, W)` depth maps per image.

**How it works:**
1. All `N*(N-1)/2` image pairs are processed through the pairwise encoder-decoder.
2. `global_aligner` (300 iterations) jointly optimises all poses and depth maps so overlapping regions agree.

**VRAM usage:** ViT-L encoder + 14 images × 91 pairs is the tightest stage. `batch_size=1` keeps it within 8 GB.

**Runtime:** ~4 minutes (first run). Subsequent runs load from `dust3r_cache.pkl`.

**Key note:** DUSt3R operates in an arbitrary scale and coordinate system. Centroids are in unitless world space, not millimetres. Relative values within one sample are meaningful; absolute values and cross-sample comparisons are not.

---

### 3.5 SAM — 2D Segmentation

**File:** `sam_runner.py`  
**Model:** SAM ViT-B (`sam_vit_b_01ec64.pth`, ~375 MB VRAM)

SAM generates candidate 2D binary masks in **automatic mode**: a 16×16 grid of prompt points is placed over each image and the decoder produces one mask per prompt.

**Filtering:**
- `pred_iou_thresh = 0.80` — minimum model confidence
- `stability_score_thresh = 0.90` — minimum stability under perturbation
- `min_mask_region_area = 200 px` — discard tiny fragments

**Top-N selection:**
After filtering, only the **top N** masks (by predicted IoU) are kept, where N comes from the PUML manifest. This is the pipeline's only use of the ground-truth label count.

**Runtime:** ~115 seconds per image on GPU. With 14 images: ~27 minutes — the dominant bottleneck. Results are cached in `sam_masks_topN.pkl`.

---

### 3.6 Back-Projection + Clustering

**File:** `backprojector.py`

**Back-projection:**
For each image and each of its N SAM masks, the mask directly indexes into `pts3d`:

```python
pts = pts3d_frame[mask]   # shape: (M, 3)
```

No camera matrix multiply, no depth conversion — the mask selects pre-computed world-space points. Each cloud is capped at 5,000 points at extraction time to bound memory usage.

**Feature vector per cloud:**
```
feature = [centroid_x, centroid_y, centroid_z,   ← 3D median centroid
           H_mean, S_mean, V_mean]                ← mean HSV of mask pixels
```
Both blocks are independently normalised to unit standard deviation.

**Ward Agglomerative Clustering:**
All `14N` clouds are clustered into exactly **N** groups using `AgglomerativeClustering(linkage="ward")`. Ward linkage minimises within-cluster variance at each merge — a globally optimal assignment that correctly groups cross-view observations of the same component even when components are touching.

**Sigma cleanup:**
After merging, outlier points are removed using a O(n) sigma filter: points farther than `mean + 2.5σ` of the distance distribution from the cluster median are discarded. This replaces DBSCAN, which caused multi-GB OOM on large merged clusters in WSL2.

**Output:** N clean float32 arrays `(M_i, 3)` — one per declared component instance.

---

### 3.7 Classification — Hungarian Assignment

**File:** `classifier.py`  
**Reference colour data:** `component_map.py`

The N proposal clouds are matched to the N PUML-declared components in a one-to-one assignment.

**Cost matrix:**

```
cost[i, j] = geometry_cost + 1.5 × colour_cost
```

- **Geometry cost** — L2 distance between the proposal's geometry feature vector and the class prior.
- **Colour cost** — L2 distance between the proposal's mean HSV and the reference colour prototype for class `j`. Prototypes are extracted from `02-resources/data/component_images/`.

**`scipy.linear_sum_assignment`** (Hungarian algorithm) minimises total cost across all N pairs simultaneously.

**Output:** N `InstanceResult` objects with `instance_id`, `cls`, `n_points`, `centroid`, `bbox_size`.

---

## 4. Output Format

`brio_3d_pipeline/outputs/sample_N/results.json`:

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

---

## 5. What to Expect from Inference

### Typical run (cold — no cache)

| Stage | Time (GPU, 14 images) |
|-------|-----------------------|
| Pre-pass crop | ~2 min |
| DUSt3R | ~4 min |
| SAM | ~27 min |
| Back-projection + clustering | ~1 min |
| Classification | < 1 sec |
| **Total per sample** | **~35 min** |

### With cache

DUSt3R and SAM caches are reused on re-runs. A cached run (backprojector + classifier only) takes under 2 minutes per sample.

### Result quality

**Good separation** (geometrically distinct components like plates + rods + nuts):
- Each instance cloud has a clearly distinct centroid position.
- `n_points` varies by size — a large plate might have 2–3M points; a small nut 100–200k.

**Difficult cases:**
- Components of the same class stacked or touching — Ward clustering may merge them.
- Small components partially occluded in all views — SAM may filter their masks.
- Components with very similar colour prototypes — colour cost contributes little.

### What the 3D positions mean

Centroid coordinates are in DUSt3R's arbitrary world-space (not metres). Do not compare centroid values across samples — the coordinate system is re-initialised for each run.

---

## 6. Known Limitations

| Limitation | Effect | Status |
|------------|--------|--------|
| `stwo3` vs `stwo3-9` class code mismatch | Colour prototype lookup returns zeros for that class | Needs PUML regex / component_map alignment |
| Single-sample scale | Crop scale differs from batch run | Always run the full intended sample set together |
| Duplicate image filenames | Images90 and Images45 share filenames; one overwrites the other in cropped/ | Manifests as DUSt3R seeing the same image twice |
| No metric scale | Centroid/bbox units are DUSt3R arbitrary | Cannot directly compare sizes across samples |
| SAM top-N assumption | Assumes the N best SAM masks correspond to the N physical components | Fails if one component produces no confident mask |
| `conda run` overhead | OOM kill on WSL2 | Fixed: launcher scripts use direct Python binary |
| Large cluster OOM | DBSCAN on million-point clusters caused multi-GB RSS spikes | Fixed: sigma cleanup + 5k-per-cloud cap |

---

## 7. Caching

Every expensive stage writes a cache file. The pipeline checks for these at startup and skips the stage if they exist.

| Cache file | Stage | Notes |
|------------|-------|-------|
| `cropped/half_N.txt` | Preprocessor | Regenerate if sample batch changes |
| `dust3r/dust3r_cache.pkl` | DUSt3R | Tied to the cropped images; delete if crop changes |
| `sam/sam_masks_topN.pkl` | SAM | N baked into filename; component count change triggers re-run |
| `proposals/proposals_NX_v3.pkl` | Back-projection | Delete to force re-cluster |

To clear all caches for a sample:
```bash
rm -rf brio_3d_pipeline/outputs/sample_114/
```

---

## 8. Configuration Reference

`brio_3d_pipeline/config.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `DUST3R_SIZE` | `512` | Max image dimension fed to DUSt3R encoder |
| `DUST3R_NITER` | `300` | Global alignment iterations |
| `DUST3R_BATCH` | `1` | Pairs per forward pass — keep at 1 for 8 GB VRAM |
| `SAM_MODEL_TYPE` | `"vit_b"` | SAM encoder size |
| `SAM_POINTS_SIDE` | `16` | Grid density for automatic mask generation |
| `SAM_IOU_THRESH` | `0.80` | Minimum SAM confidence to keep a mask |
| `SAM_STABILITY` | `0.90` | Minimum stability score |
| `SAM_MIN_AREA` | `200` | Minimum mask area in pixels |
| `AUTO_CROP_PADDING` | `0.20` | Padding around the LCC bounding box |
| `DEVICE` | `"cuda"` | Set to `"cpu"` as fallback |

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
| `rodlong` | `ro_lo` | Long rod |
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
pip install opencv-python scikit-learn scipy matplotlib

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
  14 images
    ↓  detector.py    — YOLOv8 detection (~700ms)
    ↓  triangulator.py — DLT triangulation (~5ms)
    ↓  connector.py   — proximity + slot rules (~1ms)
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

This is fundamentally stronger than monocular depth estimation (Option B): depth from a single image is ambiguous, especially for the flat-background BRIO setup. Known multi-view geometry is exact.

### Switching to Option B later

Options B and C share the same YOLO detector — no re-training or re-labelling required to evaluate Option B. To try monocular depth lifting:
1. `pip install depth-anything-v2`
2. Create `triangulator_mono.py`
3. Compare 3D accuracy against the slow pipeline's ground-truth centroids
