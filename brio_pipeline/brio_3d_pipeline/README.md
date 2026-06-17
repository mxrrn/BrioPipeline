# BRIO 3D Pipeline

Multi-view instance segmentation pipeline for BRIO construction sets.  
Turns 20 multi-view images of a construction into labelled 3D point clouds, one per component.

---

## How to run

```bash
conda activate brio-3d
cd /mnt/c/BA/00-project/brio_pipeline/brio_3d_pipeline

# Single sample
python pipeline.py --samples 113

# Multiple samples
python pipeline.py --samples 113 114 115

# SAM auto mode (original uniform grid, slower)
python pipeline.py --samples 113 --sam-mode auto

# Visualise results
python visualize_2d.py --sample 113
python visualize_2d.py --sample 113 --run run_021_20260616_2339  # specific run
```

Outputs are written to `outputs/run_NNN_YYYYMMDD_HHMM/sample_NNN/`.

---

## Pipeline overview

```
multi-view images (20 per sample)
        │
        ▼  preprocessor.py
Fixed-scale crop centred on the construction
        │
        ▼  dust3r_runner.py     ← GPU, ~10–40 min per sample
Per-pixel 3D world coordinates (pts3d) for all 20 views
        │
        ▼  sam_runner.py        ← GPU, ~0.5–0.8 s per image (prompted mode)
Instance masks per image (N × keep_factor masks per image)
        │
        ▼  backprojector.py
3D point cloud per SAM mask → voxel-overlap grouping → N instance clouds
        │
        ▼  classifier.py
Hungarian assignment: each cloud → PUML component instance
        │
        ▼  results.json + visualize_2d.py
```

---

## Stage-by-stage explanation

### 1. Preprocessor (`preprocessor.py`)

The raw multi-view images show the construction on a white table.

- Detects the **largest connected foreground component** (LCC) in each image: any pixel with all channels below 245 is foreground.
- Computes the LCC centroid and bounding-box half-size.
- Across all samples in the current run, picks the **largest** half-size as the global crop window so that 1 px = the same physical length for every sample.
- Crops a square of that fixed size centred on each image's LCC. Overflow outside the image edge is padded with white (255).
- Downscales to half resolution.

**Result**: square images where the construction is centred, white margins around it. The background is tightly framed but not removed yet.

---

### 2. DUSt3R (`dust3r_runner.py`)

All 20 cropped images are passed through DUSt3R, which estimates:
- Per-pixel 3D world coordinates (`pts3d`, shape `H × W × 3`) for every view.
- A per-pixel confidence map.

DUSt3R operates directly on image pairs; it does not need pre-calibrated cameras. Results are cached to disk so the stage is skipped on subsequent runs of the same sample.

**Settings** (`config.py`): `DUST3R_SIZE=512`, `DUST3R_NITER=100`, `DUST3R_BATCH=1` (8 GB VRAM constraint).

---

### 3. SAM — prompted mode (`sam_runner.py`)

Default mode. Runs `SamPredictor` (prompt-based) rather than `SamAutomaticMaskGenerator` (uniform grid). For each of the 20 images:

1. **Foreground mask** — re-detects the LCC inside the already-cropped image.

2. **Tight crop** (`_tight_fg_crop`) — further crops to the tight bounding box of the foreground mask (+15 px padding). SAM's 1024×1024 encoder now sees only the construction, not white margins.

3. **Black background** — `crop_bgr[~foreground] = 0`. Every background pixel is set to pure black. The background is completely removed at this point.

4. **CLAHE enhancement** — local contrast boost in LAB space, applied only to foreground pixels, so white BRIO parts stand out.

5. **Foreground-only grid prompts** (`_fg_grid_points`) — a regular grid placed only inside the foreground mask at spacing `sqrt(fg_area / (N × 2.5))`, giving ~2.5 × N points total. Zero prompts on background pixels.

6. **One encoder pass**, then one lightweight decoder pass per prompt point. Each prompt returns 3 mask candidates; the highest-confidence one is kept.

7. **Mask coordinate translation** (`_expand_masks`) — translates masks from tight-crop space back to fixed-scale image coordinates so they align with the DUSt3R `pts3d` arrays.

8. **Filter pipeline** (shared with auto mode):
   - Drop masks with < 60 % overlap with the foreground.
   - Drop masks touching the image border.
   - Decompose oversized masks by subtracting smaller kept masks.
   - Deduplicate: if two masks have IoU > 0.80, drop the lower-confidence one.
   - Prune container masks (a mask largely covered by smaller ones is a multi-part union).
   - Split disconnected blobs within one mask into separate candidates.

9. Keep the top `N × keep_factor` masks per image (default `keep_factor=2`).

**Amodal note**: rods with an occluded middle produce two separate SAM masks (one per exposed end). These are intentionally kept and merged in the backprojector stage via co-axial merge.

**Auto mode** (`--sam-mode auto`): uses `SamAutomaticMaskGenerator` with a 32-point uniform grid over the full fixed-scale crop (background set to grey). Slower (~2–5 s/image) and more prone to joint masks where two touching components share a boundary.

---

### 4. Backprojector + 3D grouping (`backprojector.py`)

For every SAM mask, across all 20 images:
- Index into `pts3d` using the boolean mask → 3D point cloud for that mask.
- Drop points below the DUSt3R confidence threshold (`DUST3R_CONF_THRESH=2.0`).
- Subsample to 5 000 points per mask to bound memory.

**Voxel-overlap grouping**:
- Discretise all mask clouds into voxels (voxel size = scene bbox diagonal / 40).
- Build a graph: two masks link if their voxel sets overlap by ≥ 20 % (intersection / smaller set).
- Union-find over this graph → connected components = initial instance groups.
- Merge or pad to reach exactly N groups (one per PUML component).

**DBSCAN cleanup** per group: removes outlier noise using `eps = 5 % × bounding-box diagonal`.

**Co-axial merge**: detects rod fragments split by occlusion. Two groups merge if:
- Both have PCA elongation ≥ 1.8 (rod-like shape).
- Their principal axes are within 20° of parallel.
- Their axis-to-axis lateral offset is < 12 mm.
- The end-to-end gap is < 120 mm.

Per group output: 3D point cloud, centroid, bounding-box size, mean HSV colour.

---

### 5. Classifier (`classifier.py`)

Assigns each 3D group to a PUML-declared component instance using **Hungarian optimal assignment** (`scipy.linear_sum_assignment`). Cost per (group, component) pair:

| Term | Weight | How it's computed |
|------|--------|------------------|
| Colour | 1.5 | L2 distance between observed mean HSV of the group and the class HSV prototype (built from `02-resources/data/component_images/`) |
| Size | 1.0 | \|relative observed 3D extent − relative nominal class size\|, both normalised by the largest value in the sample |
| Visual penalty | 8.0 | Added when the ComponentClassifier (MobileNetV3) predicts a class that disagrees with this manifest entry |

The assignment is globally optimal — no greedy local decisions.

---

## Output structure

```
outputs/
└── run_021_20260616_2339/
    └── sample_113/
        ├── cropped/              # fixed-scale crops (one per input image)
        ├── dust3r/               # DUSt3R results (pts3d, conf, cached)
        ├── sam/                  # SAM masks (cached .pkl)
        ├── proposals/            # backprojector intermediate data
        ├── image_order.json      # ordered list of cropped image paths
        ├── results.json          # final per-component assignment
        └── viz_2d.png            # visualisation grid (one row per component)
```

---

## Configuration (`config.py`)

| Key | Value | Meaning |
|-----|-------|---------|
| `DUST3R_SIZE` | 512 | Image resize for DUSt3R encoder |
| `DUST3R_NITER` | 100 | Global alignment iterations |
| `DUST3R_BATCH` | 1 | Pairs per forward pass (8 GB VRAM limit) |
| `DUST3R_CONF_THRESH` | 2.0 | Minimum per-pixel DUSt3R confidence |
| `SAM_MODEL_TYPE` | vit_b | SAM backbone (fits in 8 GB alongside DUSt3R) |
| `SAM_KEEP_FACTOR` | 2 | Keep up to 2×N masks per view |
| `SAM_FG_OVERLAP_MIN` | 0.60 | Drop masks with < 60 % foreground overlap |
| `SAM_DEDUP_IOU` | 0.80 | IoU threshold for duplicate mask removal |
| `OVERLAP_MIN` | 0.20 | Minimum voxel-overlap ratio to link two masks |
| `COAXIAL_MIN_ELONG` | 1.8 | Minimum PCA elongation to consider a group rod-like |
| `COAXIAL_ANGLE_DEG` | 20.0 | Max axis angle (°) for co-axial merge |
| `MAX_ROD_GAP_M` | 0.12 | Max end-to-end gap (m) for co-axial merge |

---

## Known limitations

- **nu_1 / empty cluster after co-axial merge**: when the co-axial merge absorbs a fragment cluster, the vacated cluster slot gets assigned to the next-smallest component (often a nut) with 0 points. Fix: remove empty clusters from the assignment pool before Hungarian matching.
- **DUSt3R runtime**: 10–40 min per sample on an RTX 2070 Super. Results are cached so re-runs are fast, but the first pass is slow.
- **White plastic components**: CLAHE boosts local contrast but colour-based assignment is weak for white-on-white (e.g. distinguishing two white block sizes). Visual classifier weights not currently loaded.
