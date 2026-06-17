# BRIO Pipeline — Change Log

Changes are listed in reverse chronological order. Each entry records the date, the files touched, and what was changed and why.

---

## 2026-06-12 (continued)

### Whole-object mask decomposition, 3D-overlap grouping, geometric coaxial merge, size-prior assignment
**Files:** `brio_3d_pipeline/sam_runner.py`, `brio_3d_pipeline/backprojector.py`, `brio_3d_pipeline/classifier.py`, `brio_3d_pipeline/config.py`, `brio_3d_pipeline/pipeline.py`, `brio_3d_pipeline/visualize_2d.py`, `brio_3d_pipeline/dust3r_runner.py`

Five fixes following the run_018/019/020 evaluation (sample 19 fully correct; sample 27 block under-covered; sample 113 grouping/labels still wrong):

- **`sam_runner.py` — mask subtraction**: oversized on-foreground masks are no longer dropped (that erased the wood block in sample 27, where one part dominates the silhouette). They are decomposed instead: subtract the kept smaller masks, keep the remainder as the dominant component. Single-component fallback keeps the largest oversized mask. Up to `SAM_KEEP_FACTOR`×N masks per view are now kept so thin/low-scoring components survive selection. Cache suffix `_filt` → `_filt2`.

- **Thin-structure recall (`config.py`)**: `SAM_POINTS_SIDE` 16→32 (a 16-grid put a prompt every ~14 px — the ~6 px rod in sample 113 was never hit), `SAM_IOU_THRESH` 0.80→0.70, `SAM_STABILITY` 0.90→0.85, `SAM_MIN_AREA` 200→80.

- **`backprojector.py` — 3D-overlap graph grouping**: replaces Ward clustering on [centroid, HSV] (which split the sample-113 plate across two clusters). Masks are linked iff their 3D clouds share ≥`OVERLAP_MIN` of voxels (voxel = scene diagonal / `OVERLAP_VOXEL_DIV`); union-find components are merged down to N by max overlap / nearest centroid. Proposals cache → `_v7`.

- **`backprojector.py` — geometric coaxial merge**: the visual-class equality gate is gone (the classifier rarely fires). Fragments merge iff both are elongated (PCA σ1/σ2 > `COAXIAL_MIN_ELONG`), axes aligned within `COAXIAL_ANGLE_DEG`, lateral axis offset < `COAXIAL_MAX_OFFSET_M` (rejects parallel rods), and the axial gap < `MAX_ROD_GAP_M` and < 50 % of the span.

- **`classifier.py` — size prior**: new `_nominal_size()` decodes class codes into stud units (blwo21 = 2, plpl53 = 5, stwo9 = 9; digit-free codes from a lookup table). Cost adds `size_weight × |rel_obs − rel_nom|` with both extents normalised by the in-sample maximum, so DUSt3R's arbitrary scale cancels. Cluster colours are now computed from RAW images (CLAHE skewed them relative to the raw-image prototypes; CLAHE remains for classifier crops only).

- **`dust3r_runner.py` — VRAM release**: DUSt3R model/scene freed (`torch.cuda.empty_cache()`) before SAM runs in the same process. Root cause of a 20+ min SAM hang at 7.9/8.2 GB VRAM.

---

## 2026-06-12

### SAM background-mask rejection + DUSt3R confidence filtering
**Files:** `brio_3d_pipeline/sam_runner.py`, `brio_3d_pipeline/dust3r_runner.py`, `brio_3d_pipeline/backprojector.py`, `brio_3d_pipeline/pipeline.py`, `brio_3d_pipeline/visualize.py`, `brio_3d_pipeline/visualize_2d.py`, `brio_3d_pipeline/config.py`

Root-cause fix for background masking and noisy 3D clouds. The old SAM stage kept the top-N masks by `predicted_iou`, which systematically selected the background/whole-scene masks (verified on run_017: in every view the #1-ranked mask was 2.4–3.8× the foreground area). All downstream clustering and labeling was unrecoverable from that input.

- **`sam_runner.py`**: Added `_foreground_mask()` (largest connected non-white region, same `BG_THRESHOLD=245` logic as the preprocessor, computed on the pre-CLAHE image) and `_filter_masks()` which rejects masks that are (1) larger than `SAM_MAX_AREA_FRAC` of the foreground (whole-object/scene masks), (2) less than `SAM_FG_OVERLAP_MIN` inside the foreground (background/table), (3) touching the image border, or (4) near-duplicates (IoU > `SAM_DEDUP_IOU`, keeping the higher predicted IoU). Top-N selection now happens *after* filtering. Cache filename gains a `_filt` suffix to invalidate old caches.

- **`dust3r_runner.py`**: Result dict now includes `confs` (per-image DUSt3R confidence maps from `scene.im_conf`). Old caches missing `confs` are regenerated.

- **`backprojector.py`**: `_extract_mask_cloud()` drops points below `DUST3R_CONF_THRESH` (textureless/misaligned background producing floating outliers). Proposals cache bumped to `_v6`.

- **`visualize.py`**: Scene cloud now filters by DUSt3R confidence and drops near-white background/padding pixels — the 3D view shows only the object.

- **`visualize_2d.py`**: Prefers the new `_filt` SAM cache over legacy unfiltered ones in resumed run dirs.

- **`config.py`**: Added `SAM_MAX_AREA_FRAC` (0.50), `SAM_FG_OVERLAP_MIN` (0.60), `SAM_DEDUP_IOU` (0.80), `DUST3R_CONF_THRESH` (2.0; DUSt3R demo default is 3.0 — raise if clouds stay noisy).

Verified against run_017 sample 113 cached masks: the filters reject 24/100 previously-kept masks, including the giant background mask in all 20 views.

---

## 2026-05-26 (continued)

### Component visual classifier (Option A)
**Files:** `brio_3d_pipeline/component_classifier.py` *(new)*, `brio_3d_pipeline/config.py`, `brio_3d_pipeline/backprojector.py`, `brio_3d_pipeline/classifier.py`, `brio_3d_pipeline/pipeline.py`, `brio_3d_pipeline/component_map.py`, `train_classifier.sh` *(new)*

The slow pipeline's component assignment was producing inconsistent labels because the cost matrix in `classifier.py` was effectively broken: `g_cost` was always 0 (comparing a proposal to itself), and `c_cost` computed the norm of a prototype rather than the distance to it. To fix this, a visual classifier was added.

- **`component_classifier.py`** (new): MobileNetV3-small fine-tuned on the isolated component dataset (`new_structure_Component`, 1,804 images, 29 classes). Contains `ComponentDataset` loading from all elevation sub-folders and root-level images, `train_classifier()` with heavy augmentation and CosineAnnealingLR, and `ComponentClassifier` inference class (`predict(bgr)` → `(cls_code, confidence)`). Also provides `crop_from_mask()` utility that extracts, pads, and squares a SAM mask bounding box.

- **`config.py`**: Added `COMPONENT_DATASET`, `CLASSIFIER_WEIGHTS`, and `CLASSIFIER_CONF_THRESH` paths/settings.

- **`backprojector.py`**: `compute_proposals()` now accepts an optional `classifier` and `conf_thresh`. For each SAM mask it extracts a crop via `crop_from_mask()` and runs the classifier. After agglomerative clustering, it majority-votes the per-mask predictions within each cluster to produce `visual_cls_per_cluster`. Return type changed from `list[np.ndarray]` to `tuple[list[np.ndarray], list[str | None]]`. Cache key bumped to `_v4` (and `_v4_vis` when classifier is active) to invalidate old caches.

- **`classifier.py`**: `assign_classes()` now accepts `visual_cls: list[str | None]`. The broken cost matrix was replaced: colour distance is now a proper L2 between proposal and manifest prototype HSV vectors; when a visual prediction is available, any manifest entry that disagrees incurs `visual_mismatch_penalty` (default 8.0) — making the visual vote the dominant signal. Assignment decisions are printed per cluster for debugging.

- **`pipeline.py`**: Added `_load_classifier()` which loads weights if present and logs a message otherwise (graceful degradation). `process_sample()` now accepts and forwards the classifier. In `main()`, the classifier is loaded once and shared across all samples. `compute_proposals` return is now unpacked as `(proposals, visual_cls)`.

- **`component_map.py`**: Fixed typo `"ro_lo"` → `"rolo"` for `rodlong` to match actual PUML class codes.

- **`train_classifier.sh`** (new): One-command launcher for classifier training via `brio-3d` conda env.

**To train:** `./train_classifier.sh` (40 epochs, ~5–10 min on RTX 2070 Super). Weights are saved to `brio_3d_pipeline/component_classifier.pth` and loaded automatically on the next pipeline run.

---

## 2026-05-26

### CLAHE contrast enhancement before SAM
**Files:** `brio_3d_pipeline/sam_runner.py`

White BRIO components were invisible to SAM because they blend into the white background. Added a `_clahe_enhance()` pre-processing step that converts each cropped image to LAB colour space, applies Contrast Limited Adaptive Histogram Equalization (CLAHE, clipLimit=2.0, tile 8×8) to the L (lightness) channel only, then converts back to BGR before SAM processes it. Colour information is untouched; only local luminance contrast is boosted. The CLAHE object is created once at module load and reused across all images.

---

### DUSt3R iteration cap and image stride reduction
**Files:** `brio_3d_pipeline/config.py`, `brio_3d_pipeline/pipeline.py`

GPU temperature was reaching 87–90 °C and each sample was projected to take 20–25 minutes due to the large number of image pairs and alignment iterations.

- **`DUST3R_NITER` 300 → 100** (`config.py`): The loss curve converged to ~0.006 by iteration ~100 in practice; the remaining 200 iterations yielded negligible improvement. Cutting to 100 reduced DUSt3R global alignment from ~25 min to ~41 s per sample.
- **Image stride 3 → 4** (`pipeline.py`, `collect_images()`): Reduces images per sample from ~27 to ~20 and pairs from 162 to 120, cutting pairwise inference time by ~26 %.

---

### Timestamped run directories
**Files:** `brio_3d_pipeline/pipeline.py`, `brio_3d_pipeline/visualize_2d.py`, `brio_3d_pipeline/visualize.py`, `visualize.sh`

Outputs were all written to a flat `outputs/sample_N/` structure, making it impossible to distinguish results from different runs.

- Added `_make_run_dir()` to `pipeline.py`: counts existing `run_*` directories under `OUTPUTS_ROOT` to assign a zero-padded run number, then creates `run_NNN_YYYYMMDD_HHMM/` (e.g. `run_001_20260526_2156`).
- `process_sample()` now writes all outputs under `run_dir/sample_N/` instead of `OUTPUTS_ROOT/sample_N/`.
- Both visualizers (`visualize_2d.py`, `visualize.py`) gained a `--run` CLI argument. When omitted, they auto-select the most recently modified `run_*` folder via `_resolve_run_dir()`.
- `visualize.sh` forwards an optional second positional argument as `--run`.

---

### SAM mask duplication fix + image ordering
**Files:** `brio_3d_pipeline/preprocessor.py`, `brio_3d_pipeline/pipeline.py`, `brio_3d_pipeline/visualize_2d.py`

Two bugs caused SAM masks to appear duplicated or misaligned across views in the 2D visualisation:

1. **Filename collision** (`preprocessor.py`): Images from different elevation folders (e.g. `Images90/IMG_001.jpg` and `Images45/IMG_001.jpg`) shared the same filename. Writing them all to `output_dir/p.name` caused overwrites — later images silently replaced earlier ones. Fixed by prefixing the output filename with the source folder: `Images90_IMG_001.jpg`, `Images45_IMG_001.jpg`, etc. Cache marker renamed to `half_{N}_r50.txt` to force invalidation of old full-resolution caches.

2. **Ordering mismatch** (`visualize_2d.py`): The visualiser re-sorted cropped files alphabetically from disk, which did not match the order SAM and DUSt3R used (pipeline collection order). Fixed by saving the ordered list of cropped paths to `image_order.json` in `process_sample()` immediately after cropping. The visualiser loads this file and uses it directly, falling back to alphabetical sort only when the file is absent.

---

### All elevation folders + stride-3 image sampling
**Files:** `brio_3d_pipeline/pipeline.py`

The previous `collect_images()` only sampled from `Images90` (all images) and `Images45` (every 4th). This missed geometric information from the 30° and 60° elevation rings.

- `collect_images()` now iterates all four elevation folders: `Images30`, `Images45`, `Images60`, `Images90`. Missing folders are silently skipped.
- Every 3rd image is taken from each folder (`ring[::3]`), giving uniform azimuth coverage per elevation ring.
- The per-sample image count printout was made dynamic to show counts per folder.

---

### Half-resolution cropped images
**Files:** `brio_3d_pipeline/preprocessor.py`

Cropped images at full resolution (`2N × 2N` px) were larger than necessary, increasing disk I/O and pre-processing time.

- After computing the fixed crop with `_crop_image_fixed()`, the output is resized by 50 % using `cv2.INTER_AREA` before saving, producing `N × N` px crops.
- Cache marker renamed from `half_{N}.txt` to `half_{N}_r50.txt` so that existing full-resolution caches are automatically detected as stale and regenerated.
- The stale-cache cleanup loop was extended to also remove `.JPG` files (previously only `.jpg` was cleared).
