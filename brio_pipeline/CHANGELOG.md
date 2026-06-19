# BRIO Pipeline — Change Log

Changes are listed in reverse chronological order. Each entry records the date, the files touched, and what was changed and why.

---

## 2026-06-19 — Documentation: full README rewrite with per-file walkthrough

### Files touched
- `brio_3d_pipeline/README.md`

### Changes

Complete rewrite of the pipeline README. The new version covers every file in `brio_3d_pipeline/` with:
- Line-by-line explanation of each function's role and logic
- Code snippets for all key operations (foreground detection, grid prompting, voxel overlap, cost matrix, elongation, amodal extension)
- Concrete data-flow example walking through sample 113 from raw images to the wrong assignment
- Failure-mode table cataloguing all 10 known root causes of wrong masking and assignment
- Output structure and cache versioning guide

---

## 2026-06-18 — Classifier: elongation feature + DBSCAN determinism + blwo11 nominal fix

### Files touched
- `brio_3d_pipeline/backprojector.py`
- `brio_3d_pipeline/classifier.py`
- `brio_3d_pipeline/pipeline.py`

### Changes

**DBSCAN determinism** (`backprojector.py`):
- `_dbscan_cleanup()` now seeds the subsampler with `np.random.default_rng(seed=42)` instead of the global state. Previously, every cache miss gave different point subsets → different bbox sizes → different classifier cost matrices. Cost matrices are now stable across regenerated proposals.
- Proposals cache bumped from `_v10` → `_v12` (v11 = intermediate run without the seed fix).

**Geometric elongation as classifier feature** (`backprojector.py`, `classifier.py`, `pipeline.py`):
- `_compute_elongation(pts)`: new helper in backprojector; returns PCA σ₁/σ₂ of a cluster (1 = sphere, >5 = rod-like).
- `compute_proposals()` now returns a 4-tuple `(clouds, visual_cls, cluster_colours, cluster_elongations)`. Cache version bumped accordingly.
- `_NOMINAL_ELONG` dict in classifier.py: nominal PCA elongation for each BRIO class (rod≈6, screw≈2.5, block≈1.5, plate≈1.67, nut≈1.2, etc.). For digit-encoded classes not in the dict, the aspect ratio of the two largest digits is used (e.g. plpl53 → 5/3≈1.67).
- `_nominal_elong(cls)` function: looks up `_NOMINAL_ELONG`, falls back to digit-ratio.
- `assign_classes()`: new `cluster_elongations` and `elong_weight=1.0` params. Adds log-scale cost `elong_weight × |log(obs_elong) − log(nom_elong)|` to each cell. Log scale treats a rod-vs-block mismatch (log(6)−log(1.5)=1.4) much more severely than plate-vs-block (log(1.67)−log(1.5)=0.11), naturally penalising wrong class–cluster pairings without needing to tune absolute thresholds.

**blwo11 nominal size fix** (`classifier.py`):
- Added `"blwo11": 9.0, "blwo21": 7.0` to `_NOMINAL_NO_DIGITS`. The digit parser previously returned `max(1,1)=1` for "blwo11", causing the large windmill block to look tiny to the size metric and always lose to rods/plates. The nominal now reflects its physical footprint (large cross-shaped block > any rod).
- `colour_weight` default: 1.5 → 2.5 (stronger colour signal to help distinguish blwo11 vs rome when both are near-white).

### Why
Run_022 regression analysis showed that with better reconstruction (32 multi-elevation views, finer voxels), the blwo11 windmill cross-block in sample 113 now correctly accumulates the most points (253K) and has a clear 3D shape — but the classifier still assigned it to `rome_1`. Root causes:

1. `_nominal_size("blwo11")=1.0` (digit bug): even with 253K pts, blwo11 scored as the smallest nominal. Fixed by explicit dict entry.
2. Non-deterministic DBSCAN: bbox changed between runs, making the cost matrix unstable. Fixed by seeding np.random.
3. Through-joint occlusion: the rome rod passes entirely through blwo11, so its 3D cluster is compact (elong≈1.24) rather than elongated (expected ≈6). The elongation signal penalises assigning any low-elong cluster to a rod class, but since no cluster in sample 113 has high elong, the signal cannot determine where the rod went. This is a known fundamental limitation: heavily occluded through-joint rods cannot be distinctly reconstructed without depth priors or DINOv2 visual classification.

### Known limitation (sample 113)
The plpl53 plate has a larger 3D bbox than the blwo11 block in the reconstruction (DUSt3R scale is arbitrary and the flat plate spans a wider 3D extent). With `bbox_max` as the size metric and `blwo11=9.0` (largest nominal), the plate always grabs the blwo11 slot with size_cost=0 (perfect match). Neither colour nor elongation can overcome the 0-cost size advantage. The correct fix is DINOv2 visual classification (planned). Switching to `sqrt(n_pts)` obs_ext would fix sample 113 but breaks sample 114 (the stwo3 strap, which has the most points, incorrectly maps to plwo53, the largest nominal).

---

## 2026-06-17 — Occlusion & grouping improvements (run_022+)

### Files touched
- `brio_3d_pipeline/config.py`
- `brio_3d_pipeline/sam_runner.py`
- `brio_3d_pipeline/backprojector.py`
- `brio_3d_pipeline/puml_parser.py`
- `brio_3d_pipeline/pipeline.py`

### Changes

**voxel-overlap grouping refinements** (`backprojector.py`, `config.py`):
- `OVERLAP_VOXEL_DIV` 40 → 80: smaller voxels so touching components (rod surface vs block face) no longer share the same hash bucket, reducing false inter-component links.
- `OVERLAP_MIN` 0.20 → 0.15: compensates for the reduced per-voxel overlap that comes with finer voxels; intra-component cross-view links stay strong.
- `COLOR_DIST_WEIGHT = 0.3`: adds an HSV repulsion term to the agglomerative distance matrix (`dist += color_weight * L2_HSV`). Differently-coloured touching parts (e.g. blue plate against white block) are now penalised from merging.
- Proposals cache bumped from `_v9` to `_v10` so old caches are not reused.

**Co-axial merge threshold** (`config.py`):
- `COAXIAL_MIN_ELONG` 1.8 → 1.5: lets borderline rod fragments (short visible stubs with lower elongation) qualify for the merge.

**Contrast enhancement** (`sam_runner.py`, `backprojector.py`, `config.py`):
- `CLAHE_CLIP_LIMIT = 3.0` (was 2.0): stronger local contrast for thin rods and small components. Both sam_runner and backprojector read from config so they stay in sync.

**Adaptive image stride** (`pipeline.py`, `config.py`):
- `IMAGE_VIEWS_PER_RING = 8`: `collect_images()` now uses `stride = max(1, len(ring)//8)` per ring. Images90 (only 8 images) previously gave 2 top-down views at stride-4; now gives all 8. Total views per sample: ~32 (up from ~20). Requires a fresh DUSt3R run.

**PUML through-joint connection parsing** (`puml_parser.py`):
- `SampleManifest.thjo_pairs`: new field listing `(rod_instance_id, [through_instance_ids])` pairs, parsed from `&{thjo}` / `&{op}` Connection objects in the PUML.

**Amodal bbox for through-joint rods** (`pipeline.py`):
- `_amodal_extend_rods()`: post-assignment step that, for each thjo rod, merges the 3D clouds of the rod + all threaded components, finds the rod axis via PCA, then extends the AABB by `ROD_PROTRUSION_M = 0.030 m` on each end. Fixes the partial-rod bbox that results when only one rod end is reconstructed well.

### Why
Through-joint rods (e.g. `rome_1` in sample 113) were receiving only ~8 700 pts in a scattered, non-elongated cloud (elong=1.25) because:
1. Rod-end voxels and block-face voxels shared the same 5 mm hash bucket (fixed by OVERLAP_VOXEL_DIV=80).
2. Only 2/8 top-down images were used (fixed by adaptive stride).
3. The co-axial merge threshold was too strict (fixed by COAXIAL_MIN_ELONG=1.5).
4. Even when reconstruction is still partial, the amodal extension recovers a correct 3D bbox using the block's cloud + rod axis.

---

## 2026-06-17 — README, .gitignore — outputs tracking
**Files:** `brio_3d_pipeline/README.md` *(new)*, `.gitignore`

- **`README.md`** (new): full pipeline documentation — stage-by-stage explanation of preprocessor, DUSt3R, SAM prompted mode, backprojector, and classifier; how-to-run commands; output directory structure; config table; known limitations.

- **`.gitignore`**: removed blanket `outputs/` exclusion. The `outputs/` directory is now tracked. Only `**/dust3r/` within outputs remains excluded (`dust3r_cache.pkl` is 76 MB per sample, near GitHub's 100 MB hard limit and a pure regeneratable cache).

---

## 2026-06-16

### SAM prompted mode — tight crop + black background + foreground-only grid prompts
**Files:** `brio_3d_pipeline/sam_runner.py`, `brio_3d_pipeline/pipeline.py`

Root cause of the joint-mask problem in auto mode: SAM's uniform 32-pt grid placed prompts on 2D boundaries between touching components, producing masks that spanned two instances simultaneously. These bridge masks then merged the corresponding 3D clusters in the voxel-overlap graph, causing incorrect group assignments. No amount of post-hoc filtering could recover them because the masks were internally coherent from SAM's perspective.

Fix: switch to `SamPredictor` (prompt-based) as the default mode. For each image:

- **`_tight_fg_crop()`** (new helper): crops the fixed-scale image to the tight bounding box of the foreground mask (+15 px padding). SAM's 1024×1024 encoder now sees only the construction, eliminating wasted capacity on white background and padding.

- **Black background**: `crop_bgr[~foreground] = 0`. Every pixel outside the foreground mask is set to pure black before SAM encoding. Background is completely removed (vs. grey-fill in auto mode).

- **`_fg_grid_points()`** (new helper): places a regular grid only inside the foreground mask at spacing `sqrt(fg_area / (N × 2.5))`, producing ~2.5 × N prompts total. Zero prompts on background pixels, unlike the old 32-pt uniform grid which covered the full fixed-scale crop including white margins.

- **`_expand_masks()`** (new helper): translates masks from tight-crop coordinates back to fixed-scale image coordinates using the stored `(x_off, y_off)` offsets, so the backprojector can index them against DUSt3R `pts3d` without modification.

- **`_run_sam_prompted()`** (new): orchestrates the above — one `predictor.set_image()` encoder pass per image, then one lightweight decoder pass per prompt point. Each prompt returns 3 mask candidates; highest-confidence one is kept. Same `_filter_masks()` post-filter pipeline as auto mode is applied after coordinate expansion.

- **`run_sam_on_images()`**: added `mode` parameter (`"prompted"` default, `"auto"` for original behaviour). Prompted mode cache file uses suffix `_prompted_filt` to avoid conflicts with auto-mode caches.

- **`pipeline.py`**: added `--sam-mode` CLI flag (`prompted` / `auto`), forwarded to `run_sam_on_images()` via `process_sample()`.

**Measured improvement (run_021, samples 113–114)**:
- SAM timing: ~0.5–0.8 s/image (vs. ~2–5 s auto mode).
- Sample 113: all 5 components correctly found — blwo11_1 (226 926 pts), rome_1 (8 718), plpl53_1 (5 022), nu_1 (2 451), nu_2 (607).
- Sample 114: 6/7 components correct — co-axial merge correctly fired on two stwo3 fragments; nu_1 received 0 pts (vacated cluster from the merge, known limitation).

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
