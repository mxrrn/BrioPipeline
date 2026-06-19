# BRIO 3D Pipeline

Multi-view instance segmentation pipeline for BRIO construction sets.  
Given ~32 photographs of a BRIO assembly taken from four elevation rings, it produces one labelled 3D point cloud per component, sized and centred in world space, matching the component list declared in the PUML instance diagram.

---

## Quick start

```bash
conda activate brio-3d
cd /mnt/c/BA/00-project/brio_pipeline/brio_3d_pipeline

# Single sample
python pipeline.py --samples 113

# Multiple samples
python pipeline.py --samples 113 114 115

# Specific device
python pipeline.py --samples 113 --device cpu

# Resume a run (reuses cached DUSt3R and SAM outputs)
python pipeline.py --samples 113 --resume run_021_20260616_2339

# SAM auto mode (uniform grid, slower, less precise)
python pipeline.py --samples 113 --sam-mode auto

# 2D verification overlay
python visualize_2d.py --sample 113
python visualize_2d.py --sample 113 --run run_021_20260616_2339

# 3D point-cloud export
python visualize.py --sample 113
```

Outputs land in `outputs/run_NNN_YYYYMMDD_HHMM/sample_NNN/`.  
Every slow stage (DUSt3R, SAM, proposals) is cached to disk and skipped on re-runs.

---

## Pipeline at a glance

```
02-resources/data/multi_view_images/Sample_<N>/Images{30,45,60,90}/
  ~32 raw JPEG photos from four elevation rings
        │
        ▼  puml_parser.py
Read component manifest from InstanceDiagramS<N>.puml
  → class list, instance IDs, through-joint pairs
        │
        ▼  preprocessor.py
Fixed-scale crop centred on the construction (largest-LCC + 20% padding)
  → half-resolution square crops, white margins, consistent across all samples
        │
        ▼  dust3r_runner.py                        [GPU – cached]
DUSt3R global alignment on all cropped images
  → pts3d (H×W×3) per view — world-space 3D coordinates for every pixel
  → confs (H×W) per view — per-pixel reconstruction confidence
        │
        ▼  sam_runner.py                           [GPU – cached]
SAM SamPredictor: tight-crop → black background → CLAHE → fg-grid prompts
  → top-(N × keep_factor) masks per view, in fixed-scale image coordinates
        │
        ▼  backprojector.py                        [CPU – cached]
For each SAM mask: index pts3d → 3D point cloud
Voxel-overlap agglomerative clustering (avg-linkage, colour-repulsion)
  → N merged instance clouds
DBSCAN noise removal per cloud
Co-axial rod-fragment merge (occluded rod stubs)
PCA elongation per cloud
        │
        ▼  classifier.py
Hungarian optimal assignment of N clouds to N PUML components
  cost = colour_weight×HSV_dist + size_weight×|rel_obs−rel_nom|
       + elong_weight×|log(obs_elong)−log(nom_elong)|
       + visual_mismatch_penalty (if MobileNetV3 is loaded)
        │
        ▼  pipeline.py (amodal extension)
Extend through-joint rod bboxes along the rod PCA axis
        │
        ▼  results.json
Per-component: instance_id, n_points, centroid (3,), bbox_size (3,)
```

---

## File-by-file reference

### `config.py` — Central settings

All hard-coded constants live here. Nothing else imports paths or thresholds directly.

```python
BA_ROOT       = Path("/mnt/c/BA")
SAM_WEIGHTS   = BA_ROOT / "03-code/sam_weights/sam_vit_b_01ec64.pth"
DATA_ROOT     = BA_ROOT / "02-resources/data"
MULTI_VIEW    = DATA_ROOT / "multi_view_images"
OUTPUTS_ROOT  = PROJECT_ROOT / "outputs"
```

**Key tunable parameters:**

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| `DUST3R_SIZE` | 512 | Image resize for DUSt3R encoder — larger = better geometry, more VRAM |
| `DUST3R_NITER` | 100 | Global alignment iterations — higher = more accurate, slower |
| `DUST3R_CONF_THRESH` | 2.0 | Drop pts3d below this confidence (noisy/textureless regions) |
| `SAM_POINTS_SIDE` | 32 | Grid density in auto mode |
| `SAM_KEEP_FACTOR` | 2 | Keep up to `2 × N` masks per image |
| `SAM_FG_OVERLAP_MIN` | 0.60 | Drop masks with < 60 % pixels inside the detected foreground |
| `SAM_DEDUP_IOU` | 0.80 | Masks with pairwise IoU > 0.80 are duplicates — keep higher-confidence one |
| `OVERLAP_MIN` | 0.15 | Min voxel-overlap ratio to link two mask clouds in the grouping graph |
| `OVERLAP_VOXEL_DIV` | 80 | Voxel size = scene bbox diagonal / 80 (~3–5 mm for typical assemblies) |
| `COLOR_DIST_WEIGHT` | 0.3 | HSV repulsion in voxel-overlap grouping (0 = disabled) |
| `COAXIAL_MIN_ELONG` | 1.5 | Min PCA σ₁/σ₂ for a cluster to qualify as a rod fragment |
| `COAXIAL_ANGLE_DEG` | 20.0 | Max angle (°) between two rod fragments' axes to merge them |
| `MAX_ROD_GAP_M` | 0.12 | Max end-to-end gap (m) for co-axial merge |
| `CLAHE_CLIP_LIMIT` | 3.0 | CLAHE contrast boost (applied in LAB space before SAM) |
| `ROD_PROTRUSION_M` | 0.030 | Estimated rod protrusion beyond each through-joint block (for amodal bbox) |
| `IMAGE_VIEWS_PER_RING` | 8 | Target views sampled from each elevation ring |

---

### `puml_parser.py` — Read the ground-truth manifest

**Input**: `InstanceDiagramS<N>.puml`  
**Output**: `SampleManifest(components, thjo_pairs)`

Every sample has a PUML instance diagram that declares which components are in the construction. The parser extracts two things:

**Component list** — matched by this regex:
```python
comp_pattern = re.compile(r'object "(\w+)_(\d+)\s*:\s*Component"')
# "blwo11_1 : Component"  →  Component(cls="blwo11", instance_id="blwo11_1")
# "rome_1 : Component"    →  Component(cls="rome",   instance_id="rome_1")
```
The result is `manifest.components`, a list of `Component` objects. The count `manifest.n_components` drives how many clusters the entire pipeline must produce — it is used by SAM (keep N × factor masks), by the backprojector (target N clusters), and by the Hungarian assignment (N × N cost matrix).

**Through-joint pairs** — matched by a second regex that finds connections with the `{thjo}` slot tag:
```python
conn_pattern = re.compile(
    r'object\s+"(\w+_\d+)&\{(\w+)\}_\d+#(\w+_\d+)&\{(\w+)\}_\d+\s*:\s*Connection"'
)
# "blwo11_1&{op}_1#rome_1&{thjo}_1 : Connection"
# → thjo_pairs = [("rome_1", ["blwo11_1"])]
```
`thjo_pairs` is consumed at the very end of the pipeline by `_amodal_extend_rods` to extend rod bounding boxes through the blocks they pass through.

```python
@dataclass
class SampleManifest:
    sample_id:  int
    components: list[Component]    # ordered list, one per BRIO piece
    thjo_pairs: list               # [(rod_instance_id, [through_id, ...])]
```

**Failure mode**: If the PUML uses a non-standard connection syntax, `thjo_pairs` is silently empty and amodal extension is skipped without any warning.

---

### `preprocessor.py` — Fixed-scale crop

**Input**: raw multi-view JPEGs  
**Output**: square half-resolution crops, consistent across all samples

The fundamental problem this solves: DUSt3R's reconstructed scale is arbitrary per sample, but the entire pipeline treats `obs_ext` (max bounding-box dimension) as a relative size proxy. To make size comparisons meaningful, every sample must be cropped so that the same physical length maps to the same number of pixels.

**Step 1 — foreground detection** (`_lcc_info`):
```python
BG_THRESHOLD = 245   # pixels above this in all channels = background (white table)

mask = np.any(img < BG_THRESHOLD, axis=2).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# connectedComponentsWithStats → take the largest component (LCC)
```
Returns the LCC centroid `(cx, cy)` and tight bounding-box half-sizes `(half_w, half_h)`.

**Step 2 — global half-size** (`compute_global_halfsize`):
```python
max_half = max(half_w, half_h)   # across ALL images of ALL samples in the run
padded   = int(np.ceil(max_half * (1 + padding)))   # padding=0.20 → +20%
```
Every sample in the batch uses this same `padded` value as the crop radius. A sample with a smaller construction gets white-filled margins around it, ensuring all crops have identical pixel-to-metre ratios.

**Step 3 — crop and downscale** (`crop_images`):
```python
cropped = _crop_image_fixed(img, cx, cy, fixed_half)   # 2×fixed_half square
cropped = cv2.resize(cropped, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
```
The output is `fixed_half × fixed_half` pixels (half-resolution saves VRAM during DUSt3R and speeds up SAM without much accuracy loss). White (255) is used for padding, matching the background used in the reference images.

**Failure mode — BG_THRESHOLD on near-white pieces**: BRIO blocks (`blwo11`) and some rods are nearly white. After crop-and-pad, these pieces can merge visually with the white margin. If the LCC centroid is pulled toward the coloured pieces, the near-white piece may be partially outside the crop window, silently cutting off part of the construction.

---

### `dust3r_runner.py` — Dense 3D reconstruction

**Input**: list of cropped image paths  
**Output**: `dict` with `pts3d`, `confs`, `poses`, `intrinsics`, `depths`

DUSt3R is a feed-forward stereo model that produces per-pixel 3D coordinates without calibrated cameras. It is the foundation that all downstream 3D operations depend on.

```python
pairs  = make_pairs(images, scene_graph="logwin", prefilter=None, symmetrize=True)
output = inference(pairs, model, device, batch_size=1, verbose=False)
scene  = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
scene.compute_global_alignment(init="mst", niter=100, schedule="cosine", lr=0.01)

pts3d = [p.detach().cpu().numpy() for p in scene.get_pts3d()]    # list of (H, W, 3)
confs = [c.detach().cpu().numpy() for c in scene.im_conf]        # list of (H, W)
```

`pts3d[i][r, c]` is the 3D world-space coordinate for pixel `(r, c)` of image `i`. This is later indexed directly by SAM boolean masks — no manual K-matrix or depth unprojection is needed.

**`logwin` scene graph**: For N images, logwin creates a sliding window of pairwise connections with logarithmic spacing, giving O(N log N) pairs. This balances reconstruction quality against computation time for 32 images.

**Confidence maps**: `confs[i]` is a scalar map in [0, ∞). Low confidence indicates textureless or poorly-matched regions. The pipeline uses `DUST3R_CONF_THRESH=2.0` to mask these out before clustering — they would otherwise produce floating 3D outlier points from background white areas.

**VRAM management**: After the alignment is complete, the model and scene are explicitly freed before SAM runs:
```python
del scene, output, model
torch.cuda.empty_cache()
```
This is critical on the 8 GB RTX 2070 Super — without it, SAM competes for the same VRAM and either OOMs or runs at 20-minute per-image speeds.

**Caching**: Results are pickled to `dust3r/dust3r_cache.pkl`. DUSt3R takes 10–40 minutes for a 32-image run, so the cache is the primary reason `--resume` is useful.

**Failure modes**:
- **Arbitrary scale**: The reconstruction scale is different for every sample and even between runs on the same sample (global alignment can converge to a different scale). All downstream features are therefore normalised per-sample, but this normalisation assumes the ranking of sizes is preserved — which fails when a flat plate reconstructs with a larger 3D bbox than a compact block.
    - Clarification: when DUST3R reconstructs a scene from images, it does not know what "10cm" is. It only sees pixel relationships between images. The output cloud lives in some abstract coordinate space where 1 unit could be 10cm or 10m. Two runs of the same sample can produce two differently scaled output point clouds - which remains valid because the global alignment optimisation (the iterative step that stitches all pairwise predictions into one coordinate frame) can converge to a different scale depending on which pairs were processed, the random initialisation, and numerical noise
    - Normalizing the sizes across samples will therefore maintain the sizes of a component occuring in a construction which is especially useful if we have more than one occurance of the component in a certain sample
- **niter=100**: Lower than the DUSt3R demo default (~300). Convergence may be incomplete on assemblies with many occluded regions, producing noisier point clouds.
- **logwin misses some pairs**: If a rod only appears clearly in 2 images that happen to not be paired in logwin, its geometry is weakly constrained and may reconstruct poorly.

---

### `sam_runner.py` — 2D instance mask generation

**Input**: cropped image paths, `n_components`  
**Output**: `list[list[dict]]` — one inner list per image, each dict has `segmentation (H×W bool)`, `area`, `predicted_iou`, `image_idx`

This is the most critical stage for downstream correctness. Every 3D cluster is entirely built from the 2D masks generated here.

#### Prompted mode (default)

For each image, the flow is:

**1. Foreground mask** (`_foreground_mask`):
```python
mask = np.any(bgr < BG_THRESHOLD, axis=2).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# → largest connected component, then dilate
```
Same `BG_THRESHOLD=245` logic as preprocessor. This is a coarse mask of "where the construction is" in the already-cropped image. It drives both the tight-crop step and the grid-prompt placement.

**2. Tight crop** (`_tight_fg_crop`):
```python
y0 = max(0, ys.min() - pad)   # pad=15 px
y1 = min(H, ys.max() + pad)
crop_bgr = bgr[y0:y1, x0:x1]
```
Crops to the tight bounding box of the foreground, so SAM's 1024×1024 encoder is entirely spent on the construction rather than white margins.

**3. Black background**:
```python
crop_bgr[~crop_fg] = 0   # pure black for all non-foreground pixels
```
Eliminates background ambiguity completely. SAM has no white pixels to generate background masks from.

**4. CLAHE** (`_clahe_enhance`):
```python
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])   # CLAHE_CLIP_LIMIT=3.0
```
Local contrast boost in the L channel only. Helps SAM distinguish near-white components (blwo11, rome) from each other and from the white margins.

**5. Foreground-only grid prompts** (`_fg_grid_points`):
```python
n_target = max(1, int(n_components * density))   # density=2.5
spacing  = max(min_spacing, int((fg_area / n_target) ** 0.5))
# Regular grid → keep only points that fall inside fg_mask
```
Only points inside the foreground are used. This guarantees no SAM decoder calls are wasted on background. Spacing adapts to construction area: small constructions get denser grids.

**6. SamPredictor — one encoder pass, M decoder passes**:
```python
predictor.set_image(crop_rgb)   # one encoder pass
for (cx, cy) in points:
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[cx, cy]]), point_labels=np.array([1]),
        multimask_output=True,   # 3 size candidates: small / medium / large
    )
    best = int(scores.argmax())
    crop_masks.append({"segmentation": masks[best], "predicted_iou": scores[best], ...})
```
Each prompt point produces three candidate masks (small/medium/large component under the point). The highest-scored one is kept. With 12–15 prompts per image and 3 candidates each, this is far cheaper than running a 32-point uniform grid.

**7. Coordinate translation** (`_expand_masks`):
```python
full = np.zeros((orig_H, orig_W), bool)
full[y_off:y_off+ch, x_off:x_off+cw] = seg_crop[:ey1-ey0, :ex1-ex0]
```
Translates tight-crop-coordinate masks back to fixed-scale image coordinates so they can be indexed against `pts3d`.

**8. Filter pipeline** (`_filter_masks`):

| Step | Logic | Purpose |
|------|-------|---------|
| fg-overlap | `inside < 0.60 × area` → drop | Remove table/background masks |
| border | `seg[0,:].any() or seg[-1,:].any()` → drop | Remove padding/scene masks |
| oversized | `area > 0.50 × fg_area` → candidate for decomposition | Whole-object masks |
| dedup | pairwise IoU > 0.80 → drop lower-confidence | Remove redundant masks per prompt |
| container | mask > 50% covered by smaller masks → subtract remainder | Decompose multi-part unions |
| blob-split | `cv2.connectedComponentsWithStats` | Split one mask covering two disconnected regions |
| oversized remainder | Subtract all small kept masks → remainder is dominant component | Recover dominant piece |

```python
# Container pruning (simplified):
final.sort(key=lambda m: m["area"])   # smallest first
for m in final:
    covered = np.logical_and(m["segmentation"], union_smaller).sum() / m["area"]
    if covered > 0.50:
        remainder = np.logical_and(m["segmentation"], ~union_smaller)
        if remainder.sum() >= min_area:
            pruned.append({**m, "segmentation": remainder})
```

**Failure modes in SAM**:
- **Thin rods get zero prompt points**: After downscaling to `fixed_half × fixed_half`, a 4 mm rod is ~3 px wide. With `min_spacing=12`, no grid point may land on it. The rod never appears in any mask.
- **Near-white pieces become background**: If `BG_THRESHOLD=245` misses a near-white blwo11 block, the foreground mask excludes it. The tight crop misses it, and `crop_bgr[~crop_fg] = 0` makes it black — SAM sees it as background and never generates a mask for it.
- **Container decomposition order matters**: If the small rod mask fails the border or fg-overlap check, it never enters the `pruned` list. The larger block+rod mask then has no smaller mask to subtract from, so its remainder is the full block+rod area — two components merged into one 2D mask.
- **Prompt inside large block generates "large" candidate**: `multimask_output=True` returns small/medium/large. A prompt inside a large block may score the "large" mask (covers the whole construction) higher than the "small" one (covers just the block). The filter pipeline's container decomposition is supposed to recover the individual piece, but it only works if smaller piece masks already exist in `final`.

---

### `backprojector.py` — 3D point clouds and instance grouping

**Input**: `dust3r_result`, `sam_masks_per_image`, `n_components`  
**Output**: `(merged, visual_cls, cluster_colours, cluster_elongations)` — four parallel lists of length N

This module bridges 2D masks and 3D geometry.

#### Per-mask back-projection (`_extract_mask_cloud`)
```python
if conf is not None:
    mask = np.logical_and(mask, conf >= conf_thresh)   # conf_thresh=2.0
pts = pts3d_frame[mask]                                # direct indexing
valid = np.linalg.norm(pts, axis=1) > 1e-6            # drop zero-norm (unobserved)
if len(pts) > _CLOUD_MAX_PTS:                          # 5000
    idx = np.random.choice(len(pts), _CLOUD_MAX_PTS, replace=False)
```
The confidence filter removes DUSt3R outliers from textureless or occluded regions. The 5000-point cap prevents memory explosion when a large component covers thousands of pixels across many views.

**Failure mode — unseeded sampling**: The `np.random.choice` for the 5000-point cap uses the global random state and is unseeded. This means two runs on the same images produce slightly different point subsets, and therefore slightly different bounding boxes, which changes the cost matrix. The per-mask sampling is not fixed; only the DBSCAN subsampling is seeded (seed=42).

#### Colour per mask (`_mean_hsv`)
```python
pixels = img_rgb[mask]
hsv = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0]
mean = hsv.mean(axis=0)
return (mean / np.array([180., 255., 255.])).astype(np.float32)   # normalise to [0,1]
```
Mean HSV of all pixels under the mask in the RAW (pre-CLAHE) image. The prototype HSVs in `classifier.py` are also built from raw images, so this comparison is consistent.

#### 3D voxel-overlap grouping (`_overlap_group`)
This is the central grouping algorithm. All individual mask clouds (potentially hundreds across 32 images) must be merged into exactly N clusters.

```python
voxel = max(diag / voxel_div, 1e-5)   # voxel_div=80 → ~3–5 mm voxels
keys = [set(map(tuple, np.floor(c / voxel).astype(np.int64))) for c in clouds]

# Pairwise overlap: ∩ / min(|A|, |B|)
overlap[a, b] = len(keys[a] & keys[b]) / min(len(keys[a]), len(keys[b]))
dist = 1.0 - overlap   # agglomerative clustering uses distance

# Optional HSV colour repulsion
if colors is not None and color_weight > 0:
    cd = np.linalg.norm(hsv_v[a] - hsv_v[b]) / 1.732   # normalise to [0,1]
    dist[a, b] += color_weight * cd                      # color_weight=0.3
```

AgglomerativeClustering with `linkage="average"` and `n_clusters=N`.

**Why average linkage**: Single linkage chained everything together (the whole construction merged into 2 groups). Complete linkage over-fragmented. Average linkage uses the mean pairwise overlap between all members of two candidate groups — it only merges groups when they consistently share voxels, not just at one shared boundary.

**Why colour repulsion is weak**: The max repulsion contribution is `0.3 × 1.0 / 1.732 ≈ 0.17`. For two clouds with voxel overlap = 0.5 (strong), the overlap distance = 0.5. After repulsion, `0.5 + 0.17 = 0.67`. The clustering threshold (where average linkage splits) depends on the other pairwise distances in the group. Colour repulsion can prevent merges at weak overlap, but for truly overlapping clouds (rod inside block), it has no effect.

**Stable ordering** by total point count: the largest cluster (most 3D points) is always cluster 0, second-largest is cluster 1, etc. This makes output order deterministic and human-readable.

#### DBSCAN cleanup (`_dbscan_cleanup`)
```python
diag = np.linalg.norm(sub.max(axis=0) - sub.min(axis=0))
eps  = max(diag * 0.05, 1e-4)   # 5% of bbox diagonal
labels = DBSCAN(eps=eps, min_samples=5).fit_predict(sub)

# For large clouds: run DBSCAN on subsample, use cluster centroid + 95th-percentile
# radius to filter the full cloud
if sub_idx is not None:
    centre = np.median(main_sub, axis=0)
    radius = np.percentile(dists_in_cluster, 95) * 2.0
    kept = pts[np.linalg.norm(pts - centre, axis=1) <= radius]
```

The seeded subsample (seed=42 via `np.random.default_rng`) ensures reproducible DBSCAN clustering even for large clouds (`> _DBSCAN_MAX_PTS = 20000`).

**Failure mode — gap kills rod reconstruction**: When a rod's two exposed stubs are separated by a 50 mm occluded gap (e.g. rome going through blwo11), `eps = 5% × 0.1m = 5 mm` cannot bridge the gap. DBSCAN treats one stub as noise and removes it. The surviving stub has too few points and appears compact (elong ≈ 1.2) rather than rod-like (elong ≈ 6).

#### Co-axial rod merge (`_coaxial_merge`)
After DBSCAN, occluded rods may exist as two separate clusters. This geometric test merges them:
```python
def _pca(pts):
    mean = pts.mean(axis=0)
    _, S, Vt = np.linalg.svd(pts - mean, full_matrices=False)
    return mean, Vt[0], S[0] / max(S[1], 1e-9)   # (centroid, axis, elongation)

# Conditions for co-axial merge:
# 1. both clusters have elong > min_elong (=1.5) — only rods qualify
# 2. |dot(axis_i, axis_j)| >= cos(20°) — axes within 20° of parallel
# 3. lateral offset < 12 mm — not two separate parallel rods
# 4. end-to-end gap < 120 mm AND < 50% of combined span
```

When a merge happens, the smaller cluster's points are concatenated into the larger cluster's array. The smaller cluster slot is set to an empty array (`np.empty((0, 3))`). This empty slot still occupies a position in the list so that the N cluster count is preserved for Hungarian.

**Failure mode — DBSCAN runs first**: If DBSCAN discards one of the two rod stubs before co-axial merge can see them, the merge never happens. The two tests happen in the wrong order for heavily occluded rods.

#### PCA elongation per cluster (`_compute_elongation`)
```python
def _compute_elongation(pts: np.ndarray) -> float:
    if len(pts) < 10:
        return 1.0
    centered = pts - pts.mean(0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    return float(s[0] / max(s[1], 1e-9))   # σ₁/σ₂
```
PCA singular value ratio. Interpretation:
- ≈ 1.0–1.3 : isotropic (nut, bolt, near-spherical blob)
- ≈ 1.5–2.0 : mildly elongated (plates, blocks)
- ≈ 4–9 : strongly elongated (rods, screws, straps)

This value is passed to `assign_classes` as `cluster_elongations` and used as the third cost term.

---

### `component_map.py` — Class-to-folder mapping and colour prototypes

**Purpose**: Maps between filesystem folder names and PUML class codes, and builds mean-HSV colour prototypes for each class from isolated component reference images.

```python
FOLDER_TO_CLASS = {
    "blockwood11": "blwo11",
    "plateplastic53": "plpl53",
    "rodmedium": "rome",
    "strapwood3": "stwo3",
    # ... 29 entries total
}
```

**Colour prototype extraction** (`build_color_prototypes`):
```python
for folder, cls in FOLDER_TO_CLASS.items():
    img_dir = component_images_dir / folder / "Images45"
    # Average HSV across all reference views
    hsv_samples = [_mean_hsv(cv2.imread(str(p))) for p in img_paths]
    prototypes[cls] = np.stack(hsv_samples).mean(axis=0)
```

Only `Images45` reference views are used. The prototype is the mean of mean-HSVs across all images of one class. These are cached to `outputs/color_prototypes.pkl`.

**Failure mode — single-elevation prototype**: Reference images are taken at 45° elevation only. The multi-view images in the dataset include 30°, 60°, and 90° views where lighting and foreshortening change apparent colour. A piece that's gray at 45° may appear lighter or darker at 90° (top-down). The prototype may not represent what SAM masks actually see.

**Failure mode — small class sample size**: Some classes have very few reference images. If `rodmedium/Images45/` has only 3 images, the prototype is averaged over 3 samples, which is noisy.

---

### `classifier.py` — Hungarian cost-matrix assignment

**Input**: N point clouds, N manifest components, optional cluster colours and elongations  
**Output**: `list[InstanceResult]` — one per proposal, with `instance_id`, `cls`, `n_points`, `centroid`, `bbox_size`

This is the final decision stage. It assigns each 3D cluster to exactly one PUML component using globally optimal matching.

#### Nominal size system

Nominal sizes encode expected physical long-dimension in BRIO stud units:
```python
_NOMINAL_NO_DIGITS = {
    "rolo": 8.0, "rome": 6.0, "rosm": 4.0,    # rods
    "sclo": 4.0, "scme": 3.0, "scsm": 2.0,    # screws
    "nu": 1.2,   "wa": 0.8,   "bo": 1.5,       # hardware
    # Special override for windmill blocks:
    # "11"/"21" in the name encode opening variant, NOT size dimension.
    # The digit parser max([1,1])=1 would make blwo11 look as small as a nut.
    "blwo11": 9.0, "blwo21": 7.0,
}

def _nominal_size(cls: str) -> float:
    if cls in _NOMINAL_NO_DIGITS:
        return _NOMINAL_NO_DIGITS[cls]
    digits = [int(d) for d in re.findall(r"\d", cls)]
    return float(max(digits)) if digits else 1.0
    # "plpl53" → digits=[5,3] → max=5
    # "stwo9"  → digits=[9]   → max=9
```

#### Nominal elongation system

```python
_NOMINAL_ELONG = {
    "rolo": 8.0, "rome": 6.0, "rosm": 4.0,    # strongly elongated rods
    "scsm": 2.5, "scme": 3.5, "sclo": 4.0,    # screws
    "stwo3": 3.0, "stwo5": 5.0, ...,           # straps
    "blwo11": 1.5, "blwo21": 1.6,              # cross-blocks (nearly isotropic)
    "nu": 1.2, "wa": 1.1,                      # hardware (isotropic)
}
_DEFAULT_ELONG = 1.7   # digit-encoded plates (plpl53: 5/3 ≈ 1.67)
```

#### Cost matrix

```python
obs_ext = (p.max(0) - p.min(0)).max()   # axis-aligned bbox max dim
rel_obs = obs_ext / max(obs_ext)        # normalised within sample
rel_nom = nom / max(nom)                # normalised within manifest

for pi in range(n):
    for mi, comp in enumerate(manifest_components):
        # Colour: L2 in normalised HSV
        c_cost = np.linalg.norm(obs_hsv - proto)

        # Size: relative bbox vs relative nominal
        s_cost = abs(rel_obs[pi] - rel_nom[mi])

        # Elongation: log-scale to make rod vs block distinction scale-invariant
        nom_e  = _nominal_elong(comp.cls)
        e_cost = abs(math.log(max(obs_elong, 1.0)) - math.log(max(nom_e, 1.0)))

        # Visual classifier knockout (if weights are loaded)
        v_penalty = 8.0 if visual_cls[pi] not in (None, comp.cls) else 0.0

        cost[pi, mi] = (2.5 * c_cost + 1.0 * s_cost + 1.0 * e_cost + v_penalty)

row_ind, col_ind = linear_sum_assignment(cost)   # globally optimal assignment
```

**Why log-scale for elongation**: `|log(6) - log(1.5)| = 1.39` (rod vs block, large cost). `|log(1.7) - log(1.5)| = 0.12` (plate vs block, small cost). Log-scale makes the cost grow fast for the rod/block distinction and slowly for the plate/block distinction, matching physical reality.

**The core bug (sample 113 case study)**:
- Cluster 1 (plpl53 plate): 96K pts, bbox_max=0.174 m (largest in sample)
- Cluster 0 (blwo11 block): 253K pts, bbox_max=0.134 m (second largest)
- Manifest: blwo11 nominal=9.0 (largest), plpl53 nominal=5.0

Because `rel_obs` ranks the plate first and `rel_nom` ranks blwo11 first, the cost matrix has `s_cost(c1→blwo11) = 0` (both are "largest"). The globally optimal assignment therefore assigns the plate to blwo11, even though elongation and colour say otherwise. The three-term cost with weights 2.5 / 1.0 / 1.0 is not enough to overcome the size=0 advantage when colour differences are small (near-white assembly).

**Visual mismatch penalty = 8.0**: This knockout term would dominate all three geometric signals combined (~4 max total). It is the intended fix for the above case, but it requires a trained `component_classifier.pth`.

---

### `component_classifier.py` — MobileNetV3 visual classifier

**Purpose**: Classify isolated component crops to class codes. When loaded, its predictions are used as a strong visual vote in `assign_classes`.

```python
class ComponentClassifier:
    def __init__(self, weights_path, device="cuda"):
        ckpt  = torch.load(weights_path)
        model = models.mobilenet_v3_small()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_cls)
        model.load_state_dict(ckpt["model_state"])

    def predict(self, bgr_crop) -> tuple[str, float]:
        probs = torch.softmax(self._model(tensor), dim=1)[0]
        return self._classes[probs.argmax()], float(probs.max())
```

**Training** (`--train` flag):
- Dataset: `COMPONENT_DATASET` directory, structured as `folder_name/Images45/img.jpg`
- Split: 85% train, 15% val
- Augmentation: RandomResizedCrop(224), RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomGrayscale
- Optimiser: AdamW(lr=3e-4, weight_decay=1e-4) + CosineAnnealingLR
- Loss: CrossEntropyLoss with label_smoothing=0.1

**In inference** (`backprojector.py`): classifier is run on CLAHE-enhanced crops of each SAM mask, batched for efficiency. Each crop uses `crop_from_mask` which grey-fills everything outside the mask so the model sees only the target component:
```python
crop[~mask_crop] = 128   # grey background
square = np.full((side, side, 3), 255)   # white padding to make it square
```

**Status**: `component_classifier.pth` is present but was trained on a limited dataset. Its predictions are gated by `CLASSIFIER_CONF_THRESH=0.65` — predictions below 65% confidence are treated as None and don't affect the assignment.

---

### `pipeline.py` — Orchestration

This is the entry point. It ties all modules together and handles the run directory, image collection, and the amodal extension step.

**Image collection** (`collect_images`):
```python
for elev in ["Images30", "Images45", "Images60", "Images90"]:
    ring   = sorted(folder.glob("*.jpg"))
    stride = max(1, len(ring) // IMAGE_VIEWS_PER_RING)   # target 8 per ring
    imgs.extend(ring[::stride])
```
Adaptive stride ensures small rings (8 images at Images90) contribute all their views while large rings (24 images) are sub-sampled. Typical result: ~32 images across four elevations.

**Run directory** (`_make_run_dir`):
```python
run_num = len(existing) + 1           # count existing run_* dirs
stamp   = datetime.now().strftime("%Y%m%d_%H%M")
run_dir = OUTPUTS_ROOT / f"run_{run_num:03d}_{stamp}"
```

**Per-sample flow** (`process_sample`):
```python
manifest = parse_puml(find_puml(sample_id))           # 1. PUML manifest
img_paths = crop_images(raw_img_paths, crop_dir, fixed_half)   # 2. Crop
dust3r_result = run_dust3r(img_paths, ...)             # 3. 3D reconstruction
sam_masks = run_sam_on_images(img_paths, ...)          # 4. 2D masks
proposals, visual_cls, colours, elongations = compute_proposals(...)  # 5. 3D clouds
results = assign_classes(proposals, manifest.components,
                         cluster_colours=colours,
                         cluster_elongations=elongations)  # 6. Assignment
if manifest.thjo_pairs:
    _amodal_extend_rods(results, proposals, manifest.thjo_pairs,
                        protrusion_m=ROD_PROTRUSION_M)    # 7. Amodal extend
```

**Amodal bbox extension for through-joint rods** (`_amodal_extend_rods`):

For each `(rod_id, [through_ids])` pair from `thjo_pairs`, this function extends the rod's bounding box to cover the full length of the rod including its hidden interior:

```python
# Gather rod + all through-component clouds
all_pts = np.concatenate([rod_pts, *through_pts], axis=0)

# Rod axis: prefer rod-cloud PCA; fall back to through-comp minor PCA axis
if len(rod_pts) >= 30:
    rod_axis = np.linalg.svd(rod_pts - rod_pts.mean(0))[2][0]  # top PCA axis
else:
    rod_axis = np.linalg.svd(through_pts - through_pts.mean(0))[2][-1]  # minor axis

# Project all points onto rod axis, extend by protrusion=30mm on each end
proj   = (all_pts - origin) @ rod_axis
ext_lo = proj.min() - protrusion_m
ext_hi = proj.max() + protrusion_m

# Build a synthetic axis line and take AABB of (all points + line)
t        = np.linspace(ext_lo, ext_hi, 50)
rod_line = origin + np.outer(t, rod_axis)
aabb_lo  = np.vstack([all_pts, rod_line]).min(axis=0)
aabb_hi  = np.vstack([all_pts, rod_line]).max(axis=0)

results[rod_idx].bbox_size = (aabb_hi - aabb_lo).astype(np.float32)
results[rod_idx].centroid  = ((aabb_lo + aabb_hi) / 2).astype(np.float32)
```

**Failure mode — depends on correct assignment**: `id_to_idx` maps `instance_id → cluster index`. If the assignment is wrong (the rome rod is mislabelled as `nu_1`), then `rod_id="rome_1"` is not in `id_to_idx` and amodal extension is silently skipped for that rod.

**Output** (`results.json`):
```json
{
  "sample_id": 113,
  "n_components": 5,
  "instances": [
    {
      "instance_id": "blwo11_1",
      "cls": "blwo11",
      "n_points": 253812,
      "centroid": [0.012, -0.003, 0.841],
      "bbox_size": [0.134, 0.127, 0.091]
    }
  ]
}
```

---

### `logger.py` — Dual stdout/file logging

Redirects both `stdout` and `stderr` through a `_Tee` object that writes to both terminal and a timestamped log file simultaneously:

```python
sys.stdout = _Tee(sys.__stdout__, fh)
sys.stderr = _Tee(sys.__stderr__, fh)
```

Log files go to `logs/YYYYMMDD_HHMMSS_samples_<N>.log`. All `print()` calls from all modules are captured without any changes to those modules.

---

### `visualize_2d.py` — 2D mask verification overlay

For each detected component, renders up to N view tiles side by side. The component's contributing SAM masks are highlighted in a distinct colour; all other masks appear as dim grey.

```bash
python visualize_2d.py --sample 113
python visualize_2d.py --sample 113 --run run_021_20260616_2339
python visualize_2d.py --sample 113 --views-per-component 6
```

Output: `outputs/run_NNN/sample_NNN/viz_2d.png`

Use this to diagnose whether SAM masks are wrong, or whether the 3D clustering/assignment is wrong — if the masks look correct but the label is wrong, the issue is in the classifier; if the masks are bad, the issue is upstream in SAM or preprocessing.

---

### `visualize.py` — 3D point-cloud export

Exports `viz_3d.ply` (full coloured point cloud) and renders a 3-panel PNG (isometric / top-down / front) via Open3D OffscreenRenderer. Component instance clouds are overlaid in bright label colours with axis-aligned bounding boxes.

```bash
python visualize.py --sample 113
python visualize.py --sample 114 --run run_021_20260616_2339
```

Open `viz_3d.ply` in CloudCompare (Windows) or MeshLab for an interactive view of the raw 3D reconstruction.

---

## Data flow: concrete example (sample 113)

Sample 113 contains: `blwo11_1`, `rome_1`, `plpl53_1`, `nu_1`, `scsm_1` (5 components).

```
Images (32 views, ~600×600 px after crop)
       │
       ▼  DUSt3R
pts3d[0..31]: each (300, 300, 3) — world-space 3D per pixel
confs[0..31]: each (300, 300)    — reconstruction confidence

       ▼  SAM (32 images × up to 10 masks each)
~320 raw mask candidates
→ filter → ~80–100 kept masks
  mask[0]: 2D bool (300,300) covering "blwo11 block, image 3"
  mask[1]: 2D bool (300,300) covering "plpl53 plate, image 3"
  ...

       ▼  Back-projection
~80 3D clouds (one per kept mask):
  cloud[0]: (5000, 3) — blwo11 surface, image 3
  cloud[5]: (5000, 3) — blwo11 surface, image 7
  ...

       ▼  Voxel-overlap grouping (80 clouds → 5 clusters)
cluster 0: 253K pts  (blwo11 — reconstructed with most views)
cluster 1:  96K pts  (plpl53 plate)
cluster 2:   6K pts  (rome rod + blwo11 interior — merged via shared voxels)
cluster 3:   1.3K pts (scsm screw)
cluster 4:    698 pts (nu nut)

       ▼  PCA elongation
cluster 0: elong=1.31  (blwo11 is nearly cubic)
cluster 1: elong=1.67  (plate is flatter)
cluster 2: elong=1.24  (rome+blwo11 blob appears compact — rod occluded)
cluster 3: elong=1.46
cluster 4: elong=1.12

       ▼  Hungarian assignment (cost matrix 5×5)
Wrong result (current):
  cluster 0 → rome_1    (bbox=0.134m, size_cost=0.23)
  cluster 1 → blwo11_1  (bbox=0.174m — largest → size_cost=0.0 for blwo11)
  cluster 2 → plpl53_1  (tiny cluster → wrong)
  cluster 3 → scsm_1
  cluster 4 → nu_1

Correct would be:
  cluster 0 → blwo11_1
  cluster 1 → plpl53_1
  cluster 2 → rome_1
```

The assignment is wrong because the plate's 3D bbox (0.174 m) is larger than the block's (0.134 m), but the block has the largest nominal size (9.0). The plate steals the blwo11 slot because `s_cost(plate→blwo11) = 0`.

---

## Known failure modes

| # | Stage | Root cause | Effect |
|---|-------|-----------|--------|
| 1 | SAM prompts | Thin rod (< 3 px wide after downscale) has zero grid points inside it | Rod never masked → zero points in 3D |
| 2 | SAM foreground mask | Near-white piece has all channels near 245 → classified as background | Piece becomes black → SAM sees only background where the piece is |
| 3 | Overlap grouping | Rod passes through block → shared voxels at junction | Rod+block merge into one cluster |
| 4 | DBSCAN cleanup | 50 mm gap between rod stubs > eps=5 mm | One stub is discarded as noise; surviving stub appears compact |
| 5 | Co-axial merge | Runs after DBSCAN, so stubs may already be discarded | Merge never fires |
| 6 | Size metric | Flat plate bbox_max > compact block bbox_max | Plate steals "largest nominal" assignment slot |
| 7 | Colour prototypes | All-white/gray assembly → HSV prototypes cluster near (0,0,0.9) | Colour cost ≈ 0 for all class pairs; no discrimination |
| 8 | Elongation feature | Through-joint rod → no elongated cluster exists | Log-elong cost is uniform across all assignments |
| 9 | Visual classifier | MobileNetV3 confidence below threshold for ambiguous crops | v_penalty = 0, falls back to geometry only |
| 10 | Amodal extension | Wrong assignment → rod_id not in id_to_idx | Silently skipped; rod bbox uses only stub points |

---

## Output structure

```
outputs/
└── run_023_20260618_2220/
    └── sample_113/
        ├── cropped/
        │   ├── Images30_IMG_001.jpg   # fixed-scale crop, half-resolution
        │   └── ...
        ├── dust3r/
        │   └── dust3r_cache.pkl       # pts3d, confs, poses, intrinsics
        ├── sam/
        │   └── sam_masks_top10_prompted_filt.pkl   # filtered masks per image
        ├── proposals/
        │   └── proposals_N5_v12.pkl   # 5 merged clusters + colours + elongations
        ├── image_order.json           # ordered list of cropped image paths
        ├── results.json               # final per-component assignment
        ├── viz_2d.png                 # 2D mask overlay grid (from visualize_2d.py)
        └── viz_3d.ply                 # coloured 3D point cloud (from visualize.py)
```

**Cache version suffix** in `proposals_N5_v12.pkl`:
- `v12` = current version (elongation feature + seeded DBSCAN)
- `_vis` suffix = run with visual classifier loaded (e.g. `proposals_N5_v12_vis.pkl`)

Changing any grouping parameter (voxel div, overlap min, colour weight, co-axial thresholds) requires deleting the `proposals_N*.pkl` file to force regeneration.
