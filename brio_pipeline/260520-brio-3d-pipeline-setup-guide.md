# BRIO 3D Pipeline — Setup and Run Guide
**Created: 2026-05-20**
**Applies to: samples 113–117 (trial run)**

This guide documents every step taken to set up the modified Open-YOLO 3D pipeline
(DUSt3R + SAM + back-projection + geometric assignment) on the local machine.

---

## Machine Specs

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 2070 Super |
| VRAM | 8 GB |
| CUDA driver | 12.7 (Windows driver 566.36, passed through WSL2) |
| OS | WSL2 Ubuntu (Linux 6.6.87.2-microsoft-standard-WSL2) |
| Conda | Miniconda at `/home/mxrn/miniconda3` |

---

## Why the GPU Was Not Working Before

The existing `sam` conda environment contained PyTorch 2.12.0+cu130, which requires
CUDA 13.0 runtime — but the installed driver only supports CUDA 12.7.
PyTorch raised a `CUDA initialization` error and fell back silently to CPU.

**Rule: always match the PyTorch `+cuXXX` suffix to the CUDA version your driver supports,
or any lower version. Never install a PyTorch build compiled against a higher CUDA than
your driver supports.**

Check your driver's maximum supported CUDA version with:
```bash
nvidia-smi   # top-right corner shows "CUDA Version: 12.7"
```

---

## Step 1 — Create a New Conda Environment

```bash
conda create -n brio-3d python=3.10 -y
```

---

## Step 2 — Install PyTorch with CUDA 12.4

CUDA 12.4 is the highest stable PyTorch build compatible with a CUDA 12.7 driver.

```bash
conda run -n brio-3d pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
```

**Verify GPU is recognised:**
```bash
conda run -n brio-3d python -c "
import torch
print('torch:', torch.__version__)          # should show 2.6.0+cu124
print('CUDA:', torch.cuda.is_available())   # should show True
print('GPU:', torch.cuda.get_device_name(0))
"
```

Expected output:
```
torch: 2.6.0+cu124
CUDA: True
GPU: NVIDIA GeForce RTX 2070 Super
```

---

## Step 3 — Install All Pipeline Dependencies

```bash
conda run -n brio-3d pip install \
    roma einops trimesh scipy tqdm huggingface-hub \
    opencv-python matplotlib segment-anything ultralytics
```

Then install DUSt3R's own requirements (the repo has no `setup.py`, so it is added to
`sys.path` at runtime rather than installed as a package):

```bash
conda run -n brio-3d pip install -r /mnt/c/BA/07-dust3r/requirements.txt
```

**Verify DUSt3R imports:**
```bash
conda run -n brio-3d python -c "
import sys; sys.path.insert(0, '/mnt/c/BA/07-dust3r')
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
print('DUSt3R OK')
"
```

You will see:
```
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
DUSt3R OK
```

The RoPE2D warning is harmless — it uses the pure-PyTorch fallback, which is slightly
slower but fully correct. To suppress it, you would need to compile the CUDA extension
(not needed for the thesis).

---

## Step 4 — Model Weights

**SAM ViT-B** checkpoint is already downloaded at:
```
/mnt/c/BA/03-code/sam_weights/sam_vit_b_01ec64.pth
```

**DUSt3R weights** are downloaded automatically on first run from HuggingFace:
```
naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt
```
This requires an internet connection on the first run (~1.6 GB download).
Subsequent runs use the cached model in `~/.cache/huggingface/`.

---

## Step 5 — Project Structure

All pipeline code lives in `00-project/brio_3d_pipeline/`:

```
00-project/brio_3d_pipeline/
├── __init__.py
├── config.py           # central paths and settings
├── puml_parser.py      # parse PUML → component manifest (N, class list)
├── dust3r_runner.py    # DUSt3R wrapper: RGB images → poses + depth maps
├── sam_runner.py       # SAM wrapper: images + N → top-N masks per image
├── backprojector.py    # masks + depth + poses → N 3D point clouds
├── classifier.py       # N clouds + PUML manifest → Hungarian assignment
├── pipeline.py         # main orchestration script (run this)
├── visualize.py        # 3D scatter plot of results
└── outputs/
    └── sample_113/
        ├── dust3r/
        │   └── dust3r_cache.pkl
        ├── sam/
        │   └── sam_masks_top5.pkl
        ├── proposals/
        │   └── proposals_N5.pkl
        ├── results.json
        └── viz_3d.png
```

All intermediate results are cached to `.pkl` files.
Re-running the pipeline skips any stage whose cache already exists.
Delete the cache file for a stage to force it to re-run.

---

## Step 6 — Configuration

Key settings are in `config.py`. Defaults are set for the trial run:

| Setting | Default | Meaning |
|---------|---------|---------|
| `IMAGE_ELEVATION` | `"Images45"` | Which azimuth ring to use (24 images) |
| `DUST3R_SIZE` | `512` | Image resize for DUSt3R encoder |
| `DUST3R_NITER` | `300` | Global alignment iterations |
| `DUST3R_BATCH` | `1` | Pairs per forward pass (keep 1 for 8 GB VRAM) |
| `SAM_MODEL_TYPE` | `"vit_b"` | SAM model variant (vit_b = 4 GB VRAM) |
| `SAM_POINTS_SIDE` | `16` | Grid density for auto mask generation |
| `SAM_IOU_THRESH` | `0.80` | Minimum SAM predicted IoU |
| `DEVICE` | `"cuda"` | Torch device — set to `"cpu"` as fallback |

---

## Step 7 — Run the Pipeline

Always activate the `brio-3d` environment first:

```bash
conda activate brio-3d
cd /mnt/c/BA/00-project/brio_3d_pipeline
```

**Single sample:**
```bash
python pipeline.py --samples 113
```

**All five trial samples:**
```bash
python pipeline.py --samples 113 114 115 116 117
```

**Force CPU (skip GPU):**
```bash
python pipeline.py --samples 113 --device cpu
```

**Use a different elevation ring (e.g. 60°):**
```bash
python pipeline.py --samples 113 --elevation Images60
```

**Expected console output (sample 113, N=5 components):**

```
============================================================
 Sample 113
============================================================
[PUML] 5 components: ['blwo11', 'plpl53', 'rome', 'nu', 'nu']
[Images] 24 images from Images45
[DUSt3R] Running on 24 images (device=cuda)
[DUSt3R] 88 image pairs, running pairwise inference...
[DUSt3R] Running global alignment...
[DUSt3R] Cached results to outputs/sample_113/dust3r/dust3r_cache.pkl
[SAM] Generating masks for 24 images (top-5 per image)
[SAM]   5/24 images processed
...
[SAM] Cached masks to outputs/sample_113/sam/sam_masks_top5.pkl
[Backproject] Back-projecting masks for 24 images
[Backproject] Merged into 5 3D instance clouds
  Instance 0: XXXXX points
  ...
[Results] Sample 113
  Instance              Points         Centroid (x,y,z)             BBox size
  blwo11_1              12345    (+0.012, -0.008, +0.503)    (0.187, 0.052, 0.031)
  ...
[Results] Saved to outputs/sample_113/results.json
```

---

## Step 8 — Visualise Results

```bash
python visualize.py --sample 113
```

Saves a 3D scatter plot of all instance point clouds to:
```
outputs/sample_113/viz_3d.png
```

---

## Step 9 — Re-running and Cache Management

Each pipeline stage caches its output. To re-run a stage:

| Stage to re-run | Delete this file |
|----------------|-----------------|
| DUSt3R | `outputs/sample_N/dust3r/dust3r_cache.pkl` |
| SAM | `outputs/sample_N/sam/sam_masks_topN.pkl` |
| Back-projection | `outputs/sample_N/proposals/proposals_NN.pkl` |
| Classification | Just re-run `pipeline.py` (no cache for this stage) |

---

## VRAM Usage Notes

The pipeline runs each stage sequentially, so VRAM is never shared between models:

| Stage | Peak VRAM |
|-------|-----------|
| DUSt3R (batch=1) | ~4–6 GB |
| SAM ViT-B | ~4 GB |
| Back-projection | CPU only |
| Classification | CPU only |

If you hit out-of-memory errors during DUSt3R:
- Reduce image count: use `Images60` (24 images) or a 12-image subset
- Reduce input size: change `DUST3R_SIZE = 384` in `config.py`
- Reduce `DUST3R_BATCH` to 1 (already the default)

If you hit OOM during SAM:
- Switch to `SAM_MODEL_TYPE = "vit_b"` (already default — smaller than ViT-H)
- Reduce `SAM_POINTS_SIDE = 8` in `config.py`

---

## What Each Module Does (Quick Reference)

| Module | Input | Output |
|--------|-------|--------|
| `puml_parser.py` | `.puml` file path | N, class list, instance IDs |
| `dust3r_runner.py` | 24 RGB images | poses (24×4×4), intrinsics (24×3×3), depth maps (24×H×W) |
| `sam_runner.py` | 24 images + N | top-N masks per image (bool arrays) |
| `backprojector.py` | masks + depth + poses | N merged 3D point clouds |
| `classifier.py` | N point clouds + PUML | N `InstanceResult` objects with class assignments |
| `pipeline.py` | sample ID | `results.json` |
| `visualize.py` | sample ID | `viz_3d.png` |

---

## Known Limitations of the Trial Run

1. **Classification is geometric only** — YOLOv8n has not yet been trained on BRIO data.
   The current classifier uses bounding-box geometry and the Hungarian algorithm to match
   point clouds to PUML-declared components. Accuracy depends entirely on DUSt3R
   reconstruction quality and whether components are visually distinct in size.

2. **DUSt3R on textureless BRIO surfaces** — wooden and plastic surfaces may confuse
   DUSt3R's feature matcher. If depth maps look wrong, check `viz_3d.png` for obvious
   outlier point clouds and consider increasing `DUST3R_NITER` to 500.

3. **SAM top-N selection** — SAM may merge two adjacent components into one mask or
   split one component into two. The PUML N-constraint forces exactly N proposals,
   so a merge/split error will propagate to the assignment stage.

4. **Scale of scene units** — DUSt3R returns depth in arbitrary scene units (not metric).
   The class prototypes in `classifier.py` are in centimetres; a scale normalization
   step will be needed for accurate matching. For the trial run, ranking-based assignment
   (largest proposal → largest class) partially compensates.

---

## Next Steps After the Trial

1. Inspect `viz_3d.png` for each sample — do the 5 point clouds look like 5 separate objects?
2. Check `results.json` — do the assigned classes match the PUML?
3. If DUSt3R quality is poor on a sample, test `Images60` as an alternative elevation ring
4. Once YOLOv8n is trained (next phase), replace the geometric classifier with
   crop-based YOLOv8n voting for better class discrimination on ambiguous components
