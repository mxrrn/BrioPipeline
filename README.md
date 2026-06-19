# From Multi-View 2D to 3D Object Detection and Structure Generation in PlantUML

**Bachelor's Thesis:** From Multi-View Images to Instance Diagrams: A Multi-View Detection Pipeline for Structural Relationship Inference in Small Assemblies

---

## Problem

BRIO construction toys are assembled from discrete physical components (bolts, nuts, screws, plates, wheels, etc.) that connect through typed attachment points. Given a set of photographs of a completed construction, the goal is to automatically identify which components are present, localise them in 3D, infer their connections, and represent the result as a PlantUML instance diagram.

---

## Domain Model

```
Construction
  └── has one or many ConnectionConfiguration(s)
        └── has one or more Connection(s)
              └── is formed by exactly 2 Slot(s)
                    └── belongs to a Component
```

| Term | Definition |
|------|-----------|
| **Component** | A physical BRIO part (e.g., bolt, nut, plate, wheel). Has one or more slots. |
| **Slot** | A typed attachment point on a component (e.g., opening, pin, thread). |
| **Connection** | A link formed by exactly two slots joining together. |
| **Connection Configuration** | A group of connections that share a common joint point. |
| **Construction** | The complete physical assembly. |

---

## Dataset

~150 annotated BRIO construction samples. Each sample contains:

| File | Description |
|------|-------------|
| `Construction.jpg` | Photograph of the physical construction |
| `InstanceDiagramSN.puml` | Ground-truth PlantUML instance diagram |
| `Mapping.drawio` | Visual mapping between photo regions and components |

**Multi-view images:** ~78 photographs per sample at four elevation rings (30°, 45°, 60°, 90°) and 24 azimuth positions.

**Complexity range:**
- Smallest: 2 components, 1 connection
- Largest: 10+ components, 13+ connections
- Typical: 4–7 components, 3–9 connections

### Component vocabulary (29 types)

| Code | Part | Code | Part |
|------|------|------|------|
| `bo` | Bolt | `nu` | Nut |
| `pl` | Plug | `sl` | Sleeve |
| `wa` | Washer | `ti` | Tire |
| `no` | Nose connector | `rolo` | Long rod |
| `rome` | Medium rod | `rosm` | Short rod |
| `sclo` | Long screw | `scme` | Medium screw |
| `scsm` | Small screw | `whre` | Red wheel |
| `whwh` | White wheel | `blwo11` | Wooden block 1×1 |
| `blwo21` | Wooden block 2×1 | `plwo21` | Wooden plate 2×1 |
| `plwo31` | Wooden plate 3×1 | `plwo33` | Wooden plate 3×3 |
| `plwo53` | Wooden plate 5×3 | `plpl53` | Plastic plate 5×3 |
| `stwo3`–`stwo9` | Wooden straps (lengths 3–9) | `stpl5` | Plastic strap 5 |

---

## Repository Layout

```
00-project/
├── README.md                    ← this file
├── CHANGELOG.md
├── .gitignore
│
└── brio_pipeline/               ← all code + launcher scripts
    ├── slow.sh                  ← annotate samples with the slow pipeline
    ├── train_classifier.sh      ← train the component visual classifier (once)
    ├── visualize.sh             ← 3D/2D visualisation
    ├── labels.sh                ← export YOLO labels from slow outputs
    ├── calibrate.sh             ← build fixed camera rig calibration (once)
    ├── train.sh                 ← train YOLOv8n
    ├── infer.sh                 ← fast inference on a sample
    │
    ├── brio_3d_pipeline/        ← slow annotation pipeline (main system)
    │   ├── README.md            ← full technical documentation (start here)
    │   ├── pipeline.py          ← entry point
    │   ├── config.py            ← all paths and hyperparameters
    │   ├── puml_parser.py       ← reads ground-truth component manifest
    │   ├── preprocessor.py      ← fixed-scale crop around the construction
    │   ├── dust3r_runner.py     ← dense 3D reconstruction (DUSt3R ViT-L)
    │   ├── sam_runner.py        ← 2D instance masks (SAM ViT-B)
    │   ├── backprojector.py     ← 3D point clouds + voxel-overlap clustering
    │   ├── classifier.py        ← Hungarian assignment to PUML components
    │   ├── component_classifier.py   ← MobileNetV3 visual classifier
    │   ├── component_map.py     ← folder→class mapping + HSV prototypes
    │   ├── logger.py            ← auto timestamped logging
    │   ├── visualize_2d.py      ← 2D mask overlay per component
    │   ├── visualize.py         ← 3D point cloud export (PLY + PNG)
    │   ├── logs/                ← one .log per run (auto-created)
    │   └── outputs/
    │       └── run_NNN_YYYYMMDD_HHMM/
    │           └── sample_N/
    │               ├── image_order.json
    │               ├── cropped/          ← fixed-scale crops
    │               ├── dust3r/           ← pts3d cache
    │               ├── sam/              ← mask cache
    │               ├── proposals/        ← 3D cluster cache
    │               ├── results.json      ← final assignment
    │               ├── viz_2d.png
    │               └── viz_3d.ply
    │
    └── brio_fast_pipeline/      ← [WORK IN PROGRESS — not yet available]
        ...                        fast inference pipeline (YOLOv8 + triangulation)
        └── outputs/
            └── sample_N/
                ├── predicted.puml
                └── results.json
```

> **Note:** `outputs/` (DUSt3R/SAM caches, ~37 MB per sample) and `component_classifier.pth` are excluded from git via `.gitignore`.

---

## How to Run

All commands run from `brio_pipeline/`:

```bash
conda activate brio-3d
cd /mnt/c/BA/00-project/brio_pipeline
```

### Phase 0 — Train the visual classifier (once)

```bash
./train_classifier.sh
```

Trains MobileNetV3-small on isolated component images (~40 epochs, ~10 min GPU). Saves weights to `brio_3d_pipeline/component_classifier.pth`. Run once; retrain only if the component dataset changes.

### Phase 1 — Annotate samples (slow pipeline)

```bash
./slow.sh 113 114 115
```

Runs the full slow pipeline: CLAHE → DUSt3R → SAM → 3D clustering → Hungarian assignment. Creates a new timestamped output folder per run. First run: ~43 min/sample. Re-runs with cache: ~2 min/sample.

```bash
# Resume from an existing run (reuses DUSt3R and SAM caches):
cd brio_3d_pipeline
python pipeline.py --samples 113 --resume run_021_20260616_2339

# Visualise results:
python visualize_2d.py --sample 113
python visualize.py --sample 113
```

### Phases 2–5 — Fast pipeline (work in progress)

> **The fast pipeline (`brio_fast_pipeline/`) is not yet available.** Phases 2–5 (YOLO label export, camera rig calibration, YOLOv8 training, fast inference) are planned but not implemented. Only the slow pipeline (Phase 0 + Phase 1) is currently functional.

### Launcher scripts reference

| Script | Arguments | What it does | Status |
|--------|-----------|--------------|--------|
| `slow.sh` | `<ids...> [--device cpu]` | Slow pipeline: annotate samples | available |
| `train_classifier.sh` | `[--epochs N] [--batch N]` | Train MobileNetV3 visual classifier | available |
| `visualize.sh` | `<id> [<run_name>]` | 3D/2D plots for one sample | available |
| `labels.sh` | `[<ids...>]` | Export YOLO labels | work in progress |
| `calibrate.sh` | `<ids...>` | Build fixed camera rig | work in progress |
| `train.sh` | `[--epochs N] [--batch N]` | Train YOLOv8n | work in progress |
| `infer.sh` | `<id> [--puml <path>]` | Fast inference on one sample | work in progress |

Logs are written automatically on every run — no manual redirection needed. Each script writes a timestamped `.log` file to the relevant `logs/` folder. To follow live:

```bash
tail -f brio_3d_pipeline/logs/20260618_222000_samples_113.log
```

---

## Pipeline Overview

### Slow pipeline (`brio_3d_pipeline/`) — produces 3D ground-truth labels

```
~32 multi-view images (4 elevation rings, adaptive stride)
  ↓  puml_parser.py   — read component manifest from PUML
  ↓  preprocessor.py  — fixed-scale crop centred on the LCC
  ↓  dust3r_runner.py — dense 3D reconstruction (pts3d per pixel)
  ↓  sam_runner.py    — SAM prompted mode: tight-crop + fg-grid prompts
  ↓  backprojector.py — masks × pts3d → 3D clouds → voxel-overlap clustering
  ↓  classifier.py    — Hungarian assignment: colour + size + elongation costs
outputs/run_NNN/sample_N/results.json
```

Full technical documentation (every file, every function, known failure modes):  
**[brio_pipeline/brio_3d_pipeline/README.md](brio_pipeline/brio_3d_pipeline/README.md)**

### Fast pipeline (`brio_fast_pipeline/`) — work in progress

Planned: YOLOv8n detection trained on slow pipeline labels + fixed camera rig triangulation for sub-second inference. Not yet implemented.

---

## Environment Setup

Hardware: RTX 2070 Super (8 GB VRAM), WSL2 on Windows 11, CUDA driver 12.7.

### Why cu124 and not cu130

The CUDA driver version reported by `nvidia-smi` is the *maximum* CUDA version the driver supports — not the version PyTorch must be built against. Installing `torch+cu130` on a 12.7 driver raises a silent `CUDA initialization` error and falls back to CPU.

```bash
nvidia-smi   # top-right corner: "CUDA Version: 12.7"
```

Always match the `+cuXXX` suffix to the driver version or lower. `cu124` is the highest stable PyTorch build compatible with CUDA 12.7.

### Install

```bash
conda create -n brio-3d python=3.10 -y
conda activate brio-3d

# PyTorch — cu124, not cu130 (see above)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# DUSt3R dependencies (repo added to sys.path at runtime, not installed as a package)
pip install -r /mnt/c/BA/07-dust3r/requirements.txt

# SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Core
pip install roma einops trimesh scipy tqdm huggingface-hub \
            opencv-python scikit-learn matplotlib pillow

# Fast pipeline
pip install ultralytics
```

### Verify GPU

```bash
python -c "
import torch
print('torch:', torch.__version__)           # 2.6.0+cu124
print('CUDA:', torch.cuda.is_available())    # True
print('GPU:', torch.cuda.get_device_name(0)) # NVIDIA GeForce RTX 2070 Super
"
```

### Verify DUSt3R

```bash
python -c "
import sys; sys.path.insert(0, '/mnt/c/BA/07-dust3r')
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
print('DUSt3R OK')
"
```

You will see a `Warning, cannot find cuda-compiled version of RoPE2D` message. This is harmless — the pipeline uses the pure-PyTorch fallback, which is slightly slower but fully correct. Compiling the CUDA extension is not needed.

### Weights

**SAM ViT-B** (download once):
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O /mnt/c/BA/03-code/sam_weights/sam_vit_b_01ec64.pth
```

**DUSt3R ViT-L** downloads automatically from HuggingFace on first use (~1.6 GB, cached in `~/.cache/huggingface/`).

**Classifier weights** are produced by `./train_classifier.sh` (run once, ~10 min).

### VRAM usage by stage

| Stage | Peak VRAM | Notes |
|-------|-----------|-------|
| DUSt3R (batch=1) | ~4–6 GB | Freed before SAM starts |
| SAM ViT-B | ~4 GB | |
| Back-projection + clustering | CPU only | |
| Hungarian assignment | CPU only | |

DUSt3R and SAM never run simultaneously — GPU memory is freed between stages (`del model; torch.cuda.empty_cache()`).

**If DUSt3R OOMs:** reduce `DUST3R_SIZE = 384` in `config.py`, or drop to fewer images per ring (`IMAGE_VIEWS_PER_RING = 6`).  
**If SAM OOMs:** switch to `SAM_MODEL_TYPE = "vit_b"` (already default) or reduce `SAM_POINTS_SIDE`.

---

## References

1. Wang et al. (2024) — *DUSt3R: Geometric 3D Vision Made Easy*
2. Kirillov et al. (2023) — *Segment Anything*
3. Jocher et al. (2023) — *Ultralytics YOLOv8*
4. Liu et al. (2022) — *PETR: Position Embedding Transformation for Multi-View 3D Object Detection*
5. Howard et al. (2019) — *Searching for MobileNetV3*
