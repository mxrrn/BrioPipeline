# From Multi-View 2D to 3D Object Detection and Structure Generation in PlantUML

**Bachelor's Thesis:** 3D Object Detection from Multi-View 2D Input Images of Multi-Component Constructions and Knowledge-Aware Structure Diagram Generation in PlantUML

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
| `no` | Nose connector | `ro_lo` | Long rod |
| `rome` | Medium rod | `rosm` | Short rod |
| `sclo` | Long screw | `scme` | Medium screw |
| `scsm` | Small screw | `whre` | Red wheel |
| `whwh` | White wheel | `blwo11` | Wooden block 1×1 |
| `blwo21` | Wooden block 2×1 | `plwo21` | Wooden plate 2×1 |
| `plwo31` | Wooden plate 3×1 | `plwo33` | Wooden plate 3×3 |
| `plwo53` | Wooden plate 5×3 | `plpl53` | Plastic plate 5×3 |
| `stwo3`–`stwo9` | Wooden straps (lengths 3–9) | `stpl5` | Plastic strap 5 |

---

## Implemented Solution

The implemented system is a two-stage pipeline.  
Full documentation: **[brio_pipeline/README.md](brio_pipeline/README.md)**

### Stage 1 — Slow Annotation Pipeline (`brio_3d_pipeline/`)

Runs once per sample to produce 3D ground-truth labels for training. Runtime ~35 minutes per sample on GPU.

```
14 multi-view images
  ↓  DUSt3R        — uncalibrated multi-view 3D reconstruction (pts3d per pixel)
  ↓  SAM (ViT-B)   — automatic mask generation (top-N masks per image)
  ↓  Back-project  — SAM masks × pts3d → 14×N raw 3D point clouds
  ↓  Ward clustering + sigma cleanup → N instance clouds
  ↓  Hungarian assignment (geometry + colour cost) → class label per cloud
outputs/sample_N/results.json   — instances with 3D centroids and class labels
```

Outputs feed directly into Stage 2 as YOLO training labels.

### Stage 2 — Fast Inference Pipeline (`brio_pipeline/brio_fast_pipeline/`)

Trained on Stage 1 labels. Inference is under 2 seconds per sample on GPU.

```
Training (once):
  label_exporter.py  →  YOLO-format dataset from Stage 1 outputs
  train.py           →  fine-tunes YOLOv8n from COCO pretrained weights

Calibration (once):
  calibrator.py      →  fixed camera rig from DUSt3R poses (normalised, averaged)

Inference:
  14 images → YOLOv8 detection → DLT triangulation → connection inference → PlantUML
```

### Launcher scripts

All phases are run from `brio_pipeline/` with short commands:

| Command | What it does |
|---------|--------------|
| `./slow.sh 113 114 115` | Annotate samples with the slow pipeline |
| `./labels.sh` | Export YOLO labels from completed samples |
| `./calibrate.sh 113` | Build fixed camera rig calibration |
| `./train.sh` | Train YOLOv8n |
| `./infer.sh 120` | Run fast inference on a sample |
| `./visualize.sh 113` | 3D scatter plot of instance clouds |

Logs are written automatically on every run.

---

## Repository Layout

```
00-project/
├── README.md               ← this file
├── .gitignore
│
├── brio_3d_pipeline/       ← slow annotation pipeline (source code)
│   ├── pipeline.py
│   ├── config.py
│   ├── backprojector.py
│   ├── classifier.py
│   └── ...
│
├── brio_pipeline/          ← fast pipeline + launcher scripts
│   ├── README.md           ← full pipeline documentation
│   ├── slow.sh / infer.sh / train.sh / ...
│   └── brio_fast_pipeline/
│       ├── infer.py
│       ├── train.py
│       └── ...
│
└── sam_trials/             ← earlier SAM integration experiments
```

> **Note:** `brio_3d_pipeline/outputs/` (DUSt3R/SAM caches, ~37 MB per sample) is excluded from git via `.gitignore`. Run `./slow.sh <sample_ids>` to regenerate.

---

## Environment

- Python 3.10, conda env `brio-3d`
- PyTorch + CUDA 12.4 (`cu124`) on RTX 2070 Super (8 GB)
- DUSt3R (ViT-L, `07-dust3r/`), SAM ViT-B, YOLOv8n (ultralytics)
- WSL2 on Windows 11

Setup instructions: [brio_pipeline/README.md § Environment Setup](brio_pipeline/README.md#10-environment-setup)

---

## References

1. Wang et al. (2024) — *DUSt3R: Geometric 3D Vision Made Easy*
2. Kirillov et al. (2023) — *Segment Anything*
3. Jocher et al. (2023) — *Ultralytics YOLOv8*
4. Liu et al. (2022) — *PETR: Position Embedding Transformation for Multi-View 3D Object Detection*
