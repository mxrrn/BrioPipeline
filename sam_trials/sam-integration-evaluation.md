# SAM Integration Evaluation: Component Annotation and Multi-View Object Detection

**Context:** Bachelor's Thesis — BRIO construction component detection from multi-view 2D images  
**Goal:** Use SAM masking to annotate the 30 BRIO component types, then train a CNN for object
detection on multi-view images that runs **under 5 seconds** per construction.

---

## 1. Problem Framing

The thesis dataset consists of:
- **~150 annotated constructions**, each with:
  - `Construction.jpeg` — photograph of the physical assembly
  - `InstanceDiagramSN.puml` — human-readable PlantUML diagram listing every component
    instance (`bo_1`, `stwo3_2`, etc.), all slots, and all connections
  - `InstanceDiagramSN_AI.puml` — same diagram with integer IDs instead of abbreviations
  - `Mapping.drawio` — visual overlay mapping photo regions to component labels
- **Multi-view image sets** per sample: 4 elevation angles (30°, 45°, 60°, 90°), with
  24 images at 30°/45°/60° and 6 images at 90° — approximately **78 images per construction**
- **30 component classes** with known slot types and connection rules

The `Mapping.drawio` files are visual and approximate; they are not directly parseable
into bounding boxes. The **PlantUML files are the authoritative, machine-readable ground
truth** — they enumerate every component instance and type present in each construction,
which directly tells the annotator which class labels to assign to SAM-generated masks.
SAM can bridge the gap between this structural ground truth and pixel-level spatial labels.

---

## 2. Why SAM Is a Strong Fit Here

SAM's zero-shot segmentation capability is particularly well-suited to this dataset for
three reasons:

1. **Visual distinctiveness of BRIO parts** — components differ in shape, size, and color
   (bolts are thin cylinders, star wheels are star-shaped, plates are flat grids). SAM's
   boundary detection is shape-driven, not class-driven, so it will naturally isolate
   individual parts without any category knowledge.

2. **Small dataset problem** — 150 training samples is too small for standard supervised
   segmentation training. SAM provides high-quality pseudo-labels for free, enabling
   supervised CNN training without a large manually annotated corpus.

3. **Dense spatial coverage** — with 78 images per construction at different viewpoints,
   the same physical component appears in many images from different angles. SAM can
   annotate each view independently; consistent component identities across views can then
   be linked via the structural diagram ground truth.

---

## 3. Proposed Pipeline

```
OFFLINE (annotation phase — SAM runs here, no time constraint)
─────────────────────────────────────────────────────────────────
Construction.jpeg
        │
        ▼
SAM Automatic Mask Generator
(points_per_side=32, multi-scale crops)
        │
        ▼
Per-image mask candidates (unclassified)
        │
        ▼
Human verifies + assigns class labels
(match masks to component abbreviations using Mapping.drawio as reference)
        │
        ▼
Labeled dataset:
  - bounding boxes  →  YOLO format  (x_center, y_center, w, h, class_id)
  - instance masks  →  COCO format  (for mask-based training)
        │
        ├── Apply to Construction.jpeg for all ~150 samples
        └── Apply to multi-view images (Images30/45/60/90) per sample

TRAINING (one-time, offline)
─────────────────────────────────────────────────────────────────
Labeled multi-view images
        │
        ▼
Train CNN detector (YOLOv8 or EfficientDet-D0)
on all viewpoints jointly (treat each image as independent training example)
        │
        ▼
Trained .pt weights (~5–40 MB)

INFERENCE (online — must be < 5 seconds per construction)
─────────────────────────────────────────────────────────────────
New construction images (4 views × N angles)
        │
        ▼
CNN detector runs on each image (~1–5ms/image on GPU, ~30–100ms on CPU)
        │
        ▼
Per-image detections: {class_id, confidence, bbox}
        │
        ▼
Multi-view aggregation (NMS across views, majority voting per component)
        │
        ▼
Component list → map to PlantUML via slot rules
```

---

## 4. Integration Strategy Options

Three distinct ways SAM can be incorporated — evaluated below.

---

### Option A: SAM as Offline Annotator Only (Recommended)

**Description:** SAM runs only during dataset preparation. The deployed detector is a
pure CNN with no SAM dependency at inference time.

**Workflow:**
1. Run `SamAutomaticMaskGenerator` on each construction image → get per-instance masks
2. Parse the corresponding `InstanceDiagramSN.puml` to get the exact component list
   and instance counts (e.g., "this construction has 1 bolt, 2 star wheels, 1 washer")
3. Human annotator matches SAM masks to the known component list — the PUML tells you
   exactly what to look for, reducing labeling to a confirmation task rather than
   open-ended identification; `Mapping.drawio` serves as a spatial reference for ambiguous cases
3. Export as YOLO bounding boxes or COCO instance masks
4. Train YOLOv8 on these labels
5. At inference: only YOLO runs — SAM is not present

**Advantages:**
- Fully meets the 5-second constraint (YOLO runs in <100ms per image on CPU)
- SAM annotation is ~10× faster than manual bounding-box drawing
- No GPU required at deployment time
- Consistent labels: SAM masks are more precise than hand-drawn boxes, reducing label noise

**Disadvantages:**
- Human verification step still required (SAM may merge touching components)
- SAM has no concept of component identity — the same component type may produce
  differently shaped masks across viewpoints (not a problem if treated as independent detections)

**Verdict:** Best option for thesis scope. Separates the expensive annotation tool from
the lightweight deployable detector.

---

### Option B: SAM + CNN at Inference (Two-Stage Detector)

**Description:** At inference time, SAM first segments the image into region proposals.
A lightweight CNN classifier then assigns a component class to each proposal.

**Workflow:**
1. SAM automatic masks → crop each masked region
2. Crop → ResNet-18 / MobileNetV3 classifier → component class
3. Aggregate across views

**Advantages:**
- No bounding box annotation needed — only image-level component presence labels per crop
- Classifier training is simpler than full detection training
- SAM handles all localization, CNN only classifies

**Disadvantages:**
- **SAM ViT-H encoder: ~50ms/image on GPU, ~5–10s on CPU** — with 78 images per
  construction, the total SAM encoding time alone would be ~390ms (GPU) to ~780s (CPU)
- The 5-second constraint is violated on CPU; only viable with GPU
- If SAM merges two touching components (e.g., a bolt through a plate), the
  downstream classifier receives an ambiguous crop with no way to recover

**Verdict:** Fails the 5-second constraint on CPU. Only viable as a GPU-only pipeline,
and even then margins are tight at 78 images. Not recommended for the thesis.

---

### Option C: SAM-Distilled Lightweight Segmenter

**Description:** Use SAM to generate pseudo-labels, then train a lightweight segmentation
network (e.g., MobileNet-based UNet or YOLO-seg) that mimics SAM but runs faster.

**Workflow:**
1. SAM generates masks for all training images
2. Train a small segmentation model on these pseudo-labels (SAM as the teacher)
3. The trained model replaces SAM entirely at inference

**Advantages:**
- Fast at inference — MobileNet-based segmentation runs in <5ms/image
- No manual bounding box annotation required — SAM labels drive training
- Better generalization than a pure detector because it learns object shapes

**Disadvantages:**
- Requires semantic labels (SAM gives unlabeled masks; someone must link masks → class IDs)
- More complex training setup (segmentation training is harder than detection training)
- With only 150 samples, pseudo-label quality may not be sufficient to train a stable
  segmentation model from scratch — transfer learning from COCO required

**Verdict:** Technically sound but adds significant complexity for marginal gain over Option A.
Worth considering as a future extension but not for the initial implementation.

---

## 5. Speed Analysis for the 5-Second Constraint

The constraint is **< 5 seconds per construction at inference**.

Each construction has ~78 images (24+24+24+6). The inference budget per image is:

| Model | Hardware | Per-image latency | 78-image total | Feasible? |
|-------|----------|------------------|----------------|-----------|
| YOLOv8n | CPU | ~30–50ms | 2.3–3.9s | Yes |
| YOLOv8s | CPU | ~60–100ms | 4.7–7.8s | Marginal |
| YOLOv8n | GPU | ~1–3ms | <0.25s | Yes |
| EfficientDet-D0 | CPU | ~60ms | ~4.7s | Marginal |
| SAM ViT-H | CPU | ~5,000ms | infeasible | No |
| SAM ViT-B | GPU | ~25ms | ~2s | GPU only |

**Recommendation:** YOLOv8n on CPU comfortably meets the constraint. If GPU is available,
any YOLO variant is trivially fast. SAM must not appear in the inference path.

Note: not all 78 images need to be processed — a strategic subset (e.g., one image per
azimuth at 45°, ~8 images) would reduce inference time to <400ms with YOLOv8n on CPU while
still providing good multi-view coverage.

---

## 6. Handling the Small Dataset Problem

With ~150 samples and 30 classes, per-class instance count will be low. SAM helps in
several ways:

### 6.1 Copy-Paste Augmentation
Once SAM masks are available, individual component instances can be cut out and pasted
onto new backgrounds at different positions, scales, and orientations. This is a
well-established technique (SIMPLE COPY-PASTE, Ghiasi et al. 2021) that can multiply the
effective training set size by 5–10×.

### 6.2 Component-Level Crops for Pre-Training
Each SAM mask can be cropped and used to pre-train a component classifier independently of
the detection head. A classifier trained on isolated component crops generalizes better than
one trained only in the detection setting where components are surrounded by distractors.

### 6.3 PUML-Guided Annotation Validation

Before exporting any labels, the number of accepted SAM masks per class can be compared
against the component counts parsed from `InstanceDiagramSN.puml`. If SAM finds 3 masks
labeled as `bo` but the PUML lists only 1 bolt instance, the annotation is inconsistent
and needs correction. This programmatic check catches merge errors and mis-labelings
before they propagate into training data — it is only possible because the PUML provides
exact per-construction ground truth that the `Mapping.drawio` files do not.

### 6.4 Multi-View Label Propagation
If a component is labeled in the `Construction.jpeg`, the same label can be propagated to
its instances in the multi-view images by matching crops via template matching or feature
similarity. This amplifies each manual label into ~78 training instances.

---

## 7. Multi-View Aggregation Strategy

After per-image detections are collected from all views, they must be aggregated into a
single component list. Two approaches:

### 7.1 Majority Voting (Simple)
For each component class, count how many views detected it with confidence > threshold.
If count ≥ k (e.g., k=3), accept it as present. This is robust to per-view occlusion.

### 7.2 3D Consistency Check (Advanced)
If camera poses are known (from the multi-view image naming convention, which implies
fixed azimuth steps), detections across views can be cross-validated geometrically.
A component detected at conflicting positions across views is likely a false positive.
This requires calibrated camera parameters.

**Recommendation for thesis:** Start with majority voting. It requires no camera
calibration and is easy to tune via the threshold k.

---

## 8. Recommended Implementation Plan

### Phase 1 — SAM Annotation (offline, ~1–2 days)
1. Install SAM: `pip install git+https://github.com/facebookresearch/segment-anything.git`
2. Download ViT-H weights: `sam_vit_h_4b8939.pth`
3. For each sample, parse `InstanceDiagramSN.puml` to extract the ground-truth component
   list and instance counts — this becomes the label manifest for that image
4. Run `SamAutomaticMaskGenerator` on the corresponding `Construction.jpeg`
5. Match SAM masks to the manifest: the PUML tells you exactly how many instances of each
   class exist, so labeling is a 1-to-1 assignment problem, not open-ended identification;
   use `Mapping.drawio` as a spatial sanity check for ambiguous or merged masks
6. Export to YOLO format: one `.txt` per image with `class_id x_center y_center w h`
7. Extend labeling to a representative subset of multi-view images (at minimum, Images45)

### Phase 2 — CNN Training
1. Use YOLOv8n (nano) as base model — smallest and fastest, suitable for 30 classes
2. Fine-tune from COCO weights (transfer learning handles the small dataset)
3. Train for 100–300 epochs on augmented dataset (flip, rotate, color jitter, copy-paste)
4. Validate on a held-out split (e.g., 120 train / 30 val)

### Phase 3 — Multi-View Inference Pipeline
1. For a new construction: run YOLOv8n on selected views (e.g., 8 images at 45°)
2. Aggregate detections via majority voting (threshold k=3)
3. Map detected component list to PlantUML slots using the known slot-type rules
4. Generate structural graph prediction

---

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| SAM merges touching components | High | Use point prompts (one per component) instead of automatic mode; `Mapping.drawio` gives approximate spatial locations; `InstanceDiagramSN.puml` tells you how many separate masks to expect |
| 30 classes with too few instances per class | High | Copy-paste augmentation; group visually similar classes for initial experiments |
| Multi-view images not labeled | Medium | Propagate labels from Construction.jpeg using template matching |
| YOLOv8 struggles with small components | Medium | Use multi-scale training; add `imgsz=1280` for higher-res inputs |
| Ground truth PlantUML has ambiguous component identity | Low | Use anonymized `_AI.puml` which already uses integer IDs |

---

## 10. Summary and Recommendation

**Use SAM as an offline annotation accelerator (Option A), not as an inference component.**

The cleanest and most feasible pipeline for the thesis is:

1. SAM semi-automates the labeling of 150 construction images, reducing annotation time
   and producing more precise masks than manual bounding boxes
2. The resulting labels train a YOLOv8n detector on multi-view images
3. At inference, only YOLOv8n runs — achieving well under 5 seconds on CPU
4. Multi-view detections are aggregated with majority voting to produce a component list
5. The component list feeds into the existing structural topology prediction logic

This approach is modular, achieves the speed target, directly addresses the small-dataset
problem, and keeps SAM's cost entirely in the offline phase where it is unconstrained.

SAM-at-inference (Option B) should only be reconsidered if a GPU is guaranteed in the
deployment environment **and** the dataset proves too small for YOLOv8 to converge, in
which case the two-stage SAM+classifier approach provides a training-data-free fallback.
