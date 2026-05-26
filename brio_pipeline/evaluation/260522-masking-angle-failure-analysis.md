# Masking Failure Across View Angles — Root Cause Analysis

**Observed:** 2D visualisation shows correct-looking masks for one elevation ring but
messy, inconsistent, or "rotating silhouette" labels across all other angles.

---

## 1. SAM runs independently on every view

`sam_runner.py` runs `SamAutomaticMaskGenerator` on each of the 14 images separately
(8 × Images90 + 6 × Images45). SAM has no cross-view awareness — it segments whatever
visual boundaries it finds in each image. At 90° (top-down) a component looks like a
flat circle or rectangle; at 45° it has visible height and connections. The resulting
masks are geometrically unrelated across angles even if they cover the same physical
component.

## 2. Top-N selection by IoU does not guarantee the right objects

Only the N masks with the highest `predicted_iou` are kept per image. At difficult
angles (heavy foreshortening, occlusion, overlapping components) the top-N masks may
cover the background, cast shadows, white padding, or merged blobs rather than the N
individual components. There is no class-level or shape-level prior to guide selection.

## 3. DUSt3R is unreliable on textureless plastic

DUSt3R builds `pts3d` by matching features across views. BRIO components are smooth,
uniformly coloured plastic — low texture, many repeated shapes, specular highlights.
Feature matching fails or produces noisy correspondences, so the 3D coordinates
assigned to mask pixels are inaccurate. Masks from different views for the same object
land at different 3D locations instead of converging.

## 4. The 6D clustering feature collapses on similar components

Each mask is represented as `[3D centroid (3) | mean HSV (3)]`. When pts3d is noisy
(issue 3) the centroid is wrong. When components share the same plastic colour (common
in BRIO) HSV provides no discriminating signal. Agglomerative clustering then groups
masks by whichever residual geometry survives the noise — which is arbitrary.

## 5. The "rotating shape" effect

Issues 1–4 combine to produce this visual artefact: one elevation ring (usually
Images45, where side-on views give SAM the clearest object boundaries) produces
plausible masks that land in roughly the right 3D region. The other ring (Images90,
top-down) produces masks that don't co-localise in 3D with the first ring's masks, so
the clustering assigns them to whichever cluster centroid happens to be nearest. When
rendered back into 2D they appear as the same silhouette projected from a rotating
camera — no new information is being added, the multi-view step is not working.

## 6. White padding from fixed-scale crop adds spurious masks

`preprocessor.py` pads smaller constructions to the largest crop window with white
pixels. SAM treats the uniform white border as a segmentable region and often returns
it as a high-confidence mask, consuming one of the N slots.

---

## Summary of failure chain

```
Textureless plastic
  → DUSt3R pts3d noise
    → 3D centroids are unreliable
      → Clustering groups wrong masks together
        → Wrong instance assignments in all views except the best-lit one
          → Messy 2D labels + rotating-shape artefact in visualisation
```

---

## Implication for the fast pipeline

Training YOLOv8 on labels derived from this pipeline would propagate all of the above
errors into the detector. The cleaner path is to generate 2D bounding-box labels
directly — either by annotating the main `Construction.jpg` with SAM and projecting
via the pre-calibrated rig, or by running SAM per-view with PUML-guided interactive
prompts (one prompt point per known component).
