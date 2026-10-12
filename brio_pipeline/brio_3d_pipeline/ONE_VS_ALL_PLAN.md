# One-vs-All Classification — Implementation Plan

**Branch:** `brio3d-improvement/free-n-mode`
**Worktree:** `/mnt/c/BA/00-free-n-mode` (kept separate from `/mnt/c/BA/00-project`, which stays on `dev` as the untouched safe baseline)
**Status:** planning complete, no code written yet. Phase 0-2 (manifest-enforcement mode toggle) is already done and verified in this worktree, independent of this plan.

---

## 1. What this replaces

`classification/component_classifier.py` — a single N-way softmax MobileNetV3-small, trained on the isolated `new_structure_Component` reference-image dataset — is being **replaced**, not supplemented. It gets commented out in place (not deleted, not deleted-and-git-rm'd), and the new implementation lives in a separate folder so the two are never accidentally entangled:

```
classification/
    classifier.py                  # Hungarian assignment (Phase 4 — untouched by this plan)
    component_classifier.py        # OLD softmax classifier — commented out in place, not deleted
    component_map.py               # shared (folder→class map, colour prototypes) — reused
    component_classifier.pth       # old weights — untouched, kept for rollback reference
    one_vs_all/                    # NEW — the entire one-vs-all implementation lives here
        __init__.py
        model.py                   # shared backbone + K binary sigmoid heads
        train.py                   # training script: imbalance handling, stratified split, metrics
        classifier.py              # inference wrapper — predict_batch() matches the interface
                                    #   compute_proposals() already calls, so pipeline.py/
                                    #   backprojector.py plumbing doesn't need to change shape
        visualize_predictions.py   # NEW — per-class contact sheets of validation predictions
                                    #   (true/false positives, false negatives) — the visual,
                                    #   qualitative counterpart to the precision/recall numbers
        weights/                   # trained checkpoints land here
    training/
        dataset_builder.py         # existing — produces the training data used here
        mapping_parser.py          # existing
        batch_build.py             # existing
        dataset_crops/             # existing 4,721 crops — the training data, used as-is
```

This is a **pure one-vs-all pipeline variant**: once wired up, `pipeline.py`'s `_load_classifier()` loads `classification/one_vs_all/classifier.py`'s model, and `component_classifier.py` is never imported. No dual-classifier comparison logic, no side-by-side scoring.

---

## 2. Why one-vs-all, and why the existing seed-derived crops are the training data

- **Softmax → one-vs-all**: a softmax classifier cannot express "this is none of my classes" — it must put probability mass somewhere. One-vs-all trains K independent binary heads (shared backbone, K sigmoid outputs), so a crop can legitimately get "no" from every head. This is what makes a genuine reject/background option possible, which free-mode clustering (Phase 2) structurally needs — it hands back clusters that may not correspond to a real object at all.

- **Why construction-image crops, not the isolated component dataset**: a raw construction photo shows 5-15 components wedged together — you cannot train a classifier on "this whole photo = nu," it's ambiguous. You need something that says *where* the component is before you can crop it out and label it. That's exactly what `Mapping.drawio` seed points already provide: one hand-placed 2D point per component instance per sample. `dataset_builder.py` lifts that point to 3D via DUSt3R, reprojects it into every other view of the sample (with an occlusion check — a view where the point is hidden gets skipped, not falsely cropped), and crops a fixed window around it in each view. One point placement → many real, multi-angle crops of the same physical part, automatically. That mechanism already solved the localization problem *and* the multi-view problem — no SAM, no Roboflow, no new annotation work needed to use it.

- **The labeling work is already done**: 149 of 150 samples already have parsed `Mapping.drawio` mappings, and 40 of those have already been converted into 4,721 real, labeled, construction-domain crops across 29 classes + `_bg`. Training needs zero new labeling to start.

- **Isolated component images (`new_structure_Component`) are deliberately excluded from training**, not just unused by default. The *old* classifier trained on exactly that dataset and — per the 2026-07-02 changelog entry — scored 29/29 on clean isolated test images but collapsed to ~100% one class on real deployment crops (a domain-mismatch failure: different background, scale, lighting, occlusion state). Blending isolated images into one-vs-all training risks the same failure mode. See Section 9 for the conditions under which this gets reconsidered, later, per-class, as a measured experiment — not a default.

---

## 3. Model architecture

- Shared MobileNetV3-small backbone (same family as the old classifier) + K independent binary sigmoid heads, one per class.
- Each head trained with its own binary cross-entropy: positives = that class's crops, negatives = every other class's crops + `_bg`. A single training image updates all K heads at once (one "yes" target for its own class, "no" targets for the rest) — not K separate training passes.
- Inference (`one_vs_all/classifier.py`): `predict_batch()` returns `(cls_or_None, confidence)` per crop — same shape the existing `compute_proposals()` call site expects, so integration doesn't require touching `backprojector.py`'s plumbing, only what's inside the classifier.

---

## 4. Class imbalance handling

Real numbers from the training set: `_bg` 2,144 / `nu` 575 down to `rolo` 6, `whwh` 8, `no` 16, `plwo53` 17. Without countermeasures, a head can learn to just always say "no" and still be right >99% of the time by volume alone — the loss barely notices the rare misses.

- **Per-class `pos_weight`** in BCE loss (`n_negative / n_positive` per head) — makes missing the rare class actually cost something during training.
- **Class-balanced batch sampling** (`WeightedRandomSampler`, oversampling thin-class positives) — complements pos_weight by shaping gradient batches directly, not just the loss scale.
- **Thin-class-biased augmentation** — heavier augmentation multiplier for classes under a count threshold (rotation, colour jitter, crop jitter), reusing the existing heavy-augmentation approach from `component_classifier.py`'s `train_classifier()`. Manufactures visual variety from the few real examples that exist; doesn't manufacture real data.
- **Per-head threshold calibration** — each head's decision threshold tuned on validation data individually, not a uniform 0.5 across heads with wildly different training set sizes.
- **Honest limit**: for a class with single-digit real examples (`rolo`: 6), none of the above fully solves the underlying problem — there isn't enough visual variety in 6 photos to learn robustly. The real fix is more data, deferred until per-class results show it's actually needed (Section 8, step 4) rather than assumed upfront.

---

## 5. Validation split — leak-proof by construction

Crops from the same `sample_id` always go **entirely** into train or **entirely** into val, never split across both. Coarser than instance-level grouping, but trivially avoids the leakage risk where two views of the same physical object end up on both sides of the split, inflating validation accuracy.

---

## 6. Visual review — `visualize_predictions.py`

After training, this script runs the model over the held-out validation crops and builds one contact-sheet image per class:
- **Green-bordered thumbnails** — predicted this class, correctly (true positive).
- **Red-bordered thumbnails** — predicted this class, incorrectly (false positive) — the actual true class is printed under the thumbnail.
- **A separate "missed" section** — crops whose true label is this class but the model predicted something else or nothing (false negative).

This is the qualitative counterpart to Section 7's precision/recall numbers: a number says `nu` scored 0.85 F1, the contact sheet shows *which* crops it's actually getting wrong and whether that's a real, fixable pattern (e.g. one bad viewing angle) or just noise. Re-runnable after every retrain, not a one-off.

---

## 7. Reliability criteria

The yardstick for evaluating the implementation once it's built and tested — check against these before deciding it's ready, not just "does it run."

### Data
- [ ] Train/val split is verified sample-level with an automated check (assertion in `train.py`, not just a design intention) — zero `sample_id` overlap between splits.
- [ ] Every class has at least one validation example; any class that doesn't is explicitly flagged as "insufficient data for a reliable estimate," never silently dropped from the report.
- [ ] `_bg` negative pool remains substantial relative to the largest class.

### Training
- [ ] Training is seeded and reproducible — two runs on identical data produce identical final metrics (matches the determinism standard already established for the rest of this pipeline, see the `_extract_mask_cloud` seed fix in `backprojector.py`).
- [ ] Per-class `pos_weight` values are logged/printed at training start, not silently computed and hidden.
- [ ] Train and validation loss curves are logged per epoch.

### Accuracy — the core criteria
- [ ] Per-class precision, recall, and F1 reported on the held-out validation split — never a single blended accuracy number.
- [ ] Classes with sufficient data (a stated minimum, e.g. N≥30 validation positives) must be evaluated against an explicit target (e.g. F1 ≥ 0.7 — adjustable once real numbers exist); classes below that data floor are reported separately as "not yet judgeable," not averaged in with the rest.
- [ ] False-positive rate on `_bg` is reported explicitly — the direct measure of whether the reject/"none" mechanism (the entire reason for choosing one-vs-all) actually works.
- [ ] Confidence separation check: correct predictions' confidence distribution is meaningfully higher than incorrect predictions'.

### Qualitative
- [ ] `visualize_predictions.py` contact sheets generated for every class, reviewed manually (Section 6) — the numbers and the visuals must agree; a class with a good F1 but visually wrong-looking true positives is a red flag, not a pass.

### Integration (forward-looking — applies once wired into the pipeline)
- [ ] `predict_batch()`'s interface is verified compatible with `compute_proposals()`'s existing call site by an actual test run.
- [ ] `component_classifier.py` is confirmed inert — grepped for, zero remaining imports/call sites anywhere in `pipeline.py`/`backprojector.py`/`classifier.py`.
- [ ] An end-to-end pipeline run using the new classifier on at least one real sample completes without crashing and produces a sane `results.json`.

---

## 8. Build order

1. `classification/one_vs_all/model.py` + `train.py` (imbalance handling, sample-level split, per-class metrics — Sections 3-5, 7)
2. Train on the existing 4,721 construction-domain crops.
3. `classification/one_vs_all/visualize_predictions.py` — generate contact sheets (Section 6).
4. Evaluate against Section 7 in full (numbers + visuals). For any class that fails: decide per-class whether the fix is more augmentation, more real data (Section 9), or accept as a known limitation.
5. Wire into `pipeline.py`/`backprojector.py`, comment out `component_classifier.py`.
6. Re-verify Section 7's integration criteria end-to-end.

---

## 9. Deferred / optional levers (not part of the initial build)

- **Isolated-image augmentation**: for a specific thin class only, after step 4 shows it's genuinely needed — added as a controlled experiment (does real-crop validation for that class improve or regress?), never blended in wholesale.
- **More real data via continued `batch_build.py` runs**: `parsed_mappings/` already has 149/150 samples ready; only 40 have been converted to crops. `batch_build.py`'s existing greedy set-cover ordering already prioritizes rare-class coverage — running it further is the most direct lever if specific classes need more real examples. Cost: ~8-15 GPU-min/sample (DUSt3R).
- **SAM-proposed + Roboflow-reviewed crops**: considered and set aside for the initial build in favor of simplicity — the existing seed-derived pipeline already solves localization and multi-view coverage. Worth reconsidering later specifically if reprojection-drift errors turn up in the visual review (Section 6) that the current mechanism can't catch, or if trustworthiness of automatically-inferred labels becomes a concern independent of data volume.
- Exact numeric thresholds in Section 7 (F1 target, minimum N for judgeability) are placeholders — set once real validation numbers exist from the first training run.
