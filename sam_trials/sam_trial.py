"""
SAM trial run on BRIO construction samples 31-41.
Generates automatic masks, visualizes results, validates against PUML,
and writes a per-sample JSON summary to 11-experiments/.
"""

import os, re, json, cv2, torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict
# [SAM] Public API — sam_model_registry maps model type strings to model classes;
#       SamAutomaticMaskGenerator wraps the full automatic pipeline (grid prompts → masks).
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ── Paths ─────────────────────────────────────────────────────────────────────
# [CUSTOM] Project-specific paths and batch-folder layout for the BRIO dataset.
CHECKPOINT   = Path("/mnt/c/BA/03-code/sam_weights/sam_vit_b_01ec64.pth")
MODEL_TYPE   = "vit_b"
CONSTRUCTIONS = Path("/mnt/c/BA/02-resources/data/constructions")
OUTPUT_DIR   = Path("/mnt/c/BA/11-experiments")
VIS_DIR      = OUTPUT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)0000

# [CUSTOM] Maps sample numbers to their enclosing batch folder.
#          150 samples are split across five folders; this dict drives get_sample_dir().
BATCH_DIRS = {
    range(1,  32):  "Sample_1_to_31",
    range(32, 52):  "Sample_32_to_51",
    range(52, 82):  "Sample_52_to_81",
    range(82, 112): "Sample_82_to_111",
    range(112, 151):"Sample_112_to_150",
}

# [CUSTOM] Resolves a sample number to its InstanceDiagram directory on disk.
def get_sample_dir(n):
    for r, batch in BATCH_DIRS.items():
        if n in r:
            return CONSTRUCTIONS / batch / f"Sample_{n}_InstanceDiagram"
    raise ValueError(f"Sample {n} not in any batch")

# [CUSTOM] Finds the construction photo (.jpeg or .jpg) for a given sample.
def get_image(n):
    d = get_sample_dir(n)
    for ext in ("Construction.jpeg", "Construction.jpg"):
        p = d / ext
        if p.exists():
            return p
    raise FileNotFoundError(f"No construction image for Sample {n}")

# [CUSTOM] Parses InstanceDiagramSN.puml to extract the ground-truth component manifest.
#          PUML objects look like:  object "bo_1 : Component"
#          The parser strips the numeric suffix (e.g. "bo_1" → "bo") and counts
#          occurrences per abbreviation, returning e.g. {"bo": 3, "sb": 2}.
#          sum(manifest.values()) is then the total number of components SAM must find.
def parse_puml(n):
    """Returns {abbr: count} from InstanceDiagramSN.puml."""
    path = get_sample_dir(n) / f"InstanceDiagramS{n}.puml"
    text = path.read_text()
    # handles both 'bo_1 : Component' and 'bo_1: Component'
    pattern = re.compile(r'object\s+"([^"]+?)\s*:\s*Component"')
    counts = defaultdict(int)
    for m in pattern.finditer(text):
        label = m.group(1).strip()
        abbr  = label.rsplit("_", 1)[0]
        counts[abbr] += 1
    return dict(counts)

# [CUSTOM] Thin wrapper: reads an image with OpenCV and converts BGR→RGB for SAM/matplotlib.
def load_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# [CUSTOM] Post-processing filter applied to SAM's raw output.
#          Drops masks that are too small (noise, < 300 px) or too large
#          (> 50 % of image area, i.e. background / table surface).
#          SAM itself has min_mask_region_area, but this adds the upper-area guard.
def filter_masks(masks, image_area):
    """Remove background and noise masks."""
    return [
        m for m in masks
        if m["area"] > 300
        and m["area"] < image_area * 0.5
    ]

# [CUSTOM] Helper that generates a random BGR-range colour for mask overlays.
def random_color():
    return [int(x) for x in np.random.randint(60, 230, 3)]

# [CUSTOM] Side-by-side visualisation: original image on the left,
#          coloured mask overlay + PUML manifest summary on the right.
#          Uses SAM mask fields: "segmentation" (bool array), "bbox" ([x,y,w,h]).
def visualize(image, masks, manifest, sample_n, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left — original image
    axes[0].imshow(image)
    axes[0].set_title(f"Sample {sample_n} — Original", fontsize=11)
    axes[0].axis("off")

    # Right — masks overlay
    overlay = image.copy()
    colors  = {}
    for i, m in enumerate(masks):
        color = random_color()
        colors[i] = color
        # [SAM] m["segmentation"] is a boolean H×W array produced by SAM;
        #       True pixels belong to this mask instance.
        seg = m["segmentation"]
        overlay[seg] = (overlay[seg] * 0.45 + np.array(color) * 0.55).astype(np.uint8)

    axes[1].imshow(overlay)
    for i, m in enumerate(masks):
        # [SAM] m["bbox"] is [x, y, width, height] in pixel coordinates, from SAM output.
        x, y, bw, bh = m["bbox"]
        rect = patches.Rectangle(
            (x, y), bw, bh,
            linewidth=1.2, edgecolor=(
                colors[i][0]/255, colors[i][1]/255, colors[i][2]/255
            ), facecolor="none"
        )
        axes[1].add_patch(rect)
        axes[1].text(x + 2, y - 4, str(i),
                     color="white", fontsize=7,
                     bbox=dict(facecolor="black", alpha=0.5, pad=1))

    puml_text = "PUML manifest:\n" + "\n".join(
        f"  {abbr}: {cnt}" for abbr, cnt in sorted(manifest.items())
    )
    puml_text += f"\n\nMasks found: {len(masks)}"
    puml_text += f"\nComponents expected: {sum(manifest.values())}"

    axes[1].set_title(f"Sample {sample_n} — SAM masks ({len(masks)} found)", fontsize=11)
    axes[1].axis("off")
    fig.text(0.51, 0.01, puml_text, ha="left", va="bottom",
             fontsize=8, family="monospace",
             bbox=dict(facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close()

# ── Load SAM ──────────────────────────────────────────────────────────────────
# [SAM] Standard model-loading pattern from the segment_anything library.
#       sam_model_registry["vit_b"] returns the ViT-Base SAM class; checkpoint
#       is the pretrained weights file downloaded from the SAM release page.
print(f"Loading SAM ({MODEL_TYPE}) from {CHECKPOINT.name} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT))
sam.to(device=device)

# [SAM] SamAutomaticMaskGenerator is the SAM library's zero-prompt pipeline:
#       it places a grid of points_per_side² prompts, predicts masks for each,
#       applies NMS, and returns a list of mask dicts.
# [CUSTOM] Parameters below are tuned for BRIO brick images:
#   - points_per_side=64       denser grid to catch small components
#   - pred_iou_thresh=0.85     only keep high-confidence masks
#   - stability_score_thresh=0.92  further filters unstable mask boundaries
#   - crop_n_layers=1          one extra crop pass improves small-object recall
#   - min_mask_region_area=300 SAM-side noise filter (matched to filter_masks lower bound)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=300,
    box_nms_thresh=0.5,
)
print("SAM loaded.\n")

# ── Run samples ───────────────────────────────────────────────────────────────
# [CUSTOM] Main evaluation loop over samples 31–41.
SAMPLES = list(range(31, 42))
summary = []

for n in SAMPLES:
    print(f"── Sample {n} ──────────────────────────────────────")
    try:
        img_path = get_image(n)
        image    = load_rgb(img_path)
        H, W     = image.shape[:2]
        # [CUSTOM] Ground truth from PUML: expected component count per class.
        manifest = parse_puml(n)
        expected = sum(manifest.values())

        print(f"  Image:    {img_path.name}  ({W}×{H}px)")
        print(f"  PUML:     {manifest}  →  {expected} components expected")

        # [SAM] Core inference call — returns list of dicts, each with:
        #       segmentation, area, bbox, predicted_iou, stability_score, etc.
        masks_raw = mask_generator.generate(image)
        # [CUSTOM] Drop background and noise masks using area thresholds.
        masks     = filter_masks(masks_raw, H * W)

        # [SAM] predicted_iou and stability_score are quality metrics produced
        #       by SAM's mask decoder; used here only for logging/diagnostics.
        ious   = [round(m["predicted_iou"], 3)   for m in masks]
        stabs  = [round(m["stability_score"], 3) for m in masks]
        areas  = [m["area"] for m in masks]

        # [CUSTOM] Validation: compare mask count to PUML-derived expected count.
        match  = len(masks) == expected
        status = "OK" if match else ("SHORT" if len(masks) < expected else "OVER")

        print(f"  Masks:    {len(masks_raw)} raw → {len(masks)} after filter")
        print(f"  Status:   {status}  (found {len(masks)}, expected {expected})")
        if ious:
            print(f"  IoU:      min={min(ious):.3f}  max={max(ious):.3f}  avg={sum(ious)/len(ious):.3f}")
            print(f"  Areas:    min={min(areas)}  max={max(areas)}  avg={int(sum(areas)/len(areas))}")

        vis_path = VIS_DIR / f"sample_{n:03d}_masks.png"
        visualize(image, masks, manifest, n, vis_path)
        print(f"  Saved:    {vis_path.name}")

        summary.append({
            "sample":           n,
            "image":            str(img_path),
            "image_size":       [W, H],
            "puml_manifest":    manifest,
            "expected_components": expected,
            "masks_raw":        len(masks_raw),
            "masks_filtered":   len(masks),
            "status":           status,
            "iou_min":          min(ious) if ious else None,
            "iou_max":          max(ious) if ious else None,
            "iou_avg":          round(sum(ious)/len(ious), 3) if ious else None,
            "area_min":         min(areas) if areas else None,
            "area_max":         max(areas) if areas else None,
            "visualization":    str(vis_path),
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        summary.append({"sample": n, "error": str(e)})

    print()

# ── Save summary ──────────────────────────────────────────────────────────────
# [CUSTOM] Persist per-sample results as JSON for later analysis.
summary_path = OUTPUT_DIR / "sam_trial_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary written to {summary_path}")

ok    = sum(1 for s in summary if s.get("status") == "OK")
short = sum(1 for s in summary if s.get("status") == "SHORT")
over  = sum(1 for s in summary if s.get("status") == "OVER")
err   = sum(1 for s in summary if "error" in s)
print(f"\nResults: {ok} OK  |  {short} SHORT  |  {over} OVER  |  {err} ERROR")
