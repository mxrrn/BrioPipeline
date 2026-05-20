"""
sam_trial_connections.py
Extended version of sam_trial.py that parses the full PUML structure
(Components, Slots, Connections, ConnectionConfigs) instead of just counting
component instances. Useful for understanding what slot/connection data is
available before building the slot-labeling and PUML-generation pipeline.

Writes per-sample JSON with full PUML graph to 11-experiments/.
"""

import os, re, json, cv2, torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT    = Path("/mnt/c/BA/03-code/sam_weights/sam_vit_b_01ec64.pth")
MODEL_TYPE    = "vit_b"
CONSTRUCTIONS = Path("/mnt/c/BA/02-resources/data/constructions")
OUTPUT_DIR    = Path("/mnt/c/BA/11-experiments")
VIS_DIR       = OUTPUT_DIR / "visualizations_conn"
VIS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_DIRS = {
    range(1,  32):  "Sample_1_to_31",
    range(32, 52):  "Sample_32_to_51",
    range(52, 82):  "Sample_52_to_81",
    range(82, 112): "Sample_82_to_111",
    range(112, 151):"Sample_112_to_150",
}

def get_sample_dir(n):
    for r, batch in BATCH_DIRS.items():
        if n in r:
            return CONSTRUCTIONS / batch / f"Sample_{n}_InstanceDiagram"
    raise ValueError(f"Sample {n} not in any batch")

def get_image(n):
    d = get_sample_dir(n)
    for ext in ("Construction.jpeg", "Construction.jpg"):
        p = d / ext
        if p.exists():
            return p
    raise FileNotFoundError(f"No construction image for Sample {n}")

# ── Extended PUML parser ───────────────────────────────────────────────────────
def parse_puml_full(n):
    """
    Parses InstanceDiagramSN.puml and returns a dict with four keys:

      components       {instance_id: abbr}
                       e.g. {"rosm_1": "rosm", "nu_1": "nu"}

      slots            {slot_id: {"component": str, "slot_type": str}}
                       e.g. {"rosm_1&{thjo}_1": {"component": "rosm_1",
                                                   "slot_type": "thjo"}}

      connections      list of dicts, one per Connection object:
                       {"slot_a": str, "comp_a": str, "slot_type_a": str,
                        "slot_b": str, "comp_b": str, "slot_type_b": str}

      connection_configs  list of lists — each inner list is a group of
                          connection-ID strings that fire together.

    PUML object format reference
    ----------------------------
    Component:        object "rosm_1 : Component"
    Slot:             object "rosm_1&{thjo}_1 : Slot"
                        component  = part before "&"   → rosm_1
                        slot_type  = text inside {}    → thjo
    Connection:       object "rosm_1&{thjo}_1#stwo7_1&{op}_1 : Connection"
                        two slot IDs separated by "#"
    ConnectionConfig: object "~CONN_A * CONN_B * ...~ : ConnectionConfig"
                        connections grouped by "~" delimiters, split on " * "
    """
    path = get_sample_dir(n) / f"InstanceDiagramS{n}.puml"
    text = path.read_text()

    # ── Components ────────────────────────────────────────────────────────────
    components = {}
    for m in re.finditer(r'object\s+"([^"]+?)\s*:\s*Component"', text):
        inst = m.group(1).strip()
        abbr = inst.rsplit("_", 1)[0]
        components[inst] = abbr

    # ── Slots ─────────────────────────────────────────────────────────────────
    # Slot ID format: "COMP_INST&{slot_type}_index"
    slots = {}
    for m in re.finditer(r'object\s+"([^"]+?)\s*:\s*Slot"', text):
        slot_id = m.group(1).strip()
        comp_part  = slot_id.split("&")[0]
        type_match = re.search(r'\{([^}]+)\}', slot_id)
        slot_type  = type_match.group(1) if type_match else "unknown"
        slots[slot_id] = {"component": comp_part, "slot_type": slot_type}

    # ── Connections ───────────────────────────────────────────────────────────
    # Connection ID format: "SLOT_A#SLOT_B"
    connections = []
    for m in re.finditer(r'object\s+"([^"#"]+?#[^"]+?)\s*:\s*Connection"', text):
        conn_id = m.group(1).strip()
        parts   = conn_id.split("#", 1)
        if len(parts) != 2:
            continue
        slot_a, slot_b = parts[0].strip(), parts[1].strip()
        comp_a = slot_a.split("&")[0]
        comp_b = slot_b.split("&")[0]
        ta     = re.search(r'\{([^}]+)\}', slot_a)
        tb     = re.search(r'\{([^}]+)\}', slot_b)
        connections.append({
            "slot_a":      slot_a,
            "comp_a":      comp_a,
            "slot_type_a": ta.group(1) if ta else "?",
            "slot_b":      slot_b,
            "comp_b":      comp_b,
            "slot_type_b": tb.group(1) if tb else "?",
        })

    # ── ConnectionConfigs ─────────────────────────────────────────────────────
    # Format: "~CONN_A * CONN_B * CONN_C~ : ConnectionConfig"
    configs = []
    for m in re.finditer(r'object\s+"~([^~]+?)~\s*:\s*ConnectionConfig"', text):
        group = [c.strip() for c in m.group(1).strip().split("*")]
        configs.append(group)

    return {
        "components":         components,
        "slots":              slots,
        "connections":        connections,
        "connection_configs": configs,
    }

def print_puml_summary(n, puml):
    """Human-readable console dump of the parsed PUML graph."""
    comps = puml["components"]
    conns = puml["connections"]
    cfgs  = puml["connection_configs"]

    print(f"  Components ({len(comps)}):")
    for inst, abbr in sorted(comps.items()):
        slots_owned = [sid for sid, s in puml["slots"].items()
                       if s["component"] == inst]
        slot_types  = [puml["slots"][s]["slot_type"] for s in slots_owned]
        print(f"    {inst:20s}  abbr={abbr:10s}  slots={slot_types}")

    print(f"  Connections ({len(conns)}):")
    for c in conns:
        print(f"    {c['comp_a']:20s} [{c['slot_type_a']:6s}]"
              f"  ──  {c['comp_b']:20s} [{c['slot_type_b']:6s}]")

    if cfgs:
        print(f"  ConnectionConfigs ({len(cfgs)}):")
        for cfg in cfgs:
            print(f"    group of {len(cfg)}: {cfg}")

def load_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def filter_masks(masks, image_area):
    return [
        m for m in masks
        if m["area"] > 300
        and m["area"] < image_area * 0.5
    ]

def random_color():
    return [int(x) for x in np.random.randint(60, 230, 3)]

def visualize(image, masks, puml, sample_n, save_path):
    comps = puml["components"]
    conns = puml["connections"]
    expected = len(comps)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    axes[0].imshow(image)
    axes[0].set_title(f"Sample {sample_n} — Original", fontsize=11)
    axes[0].axis("off")

    overlay = image.copy()
    colors  = {}
    for i, m in enumerate(masks):
        color     = random_color()
        colors[i] = color
        seg       = m["segmentation"]
        overlay[seg] = (overlay[seg] * 0.45 + np.array(color) * 0.55).astype(np.uint8)

    axes[1].imshow(overlay)
    for i, m in enumerate(masks):
        x, y, bw, bh = m["bbox"]
        rect = patches.Rectangle(
            (x, y), bw, bh,
            linewidth=1.2,
            edgecolor=(colors[i][0]/255, colors[i][1]/255, colors[i][2]/255),
            facecolor="none"
        )
        axes[1].add_patch(rect)
        axes[1].text(x + 2, y - 4, str(i),
                     color="white", fontsize=7,
                     bbox=dict(facecolor="black", alpha=0.5, pad=1))

    # Build annotation text: components + connections
    lines = ["Components:"]
    for inst in sorted(comps):
        lines.append(f"  {inst} ({comps[inst]})")
    lines.append(f"\nConnections ({len(conns)}):")
    for c in conns:
        lines.append(f"  {c['comp_a']}[{c['slot_type_a']}] ── {c['comp_b']}[{c['slot_type_b']}]")
    lines.append(f"\nMasks found:  {len(masks)}")
    lines.append(f"Expected:     {expected}")

    axes[1].set_title(f"Sample {sample_n} — SAM masks ({len(masks)} found, {expected} expected)", fontsize=11)
    axes[1].axis("off")
    fig.text(0.51, 0.01, "\n".join(lines), ha="left", va="bottom",
             fontsize=7.5, family="monospace",
             bbox=dict(facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close()

# ── Load SAM ──────────────────────────────────────────────────────────────────
print(f"Loading SAM ({MODEL_TYPE}) from {CHECKPOINT.name} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT))
sam.to(device=device)

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
SAMPLES = list(range(41, 45))
summary = []

for n in SAMPLES:
    print(f"── Sample {n} ──────────────────────────────────────")
    try:
        img_path = get_image(n)
        image    = load_rgb(img_path)
        H, W     = image.shape[:2]

        puml     = parse_puml_full(n)
        expected = len(puml["components"])

        print(f"  Image:  {img_path.name}  ({W}×{H}px)")
        print_puml_summary(n, puml)

        masks_raw = mask_generator.generate(image)
        masks     = filter_masks(masks_raw, H * W)

        ious  = [round(m["predicted_iou"], 3)   for m in masks]
        areas = [m["area"]                       for m in masks]

        match  = len(masks) == expected
        status = "OK" if match else ("SHORT" if len(masks) < expected else "OVER")

        print(f"  Masks:  {len(masks_raw)} raw → {len(masks)} after filter")
        print(f"  Status: {status}  (found {len(masks)}, expected {expected})")
        if ious:
            print(f"  IoU:    min={min(ious):.3f}  max={max(ious):.3f}  avg={sum(ious)/len(ious):.3f}")

        vis_path = VIS_DIR / f"sample_{n:03d}_conn.png"
        visualize(image, masks, puml, n, vis_path)
        print(f"  Saved:  {vis_path.name}")

        summary.append({
            "sample":              n,
            "image":               str(img_path),
            "image_size":          [W, H],
            # full PUML graph
            "puml_components":     puml["components"],
            "puml_slots":          puml["slots"],
            "puml_connections":    puml["connections"],
            "puml_configs":        puml["connection_configs"],
            # counts
            "expected_components": expected,
            "slot_count":          len(puml["slots"]),
            "connection_count":    len(puml["connections"]),
            # SAM results
            "masks_raw":           len(masks_raw),
            "masks_filtered":      len(masks),
            "status":              status,
            "iou_min":             min(ious) if ious else None,
            "iou_max":             max(ious) if ious else None,
            "iou_avg":             round(sum(ious)/len(ious), 3) if ious else None,
            "area_min":            min(areas) if areas else None,
            "area_max":            max(areas) if areas else None,
            "visualization":       str(vis_path),
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        summary.append({"sample": n, "error": str(e)})

    print()

# ── Save summary ──────────────────────────────────────────────────────────────
summary_path = OUTPUT_DIR / "sam_trial_connections_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary written to {summary_path}")

ok    = sum(1 for s in summary if s.get("status") == "OK")
short = sum(1 for s in summary if s.get("status") == "SHORT")
over  = sum(1 for s in summary if s.get("status") == "OVER")
err   = sum(1 for s in summary if "error" in s)
print(f"\nResults: {ok} OK  |  {short} SHORT  |  {over} OVER  |  {err} ERROR")

# ── Print slot-type inventory across all samples ───────────────────────────────
print("\n── Slot type inventory ─────────────────────────────────────────────")
slot_type_counts = defaultdict(int)
comp_slot_map    = defaultdict(set)   # abbr → set of slot types it carries

for entry in summary:
    if "error" in entry:
        continue
    for slot_id, slot_data in entry["puml_slots"].items():
        st   = slot_data["slot_type"]
        abbr = entry["puml_components"].get(slot_data["component"], "?")
        slot_type_counts[st] += 1
        comp_slot_map[abbr].add(st)

print("Slot types seen (across all samples in this run):")
for st, cnt in sorted(slot_type_counts.items(), key=lambda x: -x[1]):
    print(f"  {st:10s}  {cnt:3d} occurrences")

print("\nComponent abbr → slot types carried:")
for abbr, types in sorted(comp_slot_map.items()):
    print(f"  {abbr:15s}  {sorted(types)}")
