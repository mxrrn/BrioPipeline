"""Ad-hoc evaluation: run_017 (pre-fix) vs run_018 (mask filtering + conf filtering)."""
import json
import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
OLD = ROOT / "run_017_20260612_1400/sample_113"
NEW = ROOT / "run_018_20260612_1431/sample_113"


def stats(run_dir, label):
    print(f"\n=== {label}: {run_dir.parent.name} ===")
    res = json.loads((run_dir / "results.json").read_text())
    cents = []
    for inst in res["instances"]:
        c = np.array(inst["centroid"]); b = np.array(inst["bbox_size"])
        cents.append(c)
        elong = b.max() / max(b[b.argsort()[1]], 1e-6)  # max/mid extent ratio
        print(f"  {inst['instance_id']:<12} {inst['n_points']:>7d} pts  "
              f"bbox=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f})  elong={elong:.1f}")
    cents = np.array(cents)
    # pairwise centroid separation — collapsed clusters have near-zero separation
    d = np.linalg.norm(cents[:, None] - cents[None, :], axis=-1)
    iu = np.triu_indices(len(cents), 1)
    print(f"  centroid separation: min={d[iu].min():.4f}  mean={d[iu].mean():.4f}  max={d[iu].max():.4f}")

    # mask stats
    sam_dir = run_dir / "sam"
    pkls = sorted(sam_dir.glob("sam_masks_top*_filt.pkl")) or sorted(sam_dir.glob("sam_masks_top*.pkl"))
    with open(pkls[0], "rb") as f:
        masks = pickle.load(f)
    per_view = [len(m) for m in masks]
    areas = [m["area"] for view in masks for m in view]
    print(f"  masks/view: min={min(per_view)} mean={np.mean(per_view):.1f} max={max(per_view)}  "
          f"mask area: median={int(np.median(areas))} max={max(areas)} px")


stats(OLD, "OLD (top-N by IoU, no conf filter)")
stats(NEW, "NEW (fg-filtered masks + conf>=2.0)")
