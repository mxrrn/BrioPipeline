"""
Visualize the 3D instance proposals for a sample using matplotlib.
Saves a static 3D scatter plot per sample to outputs/sample_N/viz_3d.png.

Usage:
    python visualize.py --sample 113
"""
import sys
import pickle
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


def _resolve_run_dir(run_arg: str | None) -> Path:
    """Return the run directory to visualize.

    If run_arg is given, look it up directly under OUTPUTS_ROOT.
    Otherwise pick the most recently created run_* folder.
    """
    if run_arg:
        d = cfg.OUTPUTS_ROOT / run_arg
        if not d.exists():
            raise FileNotFoundError(f"Run directory not found: {d}")
        return d
    candidates = sorted(
        [d for d in cfg.OUTPUTS_ROOT.iterdir()
         if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found in {cfg.OUTPUTS_ROOT}")
    return candidates[-1]

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
]


def visualize_sample(sample_id: int, run_dir: Path):
    out_dir       = run_dir / f"sample_{sample_id}"
    results_path  = out_dir / "results.json"
    proposals_dir = out_dir / "proposals"

    if not results_path.exists():
        print(f"No results.json for sample {sample_id}. Run pipeline.py first.")
        return

    with open(results_path) as f:
        summary = json.load(f)

    n = summary["n_components"]
    # Support both proposals_N{n}.pkl and proposals_N{n}_v*.pkl
    candidates = sorted(proposals_dir.glob(f"proposals_N{n}*.pkl"))
    if not candidates:
        print(f"No proposals cache found for N={n} in {proposals_dir}.")
        return
    proposals_file = candidates[-1]

    with open(proposals_file, "rb") as f:
        raw = pickle.load(f)
    # v4+ cache returns (clouds, visual_cls) tuple; earlier versions return list directly
    proposals = raw[0] if isinstance(raw, tuple) else raw

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection="3d")

    for i, (pts, inst) in enumerate(zip(proposals, summary["instances"])):
        if len(pts) == 0:
            continue
        color = COLORS[i % len(COLORS)]
        # Subsample for speed
        idx = np.random.choice(len(pts), min(500, len(pts)), replace=False)
        pts_s = pts[idx]
        ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
                   c=color, s=1, alpha=0.6, label=inst["instance_id"])

    ax.set_title(f"Sample {sample_id} — 3D instance proposals", fontsize=13)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=8, markerscale=5)

    out_path = out_dir / "viz_3d.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Viz] Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=113)
    parser.add_argument("--run",    type=str, default=None,
                        help="Run folder name (e.g. run_001_20260526_2123). "
                             "Defaults to the most recent run_* folder.")
    args = parser.parse_args()
    run_dir = _resolve_run_dir(args.run)
    print(f"[Viz3D] Using run: {run_dir.name}")
    visualize_sample(args.sample, run_dir)
