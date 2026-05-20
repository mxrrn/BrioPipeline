"""
BRIO Fast Inference Pipeline — end-to-end entry point.

Given a set of multi-view images of an unseen BRIO construction, produces:
  - A list of identified component instances with 3D positions
  - Inferred slot connections between instances
  - A PlantUML instance diagram

Runtime: < 2 seconds on GPU (after one-time model load).

Usage
─────
    # With PUML to guide instance count:
    python infer.py --sample 120

    # With explicit image directory and component count:
    python infer.py --images /path/to/images --n-components 5

    # Specify trained weights explicitly:
    python infer.py --sample 120 --weights runs/detect/train/weights/best.pt
"""
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from logger import setup_logging


def collect_images(sample_id: int) -> list[Path]:
    """Collect the same 14-image composite used by the slow pipeline."""
    base  = Path("/mnt/c/BA/02-resources/data/multi_view_images") / f"Sample_{sample_id}"
    imgs  = []
    ring90 = sorted((base / "Images90").glob("*.jpg")) + \
             sorted((base / "Images90").glob("*.JPG"))
    imgs.extend(ring90)
    ring45 = sorted((base / "Images45").glob("*.jpg")) + \
             sorted((base / "Images45").glob("*.JPG"))
    imgs.extend(ring45[::4])
    return imgs


def run(image_paths: list[Path],
        weights: Path,
        n_per_class: dict[str, int] | None,
        output_dir: Path,
        title: str = "BRIO Inference") -> dict:
    """
    Full inference pipeline.

    Args:
        image_paths : ordered list of view images
        weights     : path to trained YOLOv8 .pt file
        n_per_class : {cls: count} from PUML; None = auto-estimate
        output_dir  : where to write results and PUML
        title       : diagram title

    Returns summary dict.
    """
    from calibrator    import load_rig
    from detector      import Detector
    from triangulator  import triangulate
    from connector     import infer_connections, connection_summary
    from puml_generator import generate_puml, save_puml

    t0 = time.perf_counter()

    # ── Load rig ─────────────────────────────────────────────────────────
    rig = load_rig()
    print(f"[Infer] Rig: {rig['n_cameras']} cameras  "
          f"(calibrated from samples {rig['source_samples']})")

    # ── YOLO detection ────────────────────────────────────────────────────
    det = Detector(weights)
    print(f"[Infer] Running YOLO on {len(image_paths)} images...")
    t_det = time.perf_counter()
    raw_dets = det.predict_batch(image_paths)
    print(f"[Infer] Detection: {time.perf_counter() - t_det:.2f}s")

    # ── Triangulation ─────────────────────────────────────────────────────
    print("[Infer] Triangulating 3D positions...")
    t_tri = time.perf_counter()
    instances = triangulate(raw_dets, rig, n_per_class=n_per_class)
    print(f"[Infer] Triangulation: {time.perf_counter() - t_tri:.2f}s  "
          f"→ {len(instances)} instances")

    # ── Connection inference ──────────────────────────────────────────────
    edges = infer_connections(instances)
    print(f"[Infer] Connections inferred: {len(edges)}")

    # ── PUML generation ───────────────────────────────────────────────────
    puml_text = generate_puml(instances, edges, title=title)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_puml(puml_text, output_dir / "predicted.puml")

    # ── JSON summary ──────────────────────────────────────────────────────
    summary = {
        "title":       title,
        "n_instances": len(instances),
        "instances": [
            {
                "instance_id": d.instance_id,
                "cls":         d.cls,
                "centroid":    d.centroid.tolist(),
                "conf":        round(d.conf, 3),
                "n_views":     d.n_views,
            }
            for d in instances
        ],
        "connections": [{"a": a, "b": b} for a, b in edges],
        "elapsed_s":  round(time.perf_counter() - t0, 2),
    }
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2))

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  {'Instance':<20} {'3D Position':>32}  Conf  Views")
    for d in instances:
        c = d.centroid
        print(f"  {d.instance_id:<20} "
              f"({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})  "
              f"{d.conf:.2f}  {d.n_views}")
    print(f"\n  Connections:")
    print(connection_summary(instances, edges))
    print(f"\n  Elapsed: {summary['elapsed_s']}s")
    print(f"  PUML → {output_dir / 'predicted.puml'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="BRIO fast inference")
    parser.add_argument("--sample",   type=int,
                        help="Sample ID (loads images from standard data path)")
    parser.add_argument("--images",   type=Path,
                        help="Directory containing view images (alternative to --sample)")
    parser.add_argument("--weights",  type=Path,
                        help="YOLOv8 weights .pt file")
    parser.add_argument("--puml",     type=Path,
                        help="Ground-truth PUML (used to set per-class count)")
    parser.add_argument("--n-components", type=int,
                        help="Total expected components (used if no PUML given)")
    parser.add_argument("--output",   type=Path,
                        help="Output directory (default: fast_pipeline/outputs/sample_N)")
    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────
    label = f"infer_sample{args.sample}" if args.sample else "infer"
    setup_logging(label)

    # ── Resolve image paths ───────────────────────────────────────────────
    if args.sample:
        image_paths = collect_images(args.sample)
        output_dir  = args.output or (cfg.FAST_PIPELINE / "outputs" / f"sample_{args.sample}")
        title = f"Sample {args.sample}"
    elif args.images:
        image_paths = sorted(args.images.glob("*.jpg")) + \
                      sorted(args.images.glob("*.JPG"))
        output_dir  = args.output or (cfg.FAST_PIPELINE / "outputs" / args.images.name)
        title = args.images.name
    else:
        parser.error("Provide --sample N or --images /path")

    if not image_paths:
        parser.error("No images found.")

    # ── Resolve weights ───────────────────────────────────────────────────
    if args.weights:
        weights = args.weights
    else:
        # Auto-find latest training run
        runs = sorted((cfg.RUNS_DIR / "detect").glob("*/weights/best.pt"))
        if not runs:
            parser.error(
                "No trained weights found. Run train.py first, or pass --weights."
            )
        weights = runs[-1]
        print(f"[Infer] Auto-selected weights: {weights}")

    # ── Resolve per-class count ───────────────────────────────────────────
    n_per_class = None
    if args.puml:
        sys.path.insert(0, str(cfg.SLOW_PIPELINE))
        from puml_parser import parse_puml
        manifest = parse_puml(args.puml)
        from collections import Counter
        counts = Counter(c.cls for c in manifest.components)
        n_per_class = dict(counts)
        print(f"[Infer] PUML manifest: {dict(n_per_class)}")
    elif args.n_components:
        print(f"[Infer] Expected {args.n_components} components total (class split unknown)")

    run(image_paths, weights, n_per_class, output_dir, title=title)


if __name__ == "__main__":
    main()
