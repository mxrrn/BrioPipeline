#!/usr/bin/env python3
"""
Generate fixed-scale crops for a sample range without running SAM/DUSt3R —
reuses pipeline.py's own collect_images / compute_global_halfsize / crop_images
so the crops are pixel-identical to what the full pipeline would produce.

Usage:
    python crop_for_annotation.py --samples 1-50 --out /mnt/c/BA/15-annotations
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from pipeline import collect_images
from detection.preprocessor import compute_global_halfsize, crop_images


def parse_range(spec: str) -> list[int]:
    ids = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="e.g. 1-50 or 1,2,5-10")
    ap.add_argument("--out", required=True, help="output root (one subfolder per sample)")
    args = ap.parse_args()

    sample_ids = parse_range(args.samples)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[Crop] Collecting images for {len(sample_ids)} samples...")
    per_sample_paths = {}
    for sid in sample_ids:
        try:
            per_sample_paths[sid] = collect_images(sid)
        except FileNotFoundError as exc:
            print(f"[Crop] Skipping sample {sid}: {exc}")

    print("[Crop] Computing global crop scale across selected samples...")
    fixed_half = compute_global_halfsize(list(per_sample_paths.values()))

    for sid, img_paths in per_sample_paths.items():
        crop_dir = out_root / f"sample_{sid}"
        crop_images(img_paths, crop_dir, fixed_half=fixed_half)
        print(f"[Crop] sample_{sid}: {len(img_paths)} views -> {crop_dir}")

    print(f"[Crop] Done. {len(per_sample_paths)} samples written to {out_root}")


if __name__ == "__main__":
    main()
