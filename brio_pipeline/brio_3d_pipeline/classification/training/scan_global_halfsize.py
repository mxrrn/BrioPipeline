"""
One-off scan: compute the dataset-wide fixed crop half-size for classifier
training data.

Why: preprocessor.compute_global_halfsize() derives the crop scale from the
largest construction *in the batch being run* — run_029 (113 alone) got 223,
run_033 (113+114) got 297. The "1 px = same physical length" guarantee
therefore only holds within one batch. A classifier that uses absolute pixel
scale as a size cue needs ONE fixed half-size across every sample it will
ever train or infer on. This scan computes max LCC half-size over the
pipeline's selected views of every usable sample, applies the same 20%
padding, and prints the value to pin in the dataset builder.
"""
import sys
from pathlib import Path

PIPE = Path("/mnt/c/BA/00-project/brio_pipeline/brio_3d_pipeline")
sys.path.insert(0, str(PIPE))

import numpy as np
from detection.preprocessor import _sample_lcc_halfsize  # same LCC logic as deployment
import config as cfg


def collect_images(sample_id: int) -> list[Path]:
    """Mirror pipeline.collect_images (8 views/ring, stride over sorted glob)."""
    base   = cfg.MULTI_VIEW / f"Sample_{sample_id}"
    target = getattr(cfg, "IMAGE_VIEWS_PER_RING", 8)
    imgs   = []
    for elev in ["Images30", "Images45", "Images60", "Images90"]:
        folder = base / elev
        if not folder.exists():
            continue
        ring   = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.JPG"))
        stride = max(1, len(ring) // target)
        imgs.extend(ring[::stride])
    return imgs


def main():
    samples = [int(s) for s in sys.argv[1:]] or list(range(1, 151))
    results = {}
    for sid in samples:
        imgs = collect_images(sid)
        if not imgs:
            print(f"[{sid}] no images", flush=True)
            continue
        h = _sample_lcc_halfsize(imgs)
        results[sid] = h
        print(f"[{sid}] max LCC half = {h:.1f} px over {len(imgs)} views", flush=True)

    if results:
        worst = max(results, key=results.get)
        raw   = results[worst]
        padded = int(np.ceil(raw * (1 + cfg.AUTO_CROP_PADDING)))
        print(f"\nGLOBAL: raw max {raw:.1f} px (sample {worst}) "
              f"+ {int(cfg.AUTO_CROP_PADDING*100)}% padding → FIXED_HALF = {padded}")
        top = sorted(results.items(), key=lambda kv: -kv[1])[:10]
        print("Top-10 largest:", ", ".join(f"{s}:{v:.0f}" for s, v in top))


if __name__ == "__main__":
    main()
