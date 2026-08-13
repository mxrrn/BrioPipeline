"""
Batch-run dataset_builder over many samples, ordered for class coverage.

Selection: greedy set cover over the 29 classes (each round picks the sample
adding the most not-yet-covered classes), then remaining samples ordered by
(rare-class content, fewest parse warnings). Samples with no trusted seeds or
no verified image match are skipped automatically by build_sample.

Run (GPU, ~8-15 min per uncached sample):
    python batch_build.py --limit 40
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
PARSED = HERE / "parsed_mappings"


def rank_samples() -> list[int]:
    per_sample: dict[int, set[str]] = {}
    warn_count: dict[int, int] = {}
    for f in PARSED.glob("sample_*.json"):
        d = json.loads(f.read_text())
        sid = d["sample_id"]
        img = (d.get("image") or {})
        diff = img.get("diff")
        if not img.get("match") or diff is None or diff > 1.0:
            continue
        cls = {p["cls"] for p in d["points"] if p["kind"] == "instance"}
        if not cls:
            continue
        per_sample[sid] = cls
        warn_count[sid] = len(d["warnings"])

    freq = Counter()
    for cls in per_sample.values():
        freq.update(cls)

    ordered: list[int] = []
    covered: set[str] = set()
    remaining = dict(per_sample)
    # Greedy cover: prefer samples adding most new classes, tie-break on
    # fewer warnings (cleaner annotations first).
    while remaining:
        best = max(remaining,
                   key=lambda s: (len(remaining[s] - covered),
                                  -warn_count[s],
                                  # rare-class bonus
                                  sum(1.0 / freq[c] for c in remaining[s])))
        if not (remaining[best] - covered) and len(ordered) >= len(per_sample):
            break
        ordered.append(best)
        covered |= remaining.pop(best)
        if not remaining:
            break
        if all(not (v - covered) for v in remaining.values()):
            # cover complete — order the rest by rare-class content
            rest = sorted(remaining,
                          key=lambda s: (-sum(1.0 / freq[c] for c in remaining[s]),
                                         warn_count[s]))
            ordered.extend(rest)
            break
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", type=Path, default=HERE / "dataset_crops")
    args = ap.parse_args()

    from dataset_builder import build_sample
    order = rank_samples()
    print(f"[batch] {len(order)} candidate samples; running first {args.limit}")
    print(f"[batch] order: {order[:args.limit]}")
    done = 0
    for sid in order[:args.limit]:
        t0 = time.time()
        try:
            build_sample(sid, args.out, render=False)
        except Exception as e:
            print(f"[{sid}] FAILED: {type(e).__name__}: {e}")
        else:
            done += 1
        print(f"[batch] sample {sid} took {time.time()-t0:.0f}s "
              f"({done} ok so far)", flush=True)


if __name__ == "__main__":
    main()
