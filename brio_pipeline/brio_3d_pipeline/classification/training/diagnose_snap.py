"""
Tier-1 diagnostic for dataset_builder.py's "snap found no foreground"
failures on ti/whre/rolo (see .claude plan "Ensure all 29 classes are
included in one-vs-all training" and ONE_VS_ALL_PLAN.md).

Purely 2D, no DUSt3R, no GPU — snap_to_foreground() only ever touches the
single annotated photo + its foreground mask + colour prototypes, so this
replays that exact search with full instrumentation: every solid point's
colour distance is recorded, not just the running best (which is all
snap_to_foreground() itself keeps). This distinguishes two very different
failure modes that look identical from the outside (both just report
"snap found no foreground"):

  (a) foreground never found at all along the search cone — points at a
      direction/max_dist/foreground-mask problem, not a colour problem.
  (b) foreground found, but its colour never gets close enough to the
      class prototype — points at a domain-mismatch (isolated studio
      photos vs. real construction lighting) or wrong-part-adjacency
      problem (search cone crossing onto a different, adjacent component).

For (b), this also reports which OTHER class's prototype the found pixels
are actually closest to — directly tests the adjacency-confusion hypothesis
(e.g. a `ti` search cone landing on `whre`-coloured pixels, since tires
mount directly onto wheelred rims).

Does not modify dataset_builder.py or snap_to_foreground() — this is a
read-only diagnostic pass. The fix, informed by this output, is a separate
step reviewed with the user first.

Usage:
    python diagnose_snap.py                          # known failing seeds
    python diagnose_snap.py --samples 62 73 143 150   # specific samples
    python diagnose_snap.py --classes ti whre rolo    # filter by class
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

from classification.training.dataset_builder import (  # noqa: E402
    load_color_prototypes, _color_metric, COLOR_ACCEPT, COLOR_FALLBK,
)

PARSED = HERE / "parsed_mappings"

# (sample_id list) known to have attempted-and-failed instances of each
# class, from the Tier-1 investigation — the default target set.
KNOWN_FAILURES = {
    "ti":   [62, 73, 143, 150],
    "whre": [62, 73, 143],
    "rolo": [62, 135],
}


def _instrumented_search(img: np.ndarray, fg: np.ndarray, x: float, y: float,
                         d: tuple[float, float] | None, max_dist: float,
                         proto: np.ndarray | None,
                         all_protos: dict[str, np.ndarray]) -> dict:
    """Same marching cone search as snap_to_foreground(), instrumented to
    record every solid point's colour distance instead of only the best."""
    H, W = fg.shape

    def solid(px: float, py: float) -> bool:
        xi, yi = int(round(px)), int(round(py))
        if not (2 <= xi < W - 2 and 2 <= yi < H - 2):
            return False
        return fg[yi - 2:yi + 3, xi - 2:xi + 3].mean() > 0.6

    def patch_median(px: float, py: float) -> np.ndarray | None:
        xi, yi = int(round(px)), int(round(py))
        patch = img[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4]
        pfg = fg[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4]
        if not pfg.any():
            return None
        return np.median(patch[pfg].astype(np.float32), axis=0)

    result: dict = {
        "found_solid_at_all": False,
        "n_solid_points": 0,
        "best_color_dist": None,
        "best_point": None,
        "best_patch_color_bgr": None,
        "closest_class_at_best": None,
        "closest_class_dist_at_best": None,
        "would_accept": False,
        "would_fallback_accept": False,
        "skip_reason": None,
    }
    if d is None or (not d[0] and not d[1]):
        result["skip_reason"] = "no arrow direction (d is None) — the seed has no dfx/dfy"
        return result

    base = float(np.arctan2(d[1], d[0]))
    angles = [base + np.deg2rad(a) for a in
              (0, 4, -4, 8, -8, 12, -12, 16, -16, 20, -20,
               24, -24, 28, -28, 32, -32, 36, -36)]

    best = None  # (color_dist, x, y, t, patch_median_bgr)
    t = 0.0
    while t <= max_dist:
        for ang in angles:
            ux, uy = np.cos(ang), np.sin(ang)
            px, py = x + t * ux, y + t * uy
            if not solid(px, py):
                continue
            result["n_solid_points"] += 1
            result["found_solid_at_all"] = True
            med = patch_median(px, py)
            if med is None:
                continue
            cd = _color_metric(med, proto) if proto is not None else 0.0
            if best is None or cd < best[0]:
                best = (cd, px, py, t, med)
        t += 2.0

    if best is not None:
        cd, px, py, t, med = best
        result["best_color_dist"] = float(cd)
        result["best_point"] = (float(px), float(py))
        result["best_patch_color_bgr"] = [float(v) for v in med]
        result["would_accept"] = bool(cd < COLOR_ACCEPT)
        result["would_fallback_accept"] = bool(cd < COLOR_FALLBK)

        closest_c, closest_d = None, float("inf")
        for cls, p in all_protos.items():
            dd = _color_metric(med, p)
            if dd < closest_d:
                closest_c, closest_d = cls, dd
        result["closest_class_at_best"] = closest_c
        result["closest_class_dist_at_best"] = float(closest_d)

    return result


def visualize_seed(ann_full: np.ndarray, fg_full: np.ndarray,
                   x: float, y: float, d: tuple[float, float] | None,
                   max_dist: float, sample_id: int, base: str,
                   out_dir: Path) -> None:
    """Overlay the foreground mask, seed point, arrow direction, and search
    cone extent on the annotated photo — a purely-2D, no-GPU stand-in for
    --render (which can't show this: the failure happens before build_sample()
    ever reaches its render loop, see diagnose_snap.py's module docstring)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = ann_full.copy()
    tint = np.zeros_like(overlay)
    tint[fg_full] = (0, 80, 0)   # faint green = foreground mask
    overlay = cv2.addWeighted(overlay, 1.0, tint, 0.5, 0)

    cv2.drawMarker(overlay, (int(x), int(y)), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    if d is not None and (d[0] or d[1]):
        base_ang = float(np.arctan2(d[1], d[0]))
        for a in (-36, -20, 0, 20, 36):
            ang = base_ang + np.deg2rad(a)
            ex = int(x + max_dist * np.cos(ang))
            ey = int(y + max_dist * np.sin(ang))
            color = (0, 255, 255) if a == 0 else (0, 140, 255)
            cv2.line(overlay, (int(x), int(y)), (ex, ey), color, 2 if a == 0 else 1)
    else:
        cv2.putText(overlay, "NO DIRECTION", (int(x) + 10, int(y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    out_path = out_dir / f"s{sample_id}_{base}.jpg"
    cv2.imwrite(str(out_path), overlay)
    print(f"    → {out_path}")


def diagnose_sample(sample_id: int, target_classes: set[str] | None,
                    all_protos: dict[str, np.ndarray],
                    visualize_dir: Path | None = None,
                    max_dist_override: float | None = None) -> list[dict]:
    parsed_file = PARSED / f"sample_{sample_id}.json"
    if not parsed_file.exists():
        print(f"[{sample_id}] no parsed mapping — skipped")
        return []
    parsed = json.loads(parsed_file.read_text())
    seeds = [p for p in parsed["points"] if p["kind"] == "instance"]
    match = (parsed.get("image") or {}).get("match")
    if not match:
        print(f"[{sample_id}] no verified image match — skipped "
              f"(structurally unusable regardless of any snap fix)")
        return []
    if target_classes is not None:
        seeds = [s for s in seeds if s["cls"] in target_classes]
    if not seeds:
        return []

    annotated = Path(match)
    ann_full = cv2.imread(str(annotated))
    if ann_full is None:
        print(f"[{sample_id}] couldn't load annotated photo {annotated} — skipped")
        return []
    Hf, Wf = ann_full.shape[:2]
    fg_raw = np.any(ann_full < 245, axis=2).astype(np.uint8)
    fg_full = cv2.morphologyEx(
        fg_raw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(bool)

    rows = []
    for s in seeds:
        x_full, y_full = s["fx"] * Wf, s["fy"] * Hf
        d = (s["dfx"] * Wf, s["dfy"] * Hf) if s.get("dfx") is not None else None
        max_dist = 400.0 if s.get("coord_src") == "abs" else 250.0
        if max_dist_override is not None:
            max_dist = max_dist_override
        proto = all_protos.get(s["cls"])
        r = _instrumented_search(ann_full, fg_full, x_full, y_full, d,
                                 max_dist, proto, all_protos)
        r["sample_id"] = sample_id
        r["base"] = s["base"]
        r["cls"] = s["cls"]
        r["has_prototype"] = proto is not None
        rows.append(r)
        if visualize_dir is not None:
            visualize_seed(ann_full, fg_full, x_full, y_full, d, max_dist,
                          sample_id, s["base"], visualize_dir)
    return rows


def _print_report(rows: list[dict]) -> None:
    print(f"\n{'sample':>6} {'base':<10} {'cls':<7} proto  solid  n_pts  "
          f"best_dist  accept  fallback  closest_class(dist)")
    for r in rows:
        proto_flag = "y" if r["has_prototype"] else "N"
        if r["skip_reason"]:
            print(f"{r['sample_id']:>6} {r['base']:<10} {r['cls']:<7} "
                  f"{proto_flag:<6} -- {r['skip_reason']}")
            continue
        solid = "y" if r["found_solid_at_all"] else "N"
        best = f"{r['best_color_dist']:.1f}" if r["best_color_dist"] is not None else "--"
        acc = "Y" if r["would_accept"] else "n"
        fb = "Y" if r["would_fallback_accept"] else "n"
        closest = (f"{r['closest_class_at_best']}({r['closest_class_dist_at_best']:.1f})"
                  if r["closest_class_at_best"] else "--")
        print(f"{r['sample_id']:>6} {r['base']:<10} {r['cls']:<7} "
              f"{proto_flag:<6} {solid:<6} {r['n_solid_points']:<6} "
              f"{best:<10} {acc:<7} {fb:<9} {closest}")

    print("\nSummary:")
    by_cls: dict[str, list[dict]] = {}
    for r in rows:
        by_cls.setdefault(r["cls"], []).append(r)
    for cls, rs in sorted(by_cls.items()):
        n = len(rs)
        n_no_solid = sum(1 for r in rs if not r["found_solid_at_all"])
        n_solid_no_match = sum(
            1 for r in rs if r["found_solid_at_all"] and not r["would_fallback_accept"])
        n_would_pass = sum(1 for r in rs if r["would_accept"] or r["would_fallback_accept"])
        print(f"  {cls}: {n} seeds — {n_no_solid} never found foreground, "
              f"{n_solid_no_match} found foreground but colour rejected it, "
              f"{n_would_pass} would actually pass today (so the live failure "
              f"must be from something else, e.g. max_dist/direction differences "
              f"at run time)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, nargs="*", default=None,
                    help="Sample IDs to check (default: KNOWN_FAILURES' samples)")
    ap.add_argument("--classes", type=str, nargs="*", default=None,
                    help="Class codes to filter to (default: ti, whre, rolo)")
    ap.add_argument("--visualize", type=Path, default=None,
                    help="Save a foreground-mask + seed + direction-cone "
                         "overlay per seed to this directory")
    ap.add_argument("--max-dist-override", type=float, default=None,
                    help="Test with a longer search distance than the "
                         "production 250/400px, to check whether max_dist "
                         "is the limiting factor")
    args = ap.parse_args()

    classes = set(args.classes) if args.classes else set(KNOWN_FAILURES.keys())
    if args.samples:
        samples = args.samples
    else:
        samples = sorted({sid for cls in classes for sid in KNOWN_FAILURES.get(cls, [])})
        if not samples:
            samples = sorted({sid for sids in KNOWN_FAILURES.values() for sid in sids})

    all_protos = load_color_prototypes()
    print(f"[Diagnose] {len(samples)} samples, classes={sorted(classes)}, "
          f"{len(all_protos)} prototypes loaded")

    all_rows = []
    for sid in samples:
        all_rows.extend(diagnose_sample(sid, classes, all_protos, args.visualize,
                                        max_dist_override=args.max_dist_override))

    if not all_rows:
        print("No matching seeds found in the given samples/classes.")
        return
    _print_report(all_rows)


if __name__ == "__main__":
    main()
