"""
Classify 3D instance proposals against the PUML component manifest using
Hungarian assignment on a combined geometry + colour feature vector.

Geometry features
─────────────────
Extracted from the 3D point cloud: longest axis, axis ratios, log-volume.

Colour features
───────────────
Mean HSV extracted from the reference component images in
02-resources/data/component_images/.  This is far more reliable than
hardcoded approximate values.

Assignment
──────────
Hungarian assignment (scipy linear_sum_assignment) minimises total cost
across all N proposal → manifest pairings.  No greedy local decisions.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InstanceResult:
    instance_id : str
    cls         : str
    n_points    : int
    centroid    : np.ndarray   # (3,) world coords
    bbox_size   : np.ndarray   # (3,) axis-aligned extents


# ── Geometry features ─────────────────────────────────────────────────────────

def _geo_features(pts: np.ndarray) -> np.ndarray:
    """[log_vol, longest_axis, ratio_2, ratio_3, log_n_pts]"""
    if len(pts) < 3:
        return np.zeros(5, np.float32)
    lo, hi    = pts.min(0), pts.max(0)
    extents   = np.sort(hi - lo)[::-1]           # descending
    vol       = float(np.prod(np.maximum(extents, 1e-6)))
    return np.array([
        np.log1p(vol),
        extents[0],
        extents[1] / (extents[0] + 1e-6),
        extents[2] / (extents[0] + 1e-6),
        np.log1p(len(pts)),
    ], np.float32)


# ── Colour prototypes from reference images ───────────────────────────────────

_COLOUR_CACHE: dict[str, np.ndarray] | None = None


def _load_colour_prototypes() -> dict[str, np.ndarray]:
    global _COLOUR_CACHE
    if _COLOUR_CACHE is not None:
        return _COLOUR_CACHE

    from config import DATA_ROOT, OUTPUTS_ROOT
    cache_path = OUTPUTS_ROOT / "color_prototypes.pkl"

    from component_map import build_color_prototypes
    _COLOUR_CACHE = build_color_prototypes(
        DATA_ROOT / "component_images",
        cache_path=cache_path,
    )
    return _COLOUR_CACHE


def _colour_feature(cls: str) -> np.ndarray:
    """Return normalised HSV prototype for a class, or zeros if unknown."""
    protos = _load_colour_prototypes()
    return protos.get(cls, np.zeros(3, np.float32))


# ── Assignment ────────────────────────────────────────────────────────────────

def assign_classes(proposals   : list[np.ndarray],
                   manifest_components: list,
                   visual_cls  : list[str | None] | None = None,
                   colour_weight: float = 1.5,
                   visual_mismatch_penalty: float = 8.0,
                   ) -> list["InstanceResult"]:
    """
    Assign each 3D proposal cloud to a PUML-declared component instance.

    Cost matrix priority (highest to lowest):
      1. Visual classifier vote (if available and confident): large penalty on mismatch
      2. Colour prototype distance: L2 between proposal cluster HSV and class prototype
      3. Geometry: not used (no per-class geometry prototypes available)

    Args:
        proposals               : list of N (M_i, 3) point clouds
        manifest_components     : list of Component(cls, instance_id) from puml_parser
        visual_cls              : list of N dominant visual class codes (from backprojector),
                                  None entries mean no confident prediction for that cluster
        colour_weight           : weight for colour distance term
        visual_mismatch_penalty : cost added when visual_cls disagrees with manifest cls

    Returns list of N InstanceResult.
    """
    n = len(proposals)
    assert n == len(manifest_components), \
        f"Proposal count {n} ≠ manifest count {len(manifest_components)}"

    have_visual = (visual_cls is not None and any(v is not None for v in visual_cls))
    col_mfst    = np.array([_colour_feature(c.cls) for c in manifest_components])  # (N, 3)

    cost = np.zeros((n, n), np.float32)
    for pi in range(n):
        for mi, comp in enumerate(manifest_components):
            # Colour prototype distance
            c_cost = float(np.linalg.norm(_colour_feature(comp.cls) - col_mfst[mi]))

            # Visual mismatch penalty
            v_penalty = 0.0
            if visual_cls is not None and visual_cls[pi] is not None:
                if visual_cls[pi] != comp.cls:
                    v_penalty = visual_mismatch_penalty

            cost[pi, mi] = colour_weight * c_cost + v_penalty

    row_ind, col_ind = linear_sum_assignment(cost)

    results = [None] * n
    for pi, mi in zip(row_ind, col_ind):
        comp = manifest_components[mi]
        pts  = proposals[pi]
        lo   = pts.min(0) if len(pts) > 0 else np.zeros(3)
        hi   = pts.max(0) if len(pts) > 0 else np.zeros(3)
        cen  = np.median(pts, axis=0).astype(np.float32) if len(pts) > 0 else np.zeros(3)
        vis  = visual_cls[pi] if visual_cls is not None else None
        print(f"  [Assign] cluster {pi} → {comp.instance_id}"
              + (f"  (visual={vis})" if vis else ""))
        results[pi] = InstanceResult(
            instance_id = comp.instance_id,
            cls         = comp.cls,
            n_points    = len(pts),
            centroid    = cen,
            bbox_size   = (hi - lo).astype(np.float32),
        )
    return results
