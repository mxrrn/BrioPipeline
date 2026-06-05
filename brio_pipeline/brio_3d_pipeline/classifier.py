"""
Classify 3D instance proposals against the PUML component manifest using
Hungarian assignment on a combined geometry + colour feature vector.

Colour features
───────────────
Per-cluster observed mean HSV (from backprojector) is compared against
per-class prototype HSV built from reference images in
02-resources/data/component_images/.  This L2 distance forms the colour cost.

Visual classifier
─────────────────
When the ComponentClassifier has made a confident prediction for a cluster,
any manifest entry whose class disagrees incurs a large fixed penalty,
making the visual vote the dominant assignment signal.

Assignment
──────────
Hungarian assignment (scipy linear_sum_assignment) minimises total cost
across all N proposal → manifest pairings.  No greedy local decisions.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InstanceResult:
    instance_id : str
    cls         : str
    n_points    : int
    centroid    : np.ndarray   # (3,) world coords
    bbox_size   : np.ndarray   # (3,) axis-aligned extents


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
                   cluster_colours: list[np.ndarray] | None = None,
                   colour_weight: float = 1.5,
                   visual_mismatch_penalty: float = 8.0,
                   ) -> list["InstanceResult"]:
    """
    Assign each 3D proposal cloud to a PUML-declared component instance.

    Cost matrix priority (highest to lowest):
      1. Visual classifier vote (if available and confident): large penalty on mismatch
      2. Colour distance: L2 between observed cluster HSV and class prototype HSV
      3. Geometry: not used (no per-class geometry prototypes available)

    Args:
        proposals               : list of N (M_i, 3) point clouds
        manifest_components     : list of Component(cls, instance_id) from puml_parser
        visual_cls              : list of N dominant visual class codes (from backprojector),
                                  None entries mean no confident prediction for that cluster
        cluster_colours         : list of N mean observed HSV (3,) arrays in [0,1],
                                  used to compute colour distance to class prototypes
        colour_weight           : weight for colour distance term
        visual_mismatch_penalty : cost added when visual_cls disagrees with manifest cls

    Returns list of N InstanceResult.
    """
    n = len(proposals)
    assert n == len(manifest_components), \
        f"Proposal count {n} ≠ manifest count {len(manifest_components)}"

    have_colour = cluster_colours is not None

    cost = np.zeros((n, n), np.float32)
    for pi in range(n):
        obs_hsv = cluster_colours[pi] if have_colour else None
        for mi, comp in enumerate(manifest_components):
            proto = _colour_feature(comp.cls)

            # L2 distance between observed cluster HSV and class prototype
            c_cost = float(np.linalg.norm(obs_hsv - proto)) if obs_hsv is not None else 0.0

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
