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
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from pathlib import Path


# ── Nominal sizes (scale-free size prior) ────────────────────────────────────
# BRIO class codes encode dimensions in studs: blwo21 = block 2×1,
# plpl53 = plate 5×3, stwo9 = strap with 9 holes, etc.  For codes without
# digits, approximate relative long-dimension in the same unit.  Only the
# RATIO between manifest entries is used, so the absolute unit cancels out
# (DUSt3R scale is arbitrary per sample).
_NOMINAL_NO_DIGITS = {
    "rolo": 8.0, "rome": 6.0, "rosm": 4.0,      # rods long/medium/small
    "sclo": 4.0, "scme": 3.0, "scsm": 2.0,      # screws
    "nu": 1.2, "wa": 0.8, "pl": 1.0, "sl": 2.0, # nut, washer, plug, sleeve
    "bo": 1.5,                                    # bolt: short thin cylinder
    "no": 2.0, "ti": 3.0,                        # nose, tire
    "whre": 2.0, "whwh": 2.0,                    # wheels
    # blwo (block-windmill-opening) pieces: "11"/"21" encodes opening variant,
    # NOT a size dimension.  Physical size >> any rod, so override the digit
    # parser which would return max(1,1)=1 or max(2,1)=2.
    "blwo11": 9.0, "blwo21": 7.0,
}


# ── Nominal elongation (σ₁/σ₂ PCA ratio) ────────────────────────────────────
# Encodes how elongated each class is geometrically (1 = sphere, high = rod).
# Used as a log-scale cost: |log(obs_elong) - log(nom_elong)|, which makes the
# rod vs. block distinction very robust to size-metric failures.
_NOMINAL_ELONG = {
    # Rods: clearly elongated cylinders
    "rolo": 8.0, "rome": 6.0, "rosm": 4.0,
    # Screws: similarly elongated
    "sclo": 4.0, "scme": 3.5, "scsm": 2.5,
    # Straps: elongated flat strips (digit encodes hole count = length proxy)
    "stwo3": 3.0, "stwo5": 5.0, "stwo7": 7.0, "stwo9": 9.0,
    # Windmill blocks: cross-shaped, nearly isotropic (digits = opening variant)
    "blwo11": 1.5, "blwo21": 1.6,
    # Small round / thin hardware
    "nu": 1.2, "wa": 1.1, "pl": 1.2, "sl": 2.5,
    "bo": 2.0, "no": 2.0, "ti": 1.3, "whre": 1.3, "whwh": 1.3,
}
_DEFAULT_ELONG = 1.7   # used for digit-encoded plates (e.g. plpl53: 5/3 ≈ 1.67)


def _nominal_elong(cls: str) -> float:
    """Nominal PCA elongation (σ₁/σ₂) for a BRIO class."""
    if cls in _NOMINAL_ELONG:
        return _NOMINAL_ELONG[cls]
    # Digit-encoded plates / blocks: long-side / short-side gives aspect ratio.
    # E.g. plpl53 → digits [5,3] → 5/3 = 1.67; blwo21 → 2/1 handled by dict.
    digits = [int(d) for d in re.findall(r"\d", cls)]
    if len(digits) >= 2:
        d = sorted(digits, reverse=True)
        return max(float(d[0]) / max(float(d[1]), 1.0), 1.0)
    return _DEFAULT_ELONG


def _nominal_size(cls: str) -> float:
    """Nominal long-dimension of a class in BRIO stud units."""
    if cls in _NOMINAL_NO_DIGITS:
        return _NOMINAL_NO_DIGITS[cls]
    digits = [int(d) for d in re.findall(r"\d", cls)]
    return float(max(digits)) if digits else 1.0


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
                   cluster_elongations: list[float] | None = None,
                   colour_weight: float = 2.5,
                   size_weight  : float = 1.0,
                   elong_weight : float = 1.0,
                   visual_mismatch_penalty: float = 8.0,
                   ) -> list["InstanceResult"]:
    """
    Assign each 3D proposal cloud to a PUML-declared component instance.

    Cost matrix priority (highest to lowest):
      1. Visual classifier vote (if available and confident): large penalty on mismatch
      2. Colour distance: L2 between observed cluster HSV and class prototype HSV
      3. Size: |relative observed extent − relative nominal extent|.  Both are
         normalised by the largest value within the sample, so the comparison
         is scale-free (DUSt3R reconstruction scale is arbitrary).

    Args:
        proposals               : list of N (M_i, 3) point clouds
        manifest_components     : list of Component(cls, instance_id) from puml_parser
        visual_cls              : list of N dominant visual class codes (from backprojector),
                                  None entries mean no confident prediction for that cluster
        cluster_colours         : list of N mean observed HSV (3,) arrays in [0,1],
                                  used to compute colour distance to class prototypes
        colour_weight           : weight for colour distance term
        elong_weight            : weight for elongation log-distance term
        visual_mismatch_penalty : cost added when visual_cls disagrees with manifest cls

    Returns list of N InstanceResult.
    """
    import math
    n = len(proposals)
    assert n == len(manifest_components), \
        f"Proposal count {n} ≠ manifest count {len(manifest_components)}"

    have_colour = cluster_colours is not None
    have_elong  = cluster_elongations is not None

    # Scale-free size features: observed max extent per proposal and nominal
    # size per manifest entry, each normalised by the in-sample maximum.
    obs_ext = np.array([
        float((p.max(0) - p.min(0)).max()) if len(p) > 0 else 0.0
        for p in proposals
    ], np.float32)
    rel_obs = obs_ext / max(obs_ext.max(), 1e-6)
    nom = np.array([_nominal_size(c.cls) for c in manifest_components], np.float32)
    rel_nom = nom / max(nom.max(), 1e-6)

    cost = np.zeros((n, n), np.float32)
    for pi in range(n):
        obs_hsv   = cluster_colours[pi] if have_colour else None
        obs_elong = cluster_elongations[pi] if have_elong else None
        for mi, comp in enumerate(manifest_components):
            proto = _colour_feature(comp.cls)

            # L2 distance between observed cluster HSV and class prototype
            c_cost = float(np.linalg.norm(obs_hsv - proto)) if obs_hsv is not None else 0.0

            # Relative-size mismatch (empty proposals carry no size signal)
            s_cost = abs(float(rel_obs[pi]) - float(rel_nom[mi])) \
                     if obs_ext[pi] > 0 else 0.0

            # Elongation mismatch: log-scale |log(obs) − log(nom)|.
            # Separates rods (elong ≈ 6) from blocks (elong ≈ 1.5) even when
            # the size metric fails due to plate bbox > block bbox.
            e_cost = 0.0
            if obs_elong is not None and obs_elong > 0:
                nom_e = _nominal_elong(comp.cls)
                e_cost = abs(math.log(max(obs_elong, 1.0)) - math.log(max(nom_e, 1.0)))

            # Visual mismatch penalty
            v_penalty = 0.0
            if visual_cls is not None and visual_cls[pi] is not None:
                if visual_cls[pi] != comp.cls:
                    v_penalty = visual_mismatch_penalty

            cost[pi, mi] = (colour_weight * c_cost + size_weight * s_cost
                            + elong_weight * e_cost + v_penalty)

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
