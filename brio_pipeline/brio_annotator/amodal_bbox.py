"""
Amodal bounding box utilities.

For standard components: tight bbox directly from the SAM mask.

For rod classes (rolo, rome, rosm) with an occluded middle: the two
exposed ends produce two SAM masks.  If those masks are co-axial in 2D
(same elongation axis, collinear centroids), they belong to one rod and are
merged into a single amodal bounding box that spans both ends.

Key distinction: YOLO annotations are bounding boxes, not pixel masks.
A bounding box is amodal by definition — it encodes the full spatial extent
of an object regardless of occlusion.  YOLO trained on amodal boxes learns
to predict the full extent even from partial observations, which is exactly
the desired inference behaviour for occluded rods.
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Annotation:
    cls         : str
    instance_id : str
    bbox_xyxy   : tuple[int, int, int, int]   # (x0, y0, x1, y1) pixel coords
    mask        : np.ndarray                   # (H, W) bool — visible pixels only
    amodal      : bool = False                 # True when bbox extended past visible pixels


# ── Bbox helpers ──────────────────────────────────────────────────────────────

def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def union_bbox(a: tuple, b: tuple) -> tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


# ── Co-axial test (2D) ────────────────────────────────────────────────────────

def _pca2d(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (centroid_xy, principal_axis_xy, elongation σ1/σ2) for a 2D mask."""
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return np.zeros(2), np.array([1.0, 0.0]), 0.0
    pts  = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    _, S, Vt = np.linalg.svd(pts - mean, full_matrices=False)
    elong = float(S[0] / max(S[1], 1e-9))
    return mean, Vt[0], elong


def _masks_are_coaxial_2d(mask_a: np.ndarray, mask_b: np.ndarray,
                            angle_tol: float = 25.0,
                            min_elong: float = 1.5) -> bool:
    """
    Return True if two masks are likely the two exposed ends of the same rod.

    Criteria:
      1. Both are elongated (σ1/σ2 > min_elong).
      2. Their principal axes agree within angle_tol degrees.
      3. The vector between their centroids is roughly co-linear with that axis.
    """
    mean_a, axis_a, elong_a = _pca2d(mask_a)
    mean_b, axis_b, elong_b = _pca2d(mask_b)

    if elong_a < min_elong or elong_b < min_elong:
        return False

    cos_axes = abs(float(np.dot(axis_a, axis_b)))
    angle    = float(np.degrees(np.arccos(np.clip(cos_axes, 0.0, 1.0))))
    if angle > angle_tol:
        return False

    # Check collinearity: the centroid-to-centroid vector should align with the axis
    d    = mean_b - mean_a
    d_n  = np.linalg.norm(d)
    if d_n < 1e-6:
        return True
    cos_col = abs(float(np.dot(d / d_n, axis_a)))
    return cos_col > np.cos(np.radians(angle_tol))


# ── Main builder ──────────────────────────────────────────────────────────────

def build_annotations(masks      : list[np.ndarray],
                       assignments: list[tuple[str, str]],
                       rod_classes: set[str],
                       ) -> list[Annotation]:
    """
    Build one Annotation per PUML component from N masks and their class/id assignments.

    Hungarian assignment is 1-to-1, so two masks are never assigned the same
    instance_id here.  The rod amodal merge therefore applies when the annotator
    explicitly passes two masks for the same instance (future enhancement: after
    a two-pass re-assignment that detects rod ends).  For the current single-pass
    flow, each mask produces one annotation with its own tight bbox.

    The co-axial utilities above are available for callers that implement the
    two-pass strategy independently.
    """
    annotations: list[Annotation] = []
    for mask, (cls, iid) in zip(masks, assignments):
        bbox = mask_bbox(mask)
        if bbox is None:
            # Emit a zero-size placeholder so the caller can still count N results
            annotations.append(Annotation(cls=cls, instance_id=iid,
                                           bbox_xyxy=(0, 0, 0, 0),
                                           mask=mask))
            continue
        annotations.append(Annotation(cls=cls, instance_id=iid,
                                       bbox_xyxy=bbox, mask=mask))
    return annotations


def merge_rod_ends(ann_a: Annotation, ann_b: Annotation) -> Annotation:
    """
    Merge two rod-end annotations into one amodal annotation.

    Call this explicitly after detecting that two annotations are co-axial
    rod ends of the same physical component.  The merged annotation takes
    the union bounding box and the larger mask's instance_id.
    """
    combined_mask = np.logical_or(ann_a.mask, ann_b.mask)
    amodal_box    = union_bbox(ann_a.bbox_xyxy, ann_b.bbox_xyxy)
    # Keep the instance_id from the annotation with more mask pixels
    primary = ann_a if ann_a.mask.sum() >= ann_b.mask.sum() else ann_b
    return Annotation(cls=primary.cls, instance_id=primary.instance_id,
                       bbox_xyxy=amodal_box, mask=combined_mask, amodal=True)
