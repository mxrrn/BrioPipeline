"""
Infer slot connections between 3D component instances.

Approach
────────
Two components are considered connected if:
  1. Their 3D centroids are within CONNECTION_DIST_THRESH world units, AND
  2. Their class pair appears in the SLOT_RULES compatibility table.

This produces an undirected connection graph.  The result is a list of
(instance_id_A, instance_id_B) edges, matching the link syntax in PUML.

Limitations
───────────
- Purely proximity-based: works well when components are touching or nearly
  touching in 3D, which is true for assembled BRIO constructions.
- Does not model which specific SLOT holes are used — only THAT a connection
  exists between two instances.
- For stacked or collinear components the distance threshold may over- or
  under-connect. Tunable via cfg.CONNECTION_DIST_THRESH.
"""
import sys
import numpy as np
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from triangulator import Detection3D


def infer_connections(instances: list[Detection3D]) -> list[tuple[str, str]]:
    """
    Return a list of (instance_id_A, instance_id_B) connection pairs.

    Args:
        instances : output of triangulator.triangulate()

    Returns list of edge tuples (unordered pairs).
    """
    edges: list[tuple[str, str]] = []

    for a, b in combinations(instances, 2):
        # Check class compatibility
        pair = frozenset({a.cls, b.cls})
        if pair not in cfg.SLOT_RULES:
            continue

        # Check 3D proximity
        dist = float(np.linalg.norm(a.centroid - b.centroid))
        if dist <= cfg.CONNECTION_DIST_THRESH:
            edges.append((a.instance_id, b.instance_id))

    return edges


def connection_summary(instances: list[Detection3D],
                       edges: list[tuple[str, str]]) -> str:
    """Human-readable connection report."""
    lines = [f"  {a} ── {b}" for a, b in edges]
    if not lines:
        return "  (no connections inferred)"
    return "\n".join(lines)
