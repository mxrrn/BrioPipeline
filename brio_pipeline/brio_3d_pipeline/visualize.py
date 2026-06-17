"""
3D scene visualisation for the BRIO pipeline.

Outputs written per sample
──────────────────────────
  viz_3d.ply   — full coloured point cloud (background + labelled components)
                 open in CloudCompare / MeshLab on Windows for interactive view
  viz_3d.png   — 3-panel PNG: isometric · top-down · front
                 (rendered headless via Open3D OffscreenRenderer)

Background layer
────────────────
Every DUSt3R frame contributes its pts3d array mapped to the actual image RGB,
giving a recognisable coloured reconstruction of the whole construction.
Component instance clouds are overlaid in bright label colours with
axis-aligned bounding boxes that naturally span split-rod fragments.

Interactive viewer
──────────────────
If DISPLAY is set (WSLg / X11 forwarding) the script also attempts to open
an interactive Open3D window after saving the files.

Usage
─────
    python visualize.py --sample 113
    python visualize.py --sample 114 --run run_007_20260612_1430
"""
import sys
import os
import json
import pickle
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# ── Label palette (RGB float, bright + distinguishable) ──────────────────────

_LABEL_COLORS: list[list[float]] = [
    [0.90, 0.10, 0.29],  # red
    [0.24, 0.71, 0.29],  # green
    [0.00, 0.51, 0.78],  # blue
    [0.96, 0.51, 0.19],  # orange
    [0.57, 0.12, 0.71],  # purple
    [0.27, 0.94, 0.94],  # cyan
    [0.94, 0.20, 0.90],  # magenta
    [0.75, 0.94, 0.27],  # lime
    [0.98, 0.75, 0.83],  # pink
    [0.00, 0.50, 0.50],  # teal
]


def _label_color(i: int) -> list[float]:
    return _LABEL_COLORS[i % len(_LABEL_COLORS)]


# ── Run-dir resolver ─────────────────────────────────────────────────────────

def _resolve_run_dir(run_arg: str | None) -> Path:
    if run_arg:
        d = cfg.OUTPUTS_ROOT / run_arg
        if not d.exists():
            raise FileNotFoundError(f"Run directory not found: {d}")
        return d
    candidates = sorted(
        [d for d in cfg.OUTPUTS_ROOT.iterdir()
         if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found in {cfg.OUTPUTS_ROOT}")
    return candidates[-1]


# ── Image loader ─────────────────────────────────────────────────────────────

def _load_rgb(path: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Scene geometry builder ────────────────────────────────────────────────────

def build_geometries(sample_id: int, run_dir: Path):
    """
    Return (geometries, instance_labels) where:
      geometries      — list of open3d.geometry objects ready for rendering
      instance_labels — list of (label_str, centroid_xyz) for annotation
    """
    import open3d as o3d

    out_dir       = run_dir / f"sample_{sample_id}"
    results_path  = out_dir / "results.json"
    proposals_dir = out_dir / "proposals"
    dust3r_cache  = out_dir / "dust3r" / "dust3r_cache.pkl"

    if not results_path.exists():
        print(f"[Viz3D] No results.json for sample {sample_id} — run pipeline first.")
        return None, None

    with open(results_path) as f:
        summary = json.load(f)

    # ── Load component proposals ──────────────────────────────────────────
    n = summary["n_components"]
    candidates = list(proposals_dir.glob(f"proposals_N{n}*.pkl"))
    if not candidates:
        print(f"[Viz3D] No proposals cache for N={n} in {proposals_dir}.")
        return None, None
    proposals_file = max(candidates, key=lambda p: p.stat().st_mtime)
    with open(proposals_file, "rb") as f:
        raw = pickle.load(f)
    proposals = raw[0] if isinstance(raw, tuple) else raw

    # ── Load DUSt3R result ────────────────────────────────────────────────
    dust3r_result = None
    if dust3r_cache.exists():
        with open(dust3r_cache, "rb") as f:
            dust3r_result = pickle.load(f)

    # ── Resolve ordered image paths (same sequence as DUSt3R / SAM) ──────
    order_file = out_dir / "image_order.json"
    if order_file.exists():
        image_paths = [Path(p) for p in json.loads(order_file.read_text())]
    elif dust3r_result:
        image_paths = [Path(p) for p in dust3r_result["image_paths"]]
    else:
        image_paths = []

    geometries: list = []
    instance_labels: list[tuple[str, np.ndarray]] = []

    # ── 1. Full coloured background cloud (Expert 1 + 3) ─────────────────
    if dust3r_result is not None and image_paths:
        confs = dust3r_result.get("confs")   # None for caches predating conf maps
        bg_pts, bg_col = [], []
        for frame_idx, (pts_frame, img_path) in enumerate(
                zip(dust3r_result["pts3d"], image_paths)):
            img_rgb = _load_rgb(img_path)
            if img_rgb is None:
                continue
            H, W = pts_frame.shape[:2]
            if img_rgb.shape[:2] != (H, W):
                img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            pts  = pts_frame.reshape(-1, 3).astype(np.float64)
            cols = img_rgb.reshape(-1, 3).astype(np.float64) / 255.0
            valid = np.linalg.norm(pts, axis=1) > 1e-6
            # Drop low-confidence DUSt3R points (textureless / misaligned regions)
            if confs is not None:
                valid &= confs[frame_idx].reshape(-1) >= cfg.DUST3R_CONF_THRESH
            # Drop white background / padding pixels — only the object remains
            valid &= np.any(img_rgb < 245, axis=2).reshape(-1)
            bg_pts.append(pts[valid])
            bg_col.append(cols[valid])

        if bg_pts:
            all_pts = np.concatenate(bg_pts, axis=0)
            all_col = np.concatenate(bg_col, axis=0)
            bg_pcd = o3d.geometry.PointCloud()
            bg_pcd.points = o3d.utility.Vector3dVector(all_pts)
            bg_pcd.colors = o3d.utility.Vector3dVector(all_col)
            bg_pcd = bg_pcd.voxel_down_sample(cfg.VOXEL_SIZE_BG)
            geometries.append(bg_pcd)
            print(f"[Viz3D] Background cloud: {len(bg_pcd.points):,} pts "
                  f"(voxel={cfg.VOXEL_SIZE_BG*100:.1f} cm)")

    # ── 2. Component instance clouds + AABBs (Expert 2 + 3 + 4) ─────────
    for i, (pts, inst) in enumerate(zip(proposals, summary["instances"])):
        if len(pts) == 0:
            continue
        color = _label_color(i)

        # Instance cloud — voxel downsampled, painted in label colour
        comp_pcd = o3d.geometry.PointCloud()
        comp_pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        comp_pcd.paint_uniform_color(color)
        comp_pcd = comp_pcd.voxel_down_sample(cfg.VOXEL_SIZE_COMPONENT)
        geometries.append(comp_pcd)

        # AABB — spans both halves of a split rod naturally
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(pts.astype(np.float64))
        )
        aabb.color = color
        geometries.append(aabb)

        centroid = np.median(pts, axis=0)
        instance_labels.append((inst["instance_id"], centroid))

    return geometries, instance_labels


# ── PLY export ────────────────────────────────────────────────────────────────

def save_ply(geometries: list, out_path: Path) -> None:
    """Merge all point clouds into a single PLY (open in CloudCompare / MeshLab)."""
    import open3d as o3d

    pts_list, col_list = [], []
    for g in geometries:
        if not isinstance(g, o3d.geometry.PointCloud) or len(g.points) == 0:
            continue
        pts_list.append(np.asarray(g.points))
        if g.has_colors():
            col_list.append(np.asarray(g.colors))
        else:
            col_list.append(np.ones((len(g.points), 3)) * 0.5)

    if not pts_list:
        return

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.concatenate(pts_list))
    merged.colors = o3d.utility.Vector3dVector(np.concatenate(col_list))
    o3d.io.write_point_cloud(str(out_path), merged)
    print(f"[Viz3D] PLY saved → {out_path}")


# ── Off-screen 3-panel PNG (Expert 5) ────────────────────────────────────────

def render_png(geometries: list, instance_labels: list, out_path: Path,
               sample_id: int) -> None:
    """
    Render three canonical views (isometric, top-down, front) into a 3-panel PNG
    using matplotlib scatter.

    Open3D OffscreenRenderer requires EGL which is not available in WSL2 even when
    DISPLAY is set (the X server on the Windows side doesn't provide EGL).  The PLY
    file is the high-quality interactive output; this PNG is a quick overview.
    """
    import open3d as o3d

    all_pts = []
    for g in geometries:
        if isinstance(g, o3d.geometry.PointCloud) and len(g.points) > 0:
            all_pts.append(np.asarray(g.points))
    if not all_pts:
        print("[Viz3D] No points — skipping PNG render.")
        return

    pts_all  = np.concatenate(all_pts, axis=0)
    centroid = pts_all.mean(axis=0)
    extent   = pts_all.max(axis=0) - pts_all.min(axis=0)
    radius   = float(np.linalg.norm(extent)) * 0.75

    panels = _matplotlib_panels(geometries, instance_labels, centroid, radius)
    _save_panel_png(panels, out_path, sample_id)


def _aabb_edges(mn: np.ndarray, mx: np.ndarray):
    """Return the 12 edge pairs of an axis-aligned bounding box."""
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    return [(corners[a], corners[b]) for a, b in edges]


def _matplotlib_panels(geometries, instance_labels, centroid, radius):
    """
    3-panel scatter plot: isometric, top-down, front.

    Background cloud: actual image RGB, small points, semi-transparent.
    Component clouds: bright label colours, larger points.
    Component AABBs: wireframe boxes in label colour.
    Axes are centred on the component clouds, not the full scene extent.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d
    from io import BytesIO

    # ── Separate background (first geometry) from component layers ────────
    bg_geom   = geometries[0] if geometries else None
    comp_geoms = []   # (pcd, aabb) pairs
    i = 1
    while i < len(geometries):
        pcd  = geometries[i]
        aabb = geometries[i + 1] if i + 1 < len(geometries) else None
        if isinstance(pcd, o3d.geometry.PointCloud):
            comp_geoms.append((pcd, aabb))
            i += 2
        else:
            i += 1

    # ── Compute tight axis limits from component clouds ───────────────────
    comp_pts = []
    for pcd, _ in comp_geoms:
        if len(pcd.points) > 0:
            comp_pts.append(np.asarray(pcd.points))
    if comp_pts:
        all_comp = np.concatenate(comp_pts)
        c_min = all_comp.min(axis=0)
        c_max = all_comp.max(axis=0)
        margin = max(float(np.linalg.norm(c_max - c_min)) * 0.4, 0.02)
        ax_cen = (c_min + c_max) / 2
    else:
        ax_cen, margin = centroid, radius

    views = [("Isometric", 20, 45), ("Top-down", 90, 0), ("Front", 0, 0)]
    panels = []

    for view_name, elev, azim in views:
        fig = plt.figure(figsize=(8, 6), facecolor="#1e1e1e")
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#1e1e1e")

        # Background cloud — actual RGB, small, semi-transparent
        if bg_geom is not None and isinstance(bg_geom, o3d.geometry.PointCloud) \
                and len(bg_geom.points) > 0:
            bg_pts  = np.asarray(bg_geom.points)
            bg_cols = np.asarray(bg_geom.colors) if bg_geom.has_colors() \
                      else np.ones((len(bg_pts), 3)) * 0.5
            # Clip background to scene area to avoid axis distortion
            in_range = np.all(
                (bg_pts >= ax_cen - margin * 1.5) &
                (bg_pts <= ax_cen + margin * 1.5), axis=1
            )
            bg_pts = bg_pts[in_range];  bg_cols = bg_cols[in_range]
            if len(bg_pts) > 0:
                n = min(15_000, len(bg_pts))
                idx = np.random.choice(len(bg_pts), n, replace=False)
                ax.scatter(bg_pts[idx, 0], bg_pts[idx, 1], bg_pts[idx, 2],
                           c=bg_cols[idx], s=1.5, alpha=0.35, linewidths=0,
                           zorder=1)

        # Component clouds — bright label colour, larger points
        for pcd, aabb in comp_geoms:
            if len(pcd.points) == 0:
                continue
            pts  = np.asarray(pcd.points)
            col  = np.asarray(pcd.colors)[0] if pcd.has_colors() else [1, 0, 0]
            n    = min(5_000, len(pts))
            idx  = np.random.choice(len(pts), n, replace=False)
            ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                       c=[col], s=4.0, alpha=0.85, linewidths=0, zorder=3)

            # AABB wireframe
            if aabb is not None and isinstance(aabb, o3d.geometry.AxisAlignedBoundingBox):
                mn = np.asarray(aabb.min_bound)
                mx = np.asarray(aabb.max_bound)
                for p0, p1 in _aabb_edges(mn, mx):
                    ax.plot3D([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                              color=list(aabb.color), lw=1.5, alpha=0.9, zorder=4)

        ax.set_xlim(ax_cen[0] - margin, ax_cen[0] + margin)
        ax.set_ylim(ax_cen[1] - margin, ax_cen[1] + margin)
        ax.set_zlim(ax_cen[2] - margin, ax_cen[2] + margin)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X", color="white"); ax.set_ylabel("Y", color="white")
        ax.set_zlabel("Z", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=120, facecolor="#1e1e1e")
        plt.close()
        buf.seek(0)
        img = np.frombuffer(buf.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        panels.append((view_name, img))

    return panels


def _save_panel_png(panels: list, out_path: Path, sample_id: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor("#1a1a1a")

    for ax, (view_name, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(view_name, color="white", fontsize=12, pad=6)
        ax.axis("off")

    fig.suptitle(f"Sample {sample_id} — 3D Instance Proposals",
                 color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()
    print(f"[Viz3D] 3-panel PNG saved → {out_path}")


# ── Main entry point ──────────────────────────────────────────────────────────

def visualize_sample(sample_id: int, run_dir: Path) -> None:
    import open3d as o3d

    out_dir = run_dir / f"sample_{sample_id}"
    geometries, instance_labels = build_geometries(sample_id, run_dir)
    if geometries is None:
        return

    # ── PLY (Expert 2B) — always saved, open in CloudCompare on Windows ──
    save_ply(geometries, out_dir / "viz_3d.ply")

    # ── 3-panel PNG via OffscreenRenderer with matplotlib fallback (E2A+5) ─
    render_png(geometries, instance_labels, out_dir / "viz_3d.png", sample_id)

    # ── Interactive window (Expert 2A) — only when a display is available ─
    if os.environ.get("DISPLAY"):
        try:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"Sample {sample_id} — BRIO 3D",
                width=1280, height=800,
            )
        except Exception as exc:
            print(f"[Viz3D] Interactive viewer unavailable: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BRIO 3D instance visualisation")
    parser.add_argument("--sample", type=int, default=113)
    parser.add_argument("--run",    type=str, default=None,
                        help="Run folder name (e.g. run_001_20260526_2123). "
                             "Defaults to the most recent run_* folder.")
    args    = parser.parse_args()
    run_dir = _resolve_run_dir(args.run)
    print(f"[Viz3D] Using run: {run_dir.name}")
    visualize_sample(args.sample, run_dir)
