"""
Build classifier training crops from Mapping.drawio seed points via DUSt3R
backprojection.

Per sample:
  1. Load the parsed mapping (mapping_parser.py output) — instance seed
     points on ONE annotated multi-view photo, verified by pixel diff.
  2. Assemble the view list: the pipeline's standard selection
     (8 views/elevation ring) plus the annotated view if not already in it.
  3. Preprocess with the deployment preprocessor (fixed-scale crop at
     half = max(297, sample's own need), half-res downscale) — crops are
     therefore in the SAME visual domain the classifier sees at inference.
  4. Run (or load cached) DUSt3R on those views.
  5. Lift each seed point to 3D: median of high-confidence pts3d in a small
     window around the seed pixel in the annotated view.
  6. Reproject the 3D point into every view; keep views that pass:
       - in-bounds (inside DUSt3R's 4:3 center-crop region, with margin)
       - occlusion test (point depth vs DUSt3R depth map, relative tol)
       - foreground test (crop centre not white background)
  7. Cut a FIXED-SIZE window around each reprojected point from the
     half-res cropped image. Fixed window == absolute scale is preserved,
     so the network can use real size as a cue (a nut fills 5% of the
     window, a plate fills all of it) — no auxiliary size input needed.
  8. Sample background windows (foreground-overlapping but far from every
     seed) as explicit negatives for the one-vs-all "none" behaviour.

Outputs under --out (default classifier_training/dataset_crops/):
    <cls>/s{ID}_{instance}_{view}.jpg     positive crops
    _bg/s{ID}_neg{k}_{view}.jpg           background negatives
    manifest_{ID}.json                    per-crop metadata + reasons for
                                          every skipped (instance, view)
    debug_{ID}/                           (--render) seed overlays per view

Usage:
    python dataset_builder.py 113            # uses cache if present
    python dataset_builder.py 113 --render   # + per-view overlay renders
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PIPE = Path("/mnt/c/BA/00-project/brio_pipeline/brio_3d_pipeline")
HERE = Path(__file__).parent
sys.path.insert(0, str(PIPE))

import config as cfg                                   # noqa: E402
from detection.preprocessor import _lcc_info, crop_images        # noqa: E402
from detection.dust3r_runner import run_dust3r                   # noqa: E402

PARSED     = HERE / "parsed_mappings"
DEFAULT_OUT = HERE / "dataset_crops"
WORK       = HERE / "preproc"          # per-sample crops + dust3r caches

MIN_HALF     = 297     # matches run_033's cache; per-sample grows if needed
WINDOW       = 112     # crop window side (half-res px) — fixed for scale cue
CONF_MIN     = cfg.DUST3R_CONF_THRESH   # min DUSt3R confidence at the seed
SEED_WIN     = 7       # seed-lift window around the annotated pixel (odd)
OCCL_TOL     = 0.10    # occlusion tolerance × scene bbox diagonal
BORDER_FRAC  = 0.25    # window may hang ≤ this fraction outside the image
BG_PER_VIEW  = 2       # background negatives sampled per view
BG_MIN_DIST  = int(WINDOW * 0.6)  # min px distance of a negative from any seed
BG_FG_MAX    = 0.6     # max foreground fraction in a negative window
BG_CENTER_FG = 0.2     # centre 15×15 must be mostly background — the model
                       # learns "what is AT THE CENTRE"; a part at the window
                       # edge is fine (and a useful hard negative), a part at
                       # the centre labelled "none" teaches the wrong thing
RNG          = np.random.default_rng(42)


# ── Geometry helpers ─────────────────────────────────────────────────────────

def crop_to_dust3r(xy: np.ndarray, crop_side: int) -> np.ndarray:
    """Half-res cropped-image px → DUSt3R pts3d px (512-wide, 4:3 center crop:
    square inputs lose (512-384)/2 = 64 rows top and bottom)."""
    s = 512.0 / crop_side
    out = xy * s
    out[..., 1] -= 64.0
    return out


def dust3r_to_crop(xy: np.ndarray, crop_side: int) -> np.ndarray:
    s = crop_side / 512.0
    out = xy.copy()
    out[..., 1] += 64.0
    return out * s


def project(P: np.ndarray, K: np.ndarray, pose_c2w: np.ndarray
            ) -> tuple[float, float, float]:
    """World point → (u, v, z_cam) in DUSt3R pixel coords of that view."""
    w2c = np.linalg.inv(pose_c2w)
    pc  = w2c[:3, :3] @ P + w2c[:3, 3]
    z   = float(pc[2])
    if z <= 1e-9:
        return -1.0, -1.0, z
    uvw = K @ pc
    return float(uvw[0] / z), float(uvw[1] / z), z


def scene_diag(dust3r: dict) -> float:
    pts = []
    for p3, cf in zip(dust3r["pts3d"], dust3r["confs"]):
        m = cf > CONF_MIN
        if m.any():
            pts.append(p3[m][::17])
    allp = np.concatenate(pts, axis=0)
    lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
    return float(np.linalg.norm(hi - lo))


def load_color_prototypes() -> dict[str, np.ndarray]:
    """Median foreground BGR per class, derived from the isolated component
    photos (cached to JSON). Used to make seed snapping class-aware: a `nu`
    (black nut) seed must not anchor on metallic rod thread it happens to
    hit first."""
    cache = HERE / "color_prototypes.json"
    if cache.exists():
        return {k: np.array(v, dtype=np.float32)
                for k, v in json.loads(cache.read_text()).items()}
    from classification.component_map import FOLDER_TO_CLASS
    protos: dict[str, np.ndarray] = {}
    for folder, code in FOLDER_TO_CLASS.items():
        cdir = cfg.COMPONENT_DATASET / folder
        imgs = (sorted(cdir.glob("*.jpg")) + sorted((cdir / "Images30").glob("*.jpg")))[:5]
        pix = []
        for p in imgs:
            im = cv2.imread(str(p))
            if im is None:
                continue
            # The object is centred on a large white field in these studio
            # photos; a whole-foreground median is dominated by soft shadow
            # (measured: nut → grey [125,128,131] instead of black). Sample
            # only the central disc, where the object itself is.
            H, W = im.shape[:2]
            r = int(min(H, W) * 0.10)
            patch = im[H // 2 - r:H // 2 + r, W // 2 - r:W // 2 + r]
            m = np.any(patch < 245, axis=2)
            if m.sum() > 50:
                pix.append(patch[m])
        if pix:
            protos[code] = np.median(np.concatenate(pix), axis=0).astype(np.float32)
    cache.write_text(json.dumps({k: v.tolist() for k, v in protos.items()}))
    return protos


COLOR_ACCEPT = 65.0    # max colour distance to the class prototype to accept
COLOR_FALLBK = 100.0   # fallback: best point along the ray must beat this


def _color_metric(patch_med: np.ndarray, proto: np.ndarray) -> float:
    """0.5·L2(BGR) + 3·L2(chroma), chroma = (B−G, G−R). The chroma term
    separates chromatic wood (B<G<R, spread ~25) from achromatic shadow on
    white plastic — in plain BGR the two are ~27 apart and a wood seed
    happily snapped onto shadowed plate (observed on 113's blwo11_1)."""
    d_bgr = float(np.linalg.norm(patch_med - proto))
    ch_p  = np.array([patch_med[0] - patch_med[1], patch_med[1] - patch_med[2]])
    ch_q  = np.array([proto[0] - proto[1], proto[1] - proto[2]])
    return 0.5 * d_bgr + 3.0 * float(np.linalg.norm(ch_p - ch_q))


def snap_to_foreground(img: np.ndarray, fg: np.ndarray, x: float, y: float,
                       d: tuple[float, float] | None, max_dist: float,
                       proto: np.ndarray | None
                       ) -> tuple[float, float, float] | None:
    """The drawio arrowheads systematically stop in empty space just outside
    the part they mean (verified pixel-level on sample 113). Recover the
    intended anchor by marching from the endpoint along the arrow direction
    until solid foreground WHOSE COLOUR MATCHES THE CLASS (so a nut seed
    walks past the rod thread until it reaches something nut-coloured),
    then push a few px further inside to clear the anti-aliased edge.
    Scans a small perpendicular band so thin parts a few px off the ray are
    still hit. Returns (x, y, marched_dist) or None."""
    H, W = fg.shape

    def solid(px: float, py: float) -> bool:
        xi, yi = int(round(px)), int(round(py))
        if not (2 <= xi < W - 2 and 2 <= yi < H - 2):
            return False
        return fg[yi - 2:yi + 3, xi - 2:xi + 3].mean() > 0.6

    def color_dist(px: float, py: float) -> float:
        xi, yi = int(round(px)), int(round(py))
        patch = img[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4]
        pfg   = fg[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4]
        if proto is None or not pfg.any():
            return 0.0 if proto is None else 1e9
        med = np.median(patch[pfg].astype(np.float32), axis=0)
        return _color_metric(med, proto)

    if d is None or (not d[0] and not d[1]):
        return None
    base = float(np.arctan2(d[1], d[0]))
    best = None               # (dist, x, y, t) — colour fallback
    # Cone search: the drawio arrow direction (even from the true exit point)
    # can be ±30° off what the arrowhead visually points at (drawio routes
    # edges, it doesn't draw the straight exit→entry line) — sweep a ±36°
    # cone, nearest hits first; the colour gate keeps wrong-part risk down.
    angles = [base + np.deg2rad(a) for a in
              (0, 4, -4, 8, -8, 12, -12, 16, -16, 20, -20,
               24, -24, 28, -28, 32, -32, 36, -36)]
    t = 0.0
    while t <= max_dist:
        for ang in angles:
            ux, uy = np.cos(ang), np.sin(ang)
            px, py = x + t * ux, y + t * uy
            if not solid(px, py):
                continue
            cd = color_dist(px, py)
            if cd < COLOR_ACCEPT:
                # push deeper while colour keeps matching (max 14 px)
                for extra in (14, 10, 6, 3, 0):
                    ex, ey = px + extra * ux, py + extra * uy
                    if solid(ex, ey) and color_dist(ex, ey) < COLOR_ACCEPT:
                        return ex, ey, t + extra
            if best is None or cd < best[0]:
                best = (cd, px, py, t)
        t += 2.0
    if best is not None and best[0] < COLOR_FALLBK:
        return best[1], best[2], best[3]
    return None


# ── Per-sample pipeline ──────────────────────────────────────────────────────

def collect_images(sample_id: int) -> list[Path]:
    """Mirror pipeline.collect_images exactly (8 views/ring, sorted-glob stride)."""
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


def sample_half(image_paths: list[Path]) -> int:
    from preprocessor import _sample_lcc_halfsize
    need = int(np.ceil(_sample_lcc_halfsize(image_paths) * (1 + cfg.AUTO_CROP_PADDING)))
    return max(MIN_HALF, need)


def build_sample(sample_id: int, out_dir: Path, render: bool = False,
                 device: str = cfg.DEVICE) -> dict | None:
    parsed_file = PARSED / f"sample_{sample_id}.json"
    if not parsed_file.exists():
        print(f"[{sample_id}] no parsed mapping — run mapping_parser first")
        return None
    parsed = json.loads(parsed_file.read_text())
    seeds = [p for p in parsed["points"] if p["kind"] == "instance"]
    match = (parsed.get("image") or {}).get("match")
    if not seeds or not match:
        print(f"[{sample_id}] skipped: {len(seeds)} trusted seeds, image match={match}")
        return None
    annotated = Path(match)

    # View list: standard pipeline selection + annotated view
    views = collect_images(sample_id)
    if annotated not in views:
        views.append(annotated)

    # Remove the two gratuitous nondeterminism sources (audit P2): DUSt3R's
    # global alignment optimises from random init — seed it per sample.
    import torch
    torch.manual_seed(42)
    np.random.seed(42)

    work = WORK / f"sample_{sample_id}"
    half = sample_half(views)
    cropped_paths = crop_images(views, work / "cropped", half)
    crop_side = half  # after half-res downscale, crops are half×half px

    dust3r = run_dust3r(cropped_paths, work / "dust3r", device=device,
                        size=cfg.DUST3R_SIZE, niter=cfg.DUST3R_NITER)
    # All per-view arrays (poses, pts3d, depths) are indexed by the order in
    # dust3r["image_paths"] — realign our crop list to it so a cache built
    # from a differently-ordered call can never silently shift views.
    by_name = {p.name: p for p in cropped_paths}
    cropped_paths = [by_name[Path(p).name] for p in dust3r["image_paths"]]
    name_to_idx = {Path(p).name: i for i, p in enumerate(dust3r["image_paths"])}
    ann_name = f"{annotated.parent.name}_{annotated.name}"
    ann_idx  = name_to_idx[ann_name]

    diag = scene_diag(dust3r)
    poses, Ks = dust3r["poses"], dust3r["intrinsics"]
    imgs_bgr = [cv2.imread(str(p)) for p in cropped_paths]

    # ── Seed lift: annotated-photo fraction → 3D point ──
    ann_full = cv2.imread(str(annotated))
    info = _lcc_info(ann_full)
    if info is None:
        print(f"[{sample_id}] annotated view has no foreground?!")
        return None
    Hf, Wf = ann_full.shape[:2]
    manifest = {"sample_id": sample_id, "window": WINDOW, "half": half,
                "annotated": str(annotated), "crops": [], "skips": [],
                "seed3d": {}}
    # Foreground incl. overexposed white-plastic interiors: the plate's top
    # face reads >245 and is invisible to the raw threshold (a plpl seed ray
    # marched straight through the plate onto the rod thread — verified on
    # 113); a wide morphological close fills it from its darker edges.
    fg_raw  = np.any(ann_full < 245, axis=2).astype(np.uint8)
    fg_full = cv2.morphologyEx(
        fg_raw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(bool)
    protos = load_color_prototypes()
    seeds3d = {}
    for s in seeds:
        x_full, y_full = s["fx"] * Wf, s["fy"] * Hf
        # Snap the stub arrowhead onto the part it points at (see docstring)
        d = None
        if s.get("dfx") is not None:
            d = (s["dfx"] * Wf, s["dfy"] * Hf)
        max_dist = 400.0 if s.get("coord_src") == "abs" else 250.0
        snap = snap_to_foreground(ann_full, fg_full, x_full, y_full, d,
                                  max_dist, protos.get(s["cls"]))
        if snap is None:
            manifest["skips"].append({"base": s["base"], "view": "*",
                                      "reason": "snap found no foreground"})
            continue
        x_full, y_full, snap_dist = snap
        # full-res → half-res cropped coords
        x_c = (x_full - (info["cx"] - half)) / 2.0
        y_c = (y_full - (info["cy"] - half)) / 2.0
        u, v = crop_to_dust3r(np.array([x_c, y_c]), crop_side)
        ui, vi = int(round(u)), int(round(v))
        p3, cf = dust3r["pts3d"][ann_idx], dust3r["confs"][ann_idx]
        Hd, Wd = cf.shape
        r = SEED_WIN // 2
        y0, y1 = max(0, vi - r), min(Hd, vi + r + 1)
        x0, x1 = max(0, ui - r), min(Wd, ui + r + 1)
        if y0 >= y1 or x0 >= x1:
            manifest["skips"].append({"base": s["base"], "view": "*",
                                      "reason": "seed outside dust3r frame"})
            continue
        win_p, win_c = p3[y0:y1, x0:x1].reshape(-1, 3), cf[y0:y1, x0:x1].reshape(-1)
        good = win_c > CONF_MIN
        if not good.any():
            manifest["skips"].append({"base": s["base"], "view": "*",
                                      "reason": "no confident pts3d at seed"})
            continue
        P = np.median(win_p[good], axis=0)
        seeds3d[s["base"]] = (P, s["cls"])
        manifest["seed3d"][s["base"]] = {"P": P.tolist(),
                                         "conf_med": float(np.median(win_c[good])),
                                         "snap_dist": snap_dist,
                                         "snapped_full": [x_full, y_full]}

    if not seeds3d:
        print(f"[{sample_id}] no seed survived 3D lift")
        return manifest

    # ── Reproject into every view, crop, filter ──
    n_pos = 0
    dbg_dir = out_dir / f"debug_{sample_id}"
    if render:
        dbg_dir.mkdir(parents=True, exist_ok=True)
    seed_px_per_view: dict[int, list[tuple[float, float]]] = {}

    for vi_idx, (cp, bgr) in enumerate(zip(cropped_paths, imgs_bgr)):
        if bgr is None:
            continue
        vname = Path(cp).stem
        dbg = bgr.copy() if render else None
        seed_px_per_view[vi_idx] = []
        for base, (P, cls) in seeds3d.items():
            u, v, z = project(P, Ks[vi_idx], poses[vi_idx])
            reason = None
            if z <= 0 or not (0 <= u < 512 and 0 <= v < 384):
                reason = "behind/outside"
            else:
                d_at = float(dust3r["depths"][vi_idx][int(v), int(u)])
                gap  = z - d_at
                if gap > OCCL_TOL * diag:
                    reason = f"occluded gap={gap/diag:.2f}d"
            if reason is None:
                xy = dust3r_to_crop(np.array([u, v]), crop_side)
                x, y = float(xy[0]), float(xy[1])
                h_, w_ = bgr.shape[:2]
                hw = WINDOW / 2
                out_frac = (max(0, hw - x) + max(0, x + hw - w_)) / WINDOW \
                         + (max(0, hw - y) + max(0, y + hw - h_)) / WINDOW
                if out_frac > BORDER_FRAC:
                    reason = "window too far outside"
                else:
                    c = bgr[max(0, int(y) - 7):int(y) + 8, max(0, int(x) - 7):int(x) + 8]
                    if c.size == 0 or c.min() > 245:
                        reason = "background at centre"
            if reason is not None:
                manifest["skips"].append({"base": base, "view": vname, "reason": reason})
                if render and reason != "behind/outside":
                    pass
                continue
            seed_px_per_view[vi_idx].append((x, y))
            # fixed window crop, white-padded at borders
            canvas = np.full((WINDOW, WINDOW, 3), 255, np.uint8)
            xi0, yi0 = int(round(x - hw)), int(round(y - hw))
            sx0, sy0 = max(0, xi0), max(0, yi0)
            sx1, sy1 = min(w_, xi0 + WINDOW), min(h_, yi0 + WINDOW)
            canvas[sy0 - yi0:sy1 - yi0, sx0 - xi0:sx1 - xi0] = bgr[sy0:sy1, sx0:sx1]
            cls_dir = out_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            fname = f"s{sample_id}_{base}_{vname}.jpg"
            cv2.imwrite(str(cls_dir / fname), canvas)
            manifest["crops"].append({"cls": cls, "base": base, "view": vname,
                                      "x": x, "y": y, "file": f"{cls}/{fname}"})
            n_pos += 1
            if render:
                cv2.circle(dbg, (int(x), int(y)), 6, (0, 0, 255), 2)
                cv2.putText(dbg, base, (int(x) + 8, int(y) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        if render:
            cv2.imwrite(str(dbg_dir / f"{vname}.jpg"), dbg)

    # ── Background negatives ──
    n_neg = 0
    bg_dir = out_dir / "_bg"
    for vi_idx, (cp, bgr) in enumerate(zip(cropped_paths, imgs_bgr)):
        if bgr is None or vi_idx not in seed_px_per_view:
            continue
        vname = Path(cp).stem
        h_, w_ = bgr.shape[:2]
        fg = np.any(bgr < 245, axis=2)
        pts = seed_px_per_view[vi_idx]
        placed, tries = 0, 0
        while placed < BG_PER_VIEW and tries < 300:
            tries += 1
            x = RNG.uniform(WINDOW / 2, w_ - WINDOW / 2)
            y = RNG.uniform(WINDOW / 2, h_ - WINDOW / 2)
            if pts and min((x - a) ** 2 + (y - b) ** 2 for a, b in pts) < BG_MIN_DIST ** 2:
                continue
            win = fg[int(y - WINDOW / 2):int(y + WINDOW / 2),
                     int(x - WINDOW / 2):int(x + WINDOW / 2)]
            # negatives should still look like the domain: demand some
            # foreground context but an off-part centre
            if not (0.02 < win.mean() < BG_FG_MAX):
                continue
            ctr = fg[int(y) - 7:int(y) + 8, int(x) - 7:int(x) + 8]
            if ctr.size == 0 or ctr.mean() > BG_CENTER_FG:
                continue
            crop = bgr[int(y - WINDOW / 2):int(y + WINDOW / 2),
                       int(x - WINDOW / 2):int(x + WINDOW / 2)]
            bg_dir.mkdir(parents=True, exist_ok=True)
            fname = f"s{sample_id}_neg{placed}_{vname}.jpg"
            cv2.imwrite(str(bg_dir / fname), crop)
            manifest["crops"].append({"cls": "_bg", "base": f"neg{placed}",
                                      "view": vname, "x": x, "y": y,
                                      "file": f"_bg/{fname}"})
            placed += 1
            n_neg += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"manifest_{sample_id}.json").write_text(json.dumps(manifest, indent=2))
    n_skip = len(manifest["skips"])
    print(f"[{sample_id}] {len(seeds3d)}/{len(seeds)} seeds lifted | "
          f"{n_pos} positive crops, {n_neg} negatives, {n_skip} skips "
          f"(half={half}, diag={diag:.3f})")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("samples", nargs="+", type=int)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--device", default=cfg.DEVICE)
    args = ap.parse_args()
    for sid in args.samples:
        try:
            build_sample(sid, args.out, render=args.render, device=args.device)
        except Exception as e:
            print(f"[{sid}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
