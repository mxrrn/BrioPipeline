"""
Parse Mapping.drawio ground-truth annotations into per-sample seed points
and connection relations.

Each Sample_N_InstanceDiagram/Mapping.drawio contains:
  * one embedded construction photo (an mxCell with shape=image),
  * one labeled vertex box per component instance (e.g. "nu_1") and per
    slot/feature of an instance (e.g. "nu_1&{th}_1"),
  * arrows from those boxes into the photo, whose (entryX, entryY) style
    attributes are FRACTIONS of the image cell's box marking where that
    instance/slot sits in the photo.

Coordinate conversion: the image cell's style carries `imageAspect=0`,
i.e. the bitmap is stretched to fill the cell box without preserving
aspect ratio — so a fraction maps to the source photo as simply
(entryX * photo_W, entryY * photo_H), regardless of the cell box's own
aspect ratio. (This was an open question in the plan doc; `imageAspect=0`
settles it, and the render step below exists to verify it visually.)

Outputs, per sample:
  * seed points  — {label, base instance, class code, slot type or None,
                    x_px, y_px} — the A2 review inputs
  * relations    — label→label edges (slot ownership / connections), kept
                   as structured data for the future PUML-generation stage
                   (decision B1); not consumed by classifier training
  * image match  — which multi-view image the embedded photo actually is
                   (verified by pixel diff, not assumed)

Usage:
    python mapping_parser.py 112 113 114 115 116 --render
    → writes <out>/sample_N.json and (with --render) seed_render_N.png
"""
import argparse
import base64
import html
import json
import re
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

CONSTRUCTIONS = Path("/mnt/c/BA/02-resources/data/constructions")
MULTI_VIEW    = Path("/mnt/c/BA/02-resources/data/multi_view_images")
DEFAULT_OUT   = Path(__file__).parent / "parsed_mappings"

# Instance label:  cls_index          e.g. nu_1, plwo53_2
# Slot label:      cls_index&{slot}_k e.g. nu_1&{th}_1, bo_1&{pi}_2
_INSTANCE_RE = re.compile(r"^([a-z][a-z0-9]*)_(\d+)$")
_SLOT_RE     = re.compile(r"^([a-z][a-z0-9]*_\d+)&\{(\w+)\}_(\d+)$")


def _find_sample_dir(sample_id: int) -> Path:
    name = f"Sample_{sample_id}_InstanceDiagram"
    hits = list(CONSTRUCTIONS.glob(f"*/{name}")) + list(CONSTRUCTIONS.glob(name))
    if not hits:
        raise FileNotFoundError(f"No {name} under {CONSTRUCTIONS}")
    return hits[0]


def _load_graph_root(drawio_path: Path) -> ET.Element:
    """Return the mxGraphModel root, handling both plain and
    deflate-compressed drawio diagrams."""
    tree = ET.parse(drawio_path)
    diagram = tree.getroot().find(".//diagram")
    if diagram is None:
        raise ValueError(f"No <diagram> in {drawio_path}")
    model = diagram.find("mxGraphModel")
    if model is not None:
        return model
    # Compressed variant: diagram text = base64(raw-deflate(urlencoded xml))
    raw = zlib.decompress(base64.b64decode(diagram.text.strip()), wbits=-15)
    from urllib.parse import unquote
    return ET.fromstring(unquote(raw.decode()))


def _clean_label(value: str) -> str:
    """Vertex values are HTML like '<font ...>nu_1&amp;{th}_1</font>' —
    strip tags, unescape entities."""
    text = re.sub(r"<[^>]+>", "", value)
    return html.unescape(text).strip()


def _style_dict(style: str) -> dict[str, str]:
    out = {}
    for part in (style or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def parse_mapping(sample_id: int) -> dict:
    sample_dir = _find_sample_dir(sample_id)
    root = _load_graph_root(sample_dir / "Mapping.drawio")

    cells = root.findall(".//mxCell")
    image_cell_id = None
    image_geom = None                    # (x, y, w, h) of the image cell box
    labels: dict[str, str] = {}          # cell id → cleaned label text
    label_geoms: dict[str, tuple[float, float, float, float]] = {}  # id → box
    for c in cells:
        style = c.get("style", "")
        if "shape=image" in style:
            image_cell_id = c.get("id")
            g = c.find("mxGeometry")
            if g is not None:
                image_geom = (float(g.get("x", 0)), float(g.get("y", 0)),
                              float(g.get("width", 0)), float(g.get("height", 0)))
        elif c.get("vertex") == "1" and c.get("value"):
            txt = _clean_label(c.get("value"))
            if txt:
                labels[c.get("id")] = txt
                g = c.find("mxGeometry")
                if g is not None:
                    label_geoms[c.get("id")] = (
                        float(g.get("x", 0)), float(g.get("y", 0)),
                        float(g.get("width", 0)), float(g.get("height", 0)))

    if image_cell_id is None:
        raise ValueError(f"Sample {sample_id}: no image cell in Mapping.drawio")

    points: list[dict] = []
    relations: list[dict] = []
    warnings: list[str] = []

    def _floating_point_in_image(c: ET.Element, which: str) -> tuple[float, float] | None:
        """Absolute-coordinate fallback (samples ~3-32 use this convention):
        the arrow has no source/target *cell* on the image side — instead a
        floating mxPoint (as="sourcePoint"/"targetPoint") carries absolute
        canvas coordinates. If that point lands inside the image cell's box,
        convert it to the same fraction entryX/entryY would have given."""
        if image_geom is None:
            return None
        for pt in c.iter("mxPoint"):
            if pt.get("as") != which:
                continue
            px, py = float(pt.get("x", "nan")), float(pt.get("y", "nan"))
            ix, iy, iw, ih = image_geom
            if iw > 0 and ih > 0 and ix <= px <= ix + iw and iy <= py <= iy + ih:
                return (px - ix) / iw, (py - iy) / ih
        return None

    def _arrow_dir(label_id: str, fx: float, fy: float,
                   edge: ET.Element | None = None
                   ) -> tuple[float, float] | None:
        """Unit-ish direction of the annotation arrow in IMAGE-FRACTION space
        (arrow origin → arrow endpoint). The arrowheads systematically stop
        short of the part they mean (verified on the 113 render: every
        instance point hovers in empty space just outside the part) — the
        direction lets a consumer march from the endpoint into the first
        foreground of the right colour to recover the intended anchor.

        Origin priority: the segment actually drawn matters, so use the LAST
        waypoint of the edge if it has any, else the exact exit point on the
        label box (exitX/exitY), else the label box centre. Centre-only
        directions were off by enough to miss small parts entirely (rome_1
        ray sailed ~50 px above the rod)."""
        if image_geom is None or label_id not in label_geoms:
            return None
        ix, iy, iw, ih = image_geom
        if iw <= 0 or ih <= 0:
            return None
        lx0, ly0, lw, lh = label_geoms[label_id]
        ox, oy = lx0 + lw / 2, ly0 + lh / 2       # default: box centre
        if edge is not None:
            esd = _style_dict(edge.get("style", ""))
            if "exitX" in esd and "exitY" in esd:
                ox = lx0 + float(esd["exitX"]) * lw
                oy = ly0 + float(esd["exitY"]) * lh
            geom = edge.find("mxGeometry")
            if geom is not None:
                arr = geom.find("Array[@as='points']")
                if arr is not None:
                    pts = arr.findall("mxPoint")
                    if pts:
                        ox = float(pts[-1].get("x", ox))
                        oy = float(pts[-1].get("y", oy))
        dx = (ix + fx * iw - ox) / iw
        dy = (iy + fy * ih - oy) / ih
        if dx == 0 and dy == 0:
            return None
        return dx, dy

    def _add_point(label: str, fx: float, fy: float,
                   coord_src: str = "entry",
                   label_id: str | None = None,
                   edge: ET.Element | None = None) -> None:
        d = _arrow_dir(label_id, fx, fy, edge) if label_id else None
        extra = {"coord_src": coord_src,
                 "dfx": d[0] if d else None, "dfy": d[1] if d else None}
        m_slot = _SLOT_RE.match(label)
        m_inst = _INSTANCE_RE.match(label)
        if m_slot:
            base, slot, slot_idx = m_slot.groups()
            cls = base.rsplit("_", 1)[0]
            points.append({"label": label, "kind": "slot", "base": base,
                           "cls": cls, "slot": slot, "slot_idx": int(slot_idx),
                           "fx": fx, "fy": fy, **extra})
        elif m_inst:
            cls = m_inst.group(1)
            points.append({"label": label, "kind": "instance", "base": label,
                           "cls": cls, "slot": None, "slot_idx": None,
                           "fx": fx, "fy": fy, **extra})
        else:
            warnings.append(f"unrecognized label format: '{label}'")

    for c in cells:
        if c.get("edge") != "1":
            continue
        src, tgt = c.get("source"), c.get("target")
        sd = _style_dict(c.get("style", ""))

        if tgt == image_cell_id or src == image_cell_id:
            label_id = src if tgt == image_cell_id else tgt
            label = labels.get(label_id)
            if label is None:
                warnings.append(f"edge {c.get('id')} → image from unlabeled cell")
                continue
            ex, ey = sd.get("entryX"), sd.get("entryY")
            if ex is not None and ey is not None:
                _add_point(label, float(ex), float(ey), "entry", label_id, c)
                continue
            # No entry fractions — try the absolute endpoint on the image side
            which = "targetPoint" if tgt == image_cell_id else "sourcePoint"
            frac = _floating_point_in_image(c, which)
            if frac is not None:
                _add_point(label, *frac, "abs", label_id, c)
            else:
                warnings.append(f"'{label}': arrow into image has no entryX/entryY")
        elif src in labels and tgt in labels:
            relations.append({"from": labels[src], "to": labels[tgt]})
        elif (label := labels.get(src) or labels.get(tgt)) is not None:
            # One end is a label cell, the other is not the image cell —
            # early-convention arrows point INTO the image purely by absolute
            # coordinates, with no target cell at all.
            label_id = tgt if labels.get(tgt) else src
            which = "sourcePoint" if labels.get(tgt) else "targetPoint"
            frac = _floating_point_in_image(c, which)
            if frac is not None:
                _add_point(label, *frac, "abs", label_id, c)

    # Labels that never got a point (arrow missing / no entry coords)
    pointed = {p["label"] for p in points}
    for lab in labels.values():
        if lab not in pointed:
            warnings.append(f"label '{lab}' has no image annotation")

    return {"sample_id": sample_id,
            "sample_dir": str(sample_dir),
            "points": points,
            "relations": relations,
            "warnings": warnings}


def verify_image_identity(sample_id: int, sample_dir: Path) -> dict:
    """Which multi-view image is Construction.jpg? Checked by pixel diff,
    starting with the Images30/Sample_N_1.jpg convention, then searching."""
    import numpy as np
    from PIL import Image

    cpath = sample_dir / "Construction.jpg"
    if not cpath.exists():
        return {"match": None, "note": "Construction.jpg missing"}
    c = np.asarray(Image.open(cpath).convert("RGB"), dtype=np.int16)

    def diff(p: Path) -> float | None:
        try:
            im = np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
        except Exception:
            return None
        if im.shape != c.shape:
            return None
        return float(np.abs(im - c).mean())

    convention = MULTI_VIEW / f"Sample_{sample_id}" / "Images30" / f"Sample_{sample_id}_1.jpg"
    if convention.exists() and (d := diff(convention)) is not None and d == 0.0:
        return {"match": str(convention), "diff": 0.0, "by_convention": True}

    best = (None, float("inf"))
    for p in sorted((MULTI_VIEW / f"Sample_{sample_id}").glob("**/*.jpg")):
        d = diff(p)
        if d is not None and d < best[1]:
            best = (str(p), d)
    return {"match": best[0], "diff": best[1], "by_convention": False}


def render_seeds(parsed: dict, image_match: dict, out_path: Path) -> None:
    """Draw the parsed seed points on the construction photo for review."""
    import cv2

    img_path = image_match.get("match") or str(Path(parsed["sample_dir"]) / "Construction.jpg")
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    for p in parsed["points"]:
        x, y = int(p["fx"] * W), int(p["fy"] * H)
        colour = (0, 0, 255) if p["kind"] == "instance" else (255, 128, 0)
        cv2.circle(img, (x, y), 10, colour, 2)
        cv2.circle(img, (x, y), 2, colour, -1)
        cv2.putText(img, p["label"], (x + 12, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, p["label"], (x + 12, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("samples", nargs="+", type=int)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--render", action="store_true",
                    help="also write seed_render_N.png for visual review")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for sid in args.samples:
        try:
            parsed = parse_mapping(sid)
        except (FileNotFoundError, ValueError) as e:
            print(f"[{sid}] SKIP: {e}")
            continue
        ident = verify_image_identity(sid, Path(parsed["sample_dir"]))
        parsed["image"] = ident

        out_json = args.out / f"sample_{sid}.json"
        out_json.write_text(json.dumps(parsed, indent=2))

        n_inst = sum(1 for p in parsed["points"] if p["kind"] == "instance")
        n_slot = len(parsed["points"]) - n_inst
        print(f"[{sid}] {n_inst} instance pts, {n_slot} slot pts, "
              f"{len(parsed['relations'])} relations, "
              f"{len(parsed['warnings'])} warnings | "
              f"image: {Path(ident['match']).name if ident.get('match') else '??'} "
              f"(diff={ident.get('diff')}, convention={ident.get('by_convention')})")
        for w in parsed["warnings"]:
            print(f"       warn: {w}")

        if args.render:
            render_seeds(parsed, ident, args.out / f"seed_render_{sid}.png")


if __name__ == "__main__":
    main()
