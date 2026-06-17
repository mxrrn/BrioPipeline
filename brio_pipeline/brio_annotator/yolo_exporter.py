"""
Export annotations to YOLO format.

YOLO label file format (one line per object):
  class_idx  cx  cy  w  h      (cx/cy/w/h normalised to [0, 1])

One .txt label file is written per image, with the same stem as the image.
A dataset.yaml is written once at run level for YOLOv8 training.
"""
from pathlib import Path
import yaml

from amodal_bbox import Annotation


def bbox_to_yolo(bbox_xyxy: tuple[int, int, int, int],
                  img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox_xyxy
    cx = ((x0 + x1) / 2) / img_w
    cy = ((y0 + y1) / 2) / img_h
    w  = (x1 - x0) / img_w
    h  = (y1 - y0) / img_h
    return cx, cy, w, h


def export_labels(annotations  : list[Annotation],
                   class_to_idx : dict[str, int],
                   img_w        : int,
                   img_h        : int,
                   output_path  : Path,
                   ) -> int:
    """
    Write a YOLO label file for one image.  Returns number of lines written.
    Skips annotations with zero-size bboxes silently.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for ann in annotations:
        x0, y0, x1, y1 = ann.bbox_xyxy
        if x1 <= x0 or y1 <= y0:
            continue
        cls_idx = class_to_idx.get(ann.cls)
        if cls_idx is None:
            print(f"  [YOLO] Unknown class '{ann.cls}' — skipping {ann.instance_id}")
            continue
        cx, cy, w, h = bbox_to_yolo(ann.bbox_xyxy, img_w, img_h)
        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    output_path.write_text("\n".join(lines))
    return len(lines)


def export_dataset_yaml(class_names: list[str],
                          train_img_dir: Path,
                          val_img_dir  : Path | None,
                          output_path  : Path,
                          ) -> None:
    """Write dataset.yaml for YOLOv8 training."""
    data = {
        "path" : str(output_path.parent),
        "train": str(train_img_dir.relative_to(output_path.parent)),
        "val"  : str((val_img_dir or train_img_dir).relative_to(output_path.parent)),
        "nc"   : len(class_names),
        "names": class_names,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"[YOLO] dataset.yaml → {output_path}")
