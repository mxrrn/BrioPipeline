"""
YOLOv8 detection wrapper.

Loads the trained model once and exposes a per-image inference method
that returns raw detections in the format expected by triangulator.py.

Detection format per image:
    [(cls_name, xc_px, yc_px, w_px, h_px, confidence), ...]
where xc, yc, w, h are pixel coordinates in the original image.
"""
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


class Detector:
    def __init__(self, weights_path: Path | str):
        """
        Args:
            weights_path : path to trained YOLOv8 .pt file
                           (e.g. runs/detect/train/weights/best.pt)
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required: pip install ultralytics"
            )
        self.model = YOLO(str(weights_path))
        print(f"[Detector] Loaded weights from {weights_path}")

    def predict(self, image_bgr: np.ndarray,
                conf: float = cfg.CONF_THRESHOLD
                ) -> list[tuple[str, float, float, float, float, float]]:
        """
        Run inference on one BGR image.

        Returns list of (cls, xc, yc, w, h, conf) in pixel coordinates.
        """
        results = self.model.predict(
            image_bgr, conf=conf, verbose=False, device="cuda"
        )
        dets = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_idx = int(box.cls[0])
                cls     = cfg.IDX_TO_CLASS.get(cls_idx, "unknown")
                xc, yc, w, h = box.xywh[0].tolist()
                confidence    = float(box.conf[0])
                dets.append((cls, xc, yc, w, h, confidence))
        return dets

    def predict_batch(
        self,
        image_paths: list[Path],
        conf: float = cfg.CONF_THRESHOLD,
    ) -> dict[int, list[tuple[str, float, float, float, float, float]]]:
        """
        Run inference on a list of image paths.

        Returns {view_idx: [(cls, xc, yc, w, h, conf), ...]}
        """
        detections: dict[int, list] = {}
        for idx, path in enumerate(image_paths):
            img = cv2.imread(str(path))
            if img is None:
                detections[idx] = []
                continue
            detections[idx] = self.predict(img, conf=conf)
            n = len(detections[idx])
            print(f"[Detector]   view {idx:2d}: {n} detections")
        return detections
