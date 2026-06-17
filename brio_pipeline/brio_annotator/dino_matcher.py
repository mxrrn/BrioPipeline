"""
DINOv2 frozen feature matching for component class assignment.

DINOv2 (ViT-S/14, no fine-tuning) produces 384-dim embeddings where
visual texture and shape similarity in pixel space maps to cosine similarity
in embedding space.  For BRIO components this is more discriminative than
mean-HSV: wooden blocks (wood-grain texture), bolts (smooth metal cylinder),
and wheels (rubber + plastic) are well-separated from ImageNet pretraining.

Workflow:
  1. Build a reference library: for each class, embed all reference images
     from component_images/<folder>/Images45/ and store the mean embedding.
  2. At annotation time: embed a mask-isolated crop and rank classes by cosine
     similarity to the reference library.

The reference library is cached to disk so embeddings are computed once.
"""
import cv2
import numpy as np
import pickle
from pathlib import Path
import sys

_3D_DIR = Path(__file__).parent.parent / "brio_3d_pipeline"
if str(_3D_DIR) not in sys.path:
    sys.path.insert(0, str(_3D_DIR))

from component_map import FOLDER_TO_CLASS

BG_THRESHOLD = 245


# ── Mask-isolated crop ────────────────────────────────────────────────────────

def mask_isolated_crop(bgr: np.ndarray, mask: np.ndarray,
                        margin: float = 0.10) -> np.ndarray | None:
    """
    Extract the bounding-box region of a boolean mask with non-mask pixels
    set to neutral grey [128, 128, 128].

    Grey-filling removes adjacent-component contamination so the model sees
    only the target component.  This is the single highest-ROI change for
    classification accuracy over the previous bounding-box-only crop.
    """
    if mask.shape != bgr.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8),
                          (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h_b, w_b = max(y1 - y0, 1), max(x1 - x0, 1)
    py = max(int(h_b * margin), 4)
    px = max(int(w_b * margin), 4)
    H, W = bgr.shape[:2]
    y0c = max(0, y0 - py);  y1c = min(H, y1 + py)
    x0c = max(0, x0 - px);  x1c = min(W, x1 + px)

    crop      = bgr[y0c:y1c, x0c:x1c].copy()
    mask_crop = mask[y0c:y1c, x0c:x1c]
    crop[~mask_crop] = 128   # grey-fill non-mask region

    # Pad to square
    ch, cw = crop.shape[:2]
    side   = max(ch, cw)
    square = np.full((side, side, 3), 128, dtype=np.uint8)
    dy, dx = (side - ch) // 2, (side - cw) // 2
    square[dy:dy + ch, dx:dx + cw] = crop
    return square


# ── DINOv2 matcher ────────────────────────────────────────────────────────────

class DINOMatcher:
    """
    Classify a BGR crop by cosine similarity to a per-class DINOv2 reference library.

    The first instantiation downloads DINOv2 weights via torch.hub (~330 MB,
    cached at ~/.cache/torch/hub/).  Subsequent instantiations use the cache.
    """

    def __init__(self, ref_dir: Path,
                 cache_path: Path | None = None,
                 model_name: str = "dinov2_vits14",
                 device: str = "cuda",
                 elevation: str = "Images45"):
        import torch
        from torchvision import transforms

        self._device = device
        print(f"[DINO] Loading {model_name}...")
        self._model = torch.hub.load("facebookresearch/dinov2", model_name,
                                      verbose=False)
        self._model.eval().to(device)

        self._tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self._ref = self._build_library(ref_dir, cache_path, elevation)
        print(f"[DINO] Reference library: {len(self._ref)} classes")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, bgr: np.ndarray) -> np.ndarray:
        """Return (384,) L2-normalised DINOv2 feature vector."""
        import torch
        from PIL import Image
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self._tf(Image.fromarray(rgb)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._model(tensor)[0].cpu().numpy()
        norm = np.linalg.norm(feat)
        return feat / max(norm, 1e-9)

    def _embed_batch(self, bgrs: list[np.ndarray]) -> np.ndarray:
        """Batch embed a list of BGR images → (N, 384) array."""
        import torch
        from PIL import Image
        tensors = [
            self._tf(Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)))
            for b in bgrs
        ]
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            feats = self._model(batch).cpu().numpy()   # (N, 384)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-9)

    # ── Reference library ─────────────────────────────────────────────────────

    def _build_library(self, ref_dir: Path, cache_path: Path | None,
                        elevation: str) -> dict[str, np.ndarray]:
        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                lib = pickle.load(f)
            print(f"[DINO] Loaded cached library from {cache_path}")
            return lib

        print("[DINO] Building reference library (runs once)...")
        lib: dict[str, np.ndarray] = {}
        for folder, cls in FOLDER_TO_CLASS.items():
            img_dir = ref_dir / folder / elevation
            if not img_dir.exists():
                continue
            paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.JPG"))
            if not paths:
                continue
            bgrs = []
            for p in paths:
                bgr = cv2.imread(str(p))
                if bgr is None:
                    continue
                # Grey-fill background in reference images too
                fg = np.any(bgr < BG_THRESHOLD, axis=2)
                clean = bgr.copy(); clean[~fg] = 128
                bgrs.append(clean)
            if not bgrs:
                continue
            embs = self._embed_batch(bgrs)   # (M, 384)
            mean = embs.mean(axis=0)
            lib[cls] = mean / max(np.linalg.norm(mean), 1e-9)
            print(f"  {cls:<14}  {len(bgrs)} refs")

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(lib, f)
            print(f"[DINO] Library cached to {cache_path}")
        return lib

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, bgr_crop: np.ndarray,
                top_k: int = 5) -> list[tuple[str, float]]:
        """
        Return top-k (class_code, cosine_similarity) pairs for a BGR crop.
        The crop should be mask-isolated (non-mask pixels = grey).
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return []
        emb    = self._embed(bgr_crop)
        scores = {cls: float(np.dot(emb, ref)) for cls, ref in self._ref.items()}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def predict_from_mask(self, bgr: np.ndarray, mask: np.ndarray,
                           top_k: int = 5) -> list[tuple[str, float]]:
        """Isolate the mask crop, then predict."""
        crop = mask_isolated_crop(bgr, mask)
        return self.predict(crop, top_k=top_k) if crop is not None else []

    def predict_batch_from_masks(self, bgr: np.ndarray,
                                  masks: list[np.ndarray],
                                  top_k: int = 5,
                                  ) -> list[list[tuple[str, float]]]:
        """Batch predict for a list of masks on the same image."""
        crops  = [mask_isolated_crop(bgr, m) for m in masks]
        valid_crops  = [(i, c) for i, c in enumerate(crops) if c is not None]
        results: list[list[tuple[str, float]]] = [[] for _ in masks]

        if not valid_crops:
            return results

        idxs, bgr_list = zip(*valid_crops)
        embs = self._embed_batch(list(bgr_list))   # (M, 384)

        classes = list(self._ref.keys())
        ref_mat = np.stack([self._ref[c] for c in classes])   # (C, 384)
        sims    = embs @ ref_mat.T                             # (M, C)

        for m_idx, (orig_idx, _) in enumerate(valid_crops):
            row = sims[m_idx]
            ranked = sorted(zip(classes, row.tolist()), key=lambda x: x[1], reverse=True)
            results[orig_idx] = ranked[:top_k]

        return results
