"""
MobileNetV3-small visual classifier trained on isolated component images.

Train:
    python component_classifier.py --train
    (or use train_classifier.sh)

Inference:
    clf = ComponentClassifier(weights_path, device)
    cls_code, confidence = clf.predict(bgr_crop)
"""
import numpy as np
import cv2
from pathlib import Path
# ── Folder name → PUML class code (single source of truth in component_map) ──
from component_map import FOLDER_TO_CLASS as FOLDER_TO_CODE

CLASSES = sorted(set(FOLDER_TO_CODE.values()))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ── Crop utility ──────────────────────────────────────────────────────────────

def crop_from_mask(bgr: np.ndarray, mask: np.ndarray,
                   margin: float = 0.15) -> np.ndarray | None:
    """
    Extract the bounding-box region under a boolean mask, pad to square.
    Returns None if the mask is empty.
    """
    if mask.shape != bgr.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (bgr.shape[1], bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h_bbox, w_bbox = max(y1 - y0, 1), max(x1 - x0, 1)
    py = max(int(h_bbox * margin), 4)
    px = max(int(w_bbox * margin), 4)
    H, W = bgr.shape[:2]
    y0 = max(0, y0 - py);  y1 = min(H, y1 + py)
    x0 = max(0, x0 - px);  x1 = min(W, x1 + px)
    crop = bgr[y0:y1, x0:x1].copy()
    # Grey-fill pixels outside the mask so the classifier sees only this component
    mask_crop = mask[y0:y1, x0:x1]
    crop[~mask_crop] = 128
    # Pad to square with white background
    ch, cw = crop.shape[:2]
    if ch == cw:
        return crop
    side = max(ch, cw)
    square = np.full((side, side, 3), 255, dtype=np.uint8)
    dy, dx = (side - ch) // 2, (side - cw) // 2
    square[dy:dy + ch, dx:dx + cw] = crop
    return square


# ── Dataset ───────────────────────────────────────────────────────────────────

class _ComponentDataset:
    """Loads all component images from the new_structure_Component directory."""

    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for folder_name, code in FOLDER_TO_CODE.items():
            cls_dir = root / folder_name
            if not cls_dir.exists():
                continue
            idx = CLASS_TO_IDX[code]
            # Root-level images (e.g. nut_1.jpg)
            search_dirs = [cls_dir]
            # Elevation sub-folders
            for elev in ("Images30", "Images45", "Images60", "Images90"):
                sub = cls_dir / elev
                if sub.exists():
                    search_dirs.append(sub)
            for d in search_dirs:
                for p in list(d.glob("*.jpg")) + list(d.glob("*.JPG")) + list(d.glob("*.png")):
                    self.samples.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        bgr = cv2.imread(str(path))
        if bgr is None:
            bgr = np.full((224, 224, 3), 200, np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            from PIL import Image
            rgb = self.transform(Image.fromarray(rgb))
        return rgb, label


# ── Training ──────────────────────────────────────────────────────────────────

def train_classifier(dataset_path: Path, output_path: Path,
                     epochs: int = 40, batch_size: int = 32,
                     lr: float = 3e-4, device: str = "cuda") -> None:
    """
    Fine-tune MobileNetV3-small (ImageNet init) on component images.
    Saves the best checkpoint (by val accuracy) to output_path.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms, models

    n_cls = len(CLASSES)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_ds = _ComponentDataset(dataset_path, transform=train_tf)
    n_total = len(full_ds)
    print(f"[Classifier] Dataset: {n_total} images, {n_cls} classes")

    # Stratified-ish split: 85% train, 15% val
    torch.manual_seed(42)
    indices = torch.randperm(n_total).tolist()
    n_val   = max(n_cls, int(n_total * 0.15))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_set = Subset(full_ds, train_idx)

    # Val set uses val transform — rebuild dataset with val_tf
    val_ds_raw  = _ComponentDataset(dataset_path, transform=val_tf)
    val_set     = Subset(val_ds_raw, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # MobileNetV3-small pretrained on ImageNet
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_cls)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * len(imgs)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += len(imgs)
        scheduler.step()

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                v_correct += (out := model(imgs)).argmax(1).eq(labels).sum().item()
                v_total   += len(imgs)
        val_acc = v_correct / max(v_total, 1)

        print(f"Epoch {epoch:>3d}/{epochs}  "
              f"loss={t_loss/t_total:.4f}  "
              f"train_acc={t_correct/t_total:.3f}  "
              f"val_acc={val_acc:.3f}"
              + ("  *" if val_acc >= best_acc else ""))

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "model_state":  model.state_dict(),
                "classes":      CLASSES,
                "class_to_idx": CLASS_TO_IDX,
                "n_cls":        n_cls,
            }, output_path)

    print(f"[Classifier] Training done. Best val_acc={best_acc:.3f}  → {output_path}")


# ── Inference ─────────────────────────────────────────────────────────────────

class ComponentClassifier:
    """
    Load a trained MobileNetV3-small checkpoint and classify BGR crops.

    Usage:
        clf = ComponentClassifier(weights_path, device="cuda")
        cls_code, conf = clf.predict(bgr_crop)
    """

    def __init__(self, weights_path: Path, device: str = "cuda"):
        import torch
        import torch.nn as nn
        from torchvision import transforms, models

        ckpt   = torch.load(str(weights_path), map_location=device, weights_only=True)
        n_cls  = ckpt["n_cls"]
        self._classes = ckpt["classes"]

        model = models.mobilenet_v3_small()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_cls)
        model.load_state_dict(ckpt["model_state"])
        model.eval().to(device)

        self._model  = model
        self._device = device
        self._tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, bgr_crop: np.ndarray) -> tuple[str, float]:
        """Return (cls_code, confidence) for a single BGR crop."""
        import torch
        from PIL import Image
        rgb    = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
        tensor = self._tf(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._model(tensor), dim=1)[0]
            idx   = int(probs.argmax())
        return self._classes[idx], float(probs[idx])

    def predict_batch(self, bgr_crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Batch predict for a list of BGR crops in a single GPU forward pass."""
        import torch
        from PIL import Image
        tensors = [
            self._tf(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            for bgr in bgr_crops
        ]
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._model(batch), dim=1)
            idxs  = probs.argmax(dim=1)
        return [
            (self._classes[int(i)], float(probs[j, int(i)]))
            for j, i in enumerate(idxs)
        ]

    def predict_top_k(self, bgr_crop: np.ndarray, k: int = 3
                      ) -> list[tuple[str, float]]:
        """Return top-k (cls_code, confidence) pairs, sorted by confidence."""
        import torch
        from PIL import Image
        rgb    = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
        tensor = self._tf(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._model(tensor), dim=1)[0]
        top_vals, top_idx = probs.topk(k)
        return [(self._classes[int(i)], float(v)) for v, i in zip(top_vals, top_idx)]


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parent))
    from config import COMPONENT_DATASET, CLASSIFIER_WEIGHTS, DEVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--epochs",  type=int,   default=40)
    parser.add_argument("--batch",   type=int,   default=32)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--device",  default=DEVICE)
    parser.add_argument("--dataset", type=Path,  default=COMPONENT_DATASET)
    parser.add_argument("--out",     type=Path,  default=CLASSIFIER_WEIGHTS)
    args = parser.parse_args()

    if args.train:
        train_classifier(args.dataset, args.out,
                         epochs=args.epochs, batch_size=args.batch,
                         lr=args.lr, device=args.device)
    else:
        if not args.out.exists():
            print(f"No weights found at {args.out}. Run with --train first.")
            sys.exit(1)
        clf = ComponentClassifier(args.out, device=args.device)
        print("Classifier loaded. Classes:", clf._classes)
