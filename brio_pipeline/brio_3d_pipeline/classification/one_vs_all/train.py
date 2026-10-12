"""
Train the one-vs-all component classifier on construction-domain crops.

Data source: classification/training/dataset_crops/{cls}/*.jpg (real crops
reprojected from Mapping.drawio seed points into every view of a sample) +
classification/training/dataset_crops/_bg/*.jpg (background negatives).
No isolated-component images, no SAM/Roboflow — see ONE_VS_ALL_PLAN.md
Sections 2 and 9 for why.

Run:
    python classification/one_vs_all/train.py
    python classification/one_vs_all/train.py --epochs 40 --device cuda
"""
import re
import sys
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent.parent  # brio_3d_pipeline/
sys.path.insert(0, str(PIPELINE_ROOT))

from classification.component_map import FOLDER_TO_CLASS  # noqa: E402
from classification.one_vs_all.model import OneVsAllClassifier  # noqa: E402

CLASSES = sorted(set(FOLDER_TO_CLASS.values()))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

DATASET_CROPS = PIPELINE_ROOT / "classification" / "training" / "dataset_crops"
WEIGHTS_DIR = HERE / "weights"

SEED = 42
THIN_CLASS_THRESHOLD = 50   # classes with fewer than this many train crops get heavier augmentation
JUDGEABLE_MIN_VAL = 30      # matches ONE_VS_ALL_PLAN.md Section 7 — classes below this val count
                            # are reported separately, not scored against the F1 target

_SAMPLE_ID_RE = re.compile(r"^s(\d+)_")


# ── Data scanning ────────────────────────────────────────────────────────────

def _sample_id_from_filename(path: Path) -> int | None:
    m = _SAMPLE_ID_RE.match(path.name)
    return int(m.group(1)) if m else None


def scan_dataset(root: Path = DATASET_CROPS) -> list[dict]:
    """
    Scan dataset_crops/{cls}/*.jpg + dataset_crops/_bg/*.jpg into a flat list
    of {path, cls_idx (None for _bg), sample_id}. Anything that isn't a real
    class folder or "_bg" (e.g. the debug_113/debug_131 stray dry-run dumps
    found during planning) is skipped and reported, not silently absorbed.
    """
    records: list[dict] = []
    skipped_dirs = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "_bg":
            cls_idx = None
        elif entry.name in CLASS_TO_IDX:
            cls_idx = CLASS_TO_IDX[entry.name]
        else:
            skipped_dirs.append(entry.name)
            continue
        for p in list(entry.glob("*.jpg")) + list(entry.glob("*.JPG")):
            sid = _sample_id_from_filename(p)
            if sid is None:
                print(f"[Data] WARNING: {p} doesn't match the s{{id}}_... naming "
                      f"convention — skipped (can't assign it to a split)")
                continue
            records.append({"path": p, "cls_idx": cls_idx, "sample_id": sid})

    if skipped_dirs:
        print(f"[Data] Skipped non-class directories: {skipped_dirs}")

    n_bg = sum(1 for r in records if r["cls_idx"] is None)
    print(f"[Data] Scanned {len(records)} crops "
          f"({len(records) - n_bg} across {N_CLASSES} classes, {n_bg} background)")

    counts = Counter(r["cls_idx"] for r in records if r["cls_idx"] is not None)
    zero_classes = [c for c in CLASSES if counts.get(CLASS_TO_IDX[c], 0) == 0]
    if zero_classes:
        print(f"[Data] WARNING: zero training crops for {zero_classes} — "
              f"these heads will never see a positive example. Not an "
              f"imbalance problem, an untrained-model problem; the model "
              f"still gets a head for them (so retraining later doesn't "
              f"need an architecture change), but predictions from these "
              f"heads are meaningless until real data exists.")
    return records


# ── Sample-level split (leak-proof by construction) ─────────────────────────

def split_by_sample(records: list[dict], val_frac: float = 0.15, seed: int = SEED
                    ) -> tuple[list[dict], list[dict]]:
    """
    Every crop from a given sample_id goes entirely into train or entirely
    into val — never split across both (ONE_VS_ALL_PLAN.md Section 5: a
    random crop-level split would let one view of a physical part train the
    model while another view of the SAME instance validates it).

    Which samples become val is NOT random — with only ~34 samples total and
    29 classes unevenly spread across them, a random pick left 16/29 classes
    with zero val examples (found during implementation, not assumed). Val
    samples are instead chosen greedily to maximise class coverage, same
    greedy set-cover idea batch_build.py already uses for data-growth
    ordering, applied here to split selection instead.

    A class that appears in only one sample can never have both train and
    val representation — that sample is excluded from val candidacy
    entirely (protecting its only training source takes priority over
    covering it in val), and it stays "no_val_data" until more real data
    exists for it. Reported explicitly, not silently absorbed into "unlucky
    split" the way a purely random selection would look.
    """
    by_sample: dict[int, list[dict]] = defaultdict(list)
    classes_in_sample: dict[int, set[int]] = defaultdict(set)
    for r in records:
        by_sample[r["sample_id"]].append(r)
        if r["cls_idx"] is not None:
            classes_in_sample[r["sample_id"]].add(r["cls_idx"])

    samples_per_class: dict[int, set[int]] = defaultdict(set)
    for sid, classes in classes_in_sample.items():
        for c in classes:
            samples_per_class[c].add(sid)

    singleton_classes = {c for c, samples in samples_per_class.items() if len(samples) == 1}
    protected_samples = {next(iter(samples_per_class[c])) for c in singleton_classes}

    sample_ids = sorted(by_sample.keys())
    candidates = set(sample_ids) - protected_samples

    covered: set[int] = set()
    val_ids: list[int] = []
    remaining = candidates
    while remaining:
        best = max(sorted(remaining), key=lambda sid: (
            len(classes_in_sample[sid] - covered),   # most newly-covered classes
            -len(by_sample[sid]),                    # tie-break: fewer crops (keep val lean)
        ))
        new_cov = classes_in_sample[best] - covered
        if not new_cov:
            break   # no remaining candidate adds any new class coverage — stop
        val_ids.append(best)
        covered |= classes_in_sample[best]
        remaining = remaining - {best}

    val_ids = set(val_ids)
    train_ids = set(sample_ids) - val_ids

    assert train_ids.isdisjoint(val_ids), "sample-level split leaked — this should be impossible"

    train_records = [r for r in records if r["sample_id"] in train_ids]
    val_records = [r for r in records if r["sample_id"] in val_ids]
    print(f"[Split] {len(train_ids)} samples / {len(train_records)} crops → train, "
          f"{len(val_ids)} samples / {len(val_records)} crops → val "
          f"(assert-verified zero sample overlap, greedy class-coverage selection)")

    if singleton_classes:
        print(f"[Split] Single-sample classes (only ever in train, val-unjudgeable "
              f"until more real data exists): {sorted(CLASSES[c] for c in singleton_classes)}")

    _report_class_coverage(train_records, val_records)
    return train_records, val_records


def _report_class_coverage(train_records: list[dict], val_records: list[dict]) -> None:
    train_counts = Counter(r["cls_idx"] for r in train_records if r["cls_idx"] is not None)
    val_counts = Counter(r["cls_idx"] for r in val_records if r["cls_idx"] is not None)
    no_val = [c for c in CLASSES
              if train_counts.get(CLASS_TO_IDX[c], 0) > 0 and val_counts.get(CLASS_TO_IDX[c], 0) == 0]
    if no_val:
        print(f"[Split] WARNING: classes with train data but ZERO val examples "
              f"after greedy coverage selection (genuinely unrecoverable this "
              f"run, not a split-luck problem): {no_val}")


# ── Dataset / augmentation ───────────────────────────────────────────────────

def _build_transforms():
    from torchvision import transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        normalize,
    ])
    # Heavier augmentation for thin classes: manufactures visual variety
    # from few real examples (ONE_VS_ALL_PLAN.md Section 4) — doesn't
    # manufacture real data, just squeezes more out of what exists.
    heavy_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, heavy_tf, val_tf


class CropDataset:
    def __init__(self, records: list[dict], thin_classes: set[int],
                train: bool, train_tf, heavy_tf, val_tf):
        self.records = records
        self.thin_classes = thin_classes
        self.train = train
        self.train_tf, self.heavy_tf, self.val_tf = train_tf, heavy_tf, val_tf

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        from PIL import Image
        rec = self.records[i]
        bgr = cv2.imread(str(rec["path"]))
        if bgr is None:
            bgr = np.full((224, 224, 3), 255, np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        cls_idx = rec["cls_idx"]
        if self.train:
            tf = self.heavy_tf if (cls_idx is not None and cls_idx in self.thin_classes) \
                 else self.train_tf
        else:
            tf = self.val_tf
        tensor = tf(img)

        target = np.zeros(N_CLASSES, dtype=np.float32)
        if cls_idx is not None:
            target[cls_idx] = 1.0
        # cls_idx_for_metrics: -1 sentinel for _bg, used to bucket per-class
        # metrics and the _bg false-positive check at eval time
        return tensor, target, (cls_idx if cls_idx is not None else -1)


# ── Class imbalance: per-class pos_weight + balanced sampling ───────────────

def compute_pos_weight(train_records: list[dict]) -> "torch.Tensor":
    """
    BCEWithLogitsLoss pos_weight per class = n_negative / n_positive (train
    split only — never computed from val, that would leak split information
    into training). A class with zero train positives gets weight 1.0 (moot,
    it never receives a positive gradient either way).
    """
    import torch
    n_total = len(train_records)
    pos_counts = Counter(r["cls_idx"] for r in train_records if r["cls_idx"] is not None)
    weights = torch.ones(N_CLASSES)
    for c in range(N_CLASSES):
        n_pos = pos_counts.get(c, 0)
        if n_pos > 0:
            weights[c] = (n_total - n_pos) / n_pos
    print("[Imbalance] Per-class pos_weight (n_negative/n_positive, train split):")
    for c in range(N_CLASSES):
        n_pos = pos_counts.get(c, 0)
        flag = " (UNTRAINED — 0 positives)" if n_pos == 0 else ""
        print(f"    {CLASSES[c]:<10} n_pos={n_pos:<5} weight={weights[c]:.1f}{flag}")
    return weights


def build_sampler(train_records: list[dict]) -> "torch.utils.data.WeightedRandomSampler":
    """
    Class-balanced batch sampling: weight = 1/count for each crop's class
    (with _bg treated as one more class for this purpose), so every class —
    including _bg — is drawn with roughly equal frequency per epoch,
    independent of how many crops it actually has on disk.
    """
    import torch
    from torch.utils.data import WeightedRandomSampler
    counts = Counter(
        r["cls_idx"] if r["cls_idx"] is not None else "_bg" for r in train_records
    )
    weights = [
        1.0 / counts[r["cls_idx"] if r["cls_idx"] is not None else "_bg"]
        for r in train_records
    ]
    return WeightedRandomSampler(
        torch.DoubleTensor(weights), num_samples=len(train_records), replacement=True
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(epochs: int = 40, batch_size: int = 32, lr: float = 3e-4,
         device: str = "cuda", out_path: Path | None = None) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    # Reproducibility: matches the determinism standard already established
    # elsewhere in this pipeline (see backprojector.py's _extract_mask_cloud
    # seed fix). num_workers=0 to avoid separate worker-process RNG streams.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    records = scan_dataset()
    train_records, val_records = split_by_sample(records)

    train_counts = Counter(r["cls_idx"] for r in train_records if r["cls_idx"] is not None)
    thin_classes = {c for c in range(N_CLASSES) if train_counts.get(c, 0) < THIN_CLASS_THRESHOLD}
    print(f"[Imbalance] Thin classes (<{THIN_CLASS_THRESHOLD} train crops, heavier "
          f"augmentation): {[CLASSES[c] for c in sorted(thin_classes)]}")

    train_tf, heavy_tf, val_tf = _build_transforms()
    train_set = CropDataset(train_records, thin_classes, True, train_tf, heavy_tf, val_tf)
    val_set = CropDataset(val_records, thin_classes, False, train_tf, heavy_tf, val_tf)

    sampler = build_sampler(train_records)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    pos_weight = compute_pos_weight(train_records).to(device)

    model = OneVsAllClassifier(CLASSES, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_n = 0.0, 0
        for imgs, targets, _ in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(imgs)
            t_n += len(imgs)
        scheduler.step()

        model.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                logits = model(imgs)
                v_loss += criterion(logits, targets).item() * len(imgs)
                v_n += len(imgs)

        train_loss, val_loss = t_loss / max(t_n, 1), v_loss / max(v_n, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:>3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    thresholds, report = evaluate(model, val_loader, device)

    out_path = out_path or (WEIGHTS_DIR / "one_vs_all.pth")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "n_cls": N_CLASSES,
        "thresholds": thresholds,   # per-class calibrated decision thresholds
        "pos_weight": pos_weight.cpu().tolist(),
        "history": history,
    }, out_path)
    print(f"[Train] Saved → {out_path}")

    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[Train] Per-class report → {report_path}")

    return report


# ── Evaluation: per-class metrics, threshold calibration, _bg check ─────────

def evaluate(model, val_loader, device: str) -> tuple[dict, dict]:
    """
    Runs the val set once, then for each class sweeps thresholds 0.05-0.95
    to find the one maximizing F1 (per-head threshold calibration —
    ONE_VS_ALL_PLAN.md Section 4). Reports precision/recall/F1 at the
    calibrated threshold, the _bg false-positive rate, and a confidence-
    separation summary (mean confidence of correct vs incorrect predictions).
    """
    import torch

    model.eval()
    all_probs, all_targets, all_cls_idx = [], [], []
    with torch.no_grad():
        for imgs, targets, cls_idx in val_loader:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_cls_idx.extend(cls_idx.tolist())
    probs = np.concatenate(all_probs, axis=0)      # (N, K)
    targets = np.concatenate(all_targets, axis=0)  # (N, K)
    cls_idx = np.array(all_cls_idx)                # (N,), -1 = _bg

    thresholds: dict[str, float] = {}
    per_class: dict[str, dict] = {}
    val_counts = Counter(cls_idx[cls_idx >= 0].tolist())

    for c in range(N_CLASSES):
        y_true = targets[:, c]
        n_val = int(val_counts.get(c, 0))
        if n_val == 0:
            per_class[CLASSES[c]] = {"status": "no_val_data", "n_val": 0}
            thresholds[CLASSES[c]] = 0.5
            continue

        best_f1, best_t = -1.0, 0.5
        for t in np.arange(0.05, 1.0, 0.05):
            y_pred = (probs[:, c] >= t).astype(np.float32)
            tp = float(((y_pred == 1) & (y_true == 1)).sum())
            fp = float(((y_pred == 1) & (y_true == 0)).sum())
            fn = float(((y_pred == 0) & (y_true == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

        y_pred = (probs[:, c] >= best_t).astype(np.float32)
        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        correct_conf = probs[(y_true == 1) & (y_pred == 1), c]
        wrong_conf = probs[(y_true == 0) & (y_pred == 1), c]

        thresholds[CLASSES[c]] = best_t
        per_class[CLASSES[c]] = {
            "status": "judgeable" if n_val >= JUDGEABLE_MIN_VAL else "insufficient_val_data",
            "n_val": n_val,
            "threshold": best_t,
            "precision": prec, "recall": rec, "f1": best_f1,
            "mean_conf_true_positive": float(correct_conf.mean()) if len(correct_conf) else None,
            "mean_conf_false_positive": float(wrong_conf.mean()) if len(wrong_conf) else None,
        }

    # _bg false-positive rate: fraction of true-background val crops where
    # ANY head fires above its calibrated threshold — the direct test of
    # whether the reject mechanism (the whole point of one-vs-all) works.
    bg_mask = cls_idx == -1
    n_bg = int(bg_mask.sum())
    if n_bg > 0:
        thresh_vec = np.array([thresholds[CLASSES[c]] for c in range(N_CLASSES)])
        any_fire = (probs[bg_mask] >= thresh_vec[None, :]).any(axis=1)
        bg_fp_rate = float(any_fire.mean())
    else:
        bg_fp_rate = None

    print("\n[Eval] Per-class report (calibrated thresholds):")
    for c in CLASSES:
        r = per_class[c]
        if r["status"] in ("no_val_data", "insufficient_val_data"):
            print(f"    {c:<10} {r['status']:<22} n_val={r['n_val']}")
        else:
            print(f"    {c:<10} n_val={r['n_val']:<5} thresh={r['threshold']:.2f}  "
                  f"P={r['precision']:.2f}  R={r['recall']:.2f}  F1={r['f1']:.2f}")
    print(f"    {'_bg':<10} n_val={n_bg:<5} false_positive_rate="
          f"{bg_fp_rate:.3f}" if bg_fp_rate is not None else "no _bg val data")

    report = {
        "per_class": per_class,
        "bg_false_positive_rate": bg_fp_rate,
        "n_bg_val": n_bg,
        "judgeable_min_val": JUDGEABLE_MIN_VAL,
    }
    return thresholds, report


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr,
         device=args.device, out_path=args.out)
