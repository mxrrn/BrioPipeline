"""
Build a fixed camera rig model from one or more slow-pipeline DUSt3R runs.

Why this works
──────────────
The BRIO multi-view capture setup uses a fixed camera and a turntable.
The camera-to-construction geometry is the same across all 150 samples —
only the object on the turntable changes.  DUSt3R estimates camera poses
in an arbitrary world frame for each run, but the RELATIVE poses between
cameras are stable.

Calibration procedure
─────────────────────
1. Load DUSt3R cache(s) from reference samples.
2. For each run, normalise poses so camera 0 = identity (world origin).
3. Average the relative poses across runs.
4. Store the averaged intrinsics and normalised poses.

At inference, these fixed poses are applied directly without running DUSt3R.
"""
import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from logger import setup_logging


# ── Maths helpers ─────────────────────────────────────────────────────────

def _normalise_poses(poses: np.ndarray) -> np.ndarray:
    """
    Normalise a (N, 4, 4) camera-to-world array so pose[0] = I.
    All poses are expressed relative to the first camera.
    """
    P0_inv = np.linalg.inv(poses[0])
    return np.array([P0_inv @ P for P in poses], dtype=np.float64)


def _average_poses(pose_list: list[np.ndarray]) -> np.ndarray:
    """
    Average a list of (N, 4, 4) normalised pose arrays.
    Rotation is averaged via the mean of the rotation matrices
    (close enough for small inter-run variations; proper SO(3) averaging
    would use Procrustes, but this is fine for a fixed rig).
    """
    stacked = np.stack(pose_list, axis=0)  # (R, N, 4, 4)
    return stacked.mean(axis=0)             # (N, 4, 4)


def _average_intrinsics(K_list: list[np.ndarray]) -> np.ndarray:
    """Average a list of (N, 3, 3) intrinsic arrays across runs."""
    stacked = np.stack(K_list, axis=0)   # (R, N, 3, 3)
    return stacked.mean(axis=0)          # (N, 3, 3)


# ── Public API ────────────────────────────────────────────────────────────

def build_rig(sample_ids: list[int] | None = None,
              save_path: Path | None = None) -> dict:
    """
    Build the fixed camera rig model from DUSt3R caches.

    Args:
        sample_ids : reference sample IDs; defaults to cfg.CALIBRATION_SAMPLES
        save_path  : where to save the pkl; defaults to cfg.CALIBRATION_FILE

    Returns dict with keys:
        poses      : (N, 4, 4) averaged normalised camera-to-world poses
        intrinsics : (N, 3, 3) averaged estimated intrinsics
        n_cameras  : N
        source_samples : list of sample IDs used
    """
    if sample_ids is None:
        sample_ids = cfg.CALIBRATION_SAMPLES
    if save_path is None:
        save_path = cfg.CALIBRATION_FILE

    all_poses  = []
    all_K      = []

    for sid in sample_ids:
        cache = cfg.SLOW_OUTPUTS / f"sample_{sid}" / "dust3r" / "dust3r_cache.pkl"
        if not cache.exists():
            print(f"[Calibrate] Sample {sid}: no DUSt3R cache — skipping")
            continue

        with open(cache, "rb") as f:
            result = pickle.load(f)

        poses = result["poses"].astype(np.float64)       # (N, 4, 4)
        K     = result["intrinsics"].astype(np.float64)  # (N, 3, 3)

        norm  = _normalise_poses(poses)
        all_poses.append(norm)
        all_K.append(K)
        print(f"[Calibrate] Loaded {len(poses)} poses from sample {sid}")

    if not all_poses:
        raise RuntimeError("No DUSt3R caches found for calibration samples.")

    # Align all runs to the same camera count (use minimum)
    n_cams = min(p.shape[0] for p in all_poses)
    all_poses = [p[:n_cams] for p in all_poses]
    all_K     = [k[:n_cams] for k in all_K]

    avg_poses = _average_poses(all_poses)
    avg_K     = _average_intrinsics(all_K)

    rig = {
        "poses":          avg_poses,
        "intrinsics":     avg_K,
        "n_cameras":      n_cams,
        "source_samples": sample_ids,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(rig, f)

    print(f"[Calibrate] Rig saved: {n_cams} cameras, {len(all_poses)} reference runs")
    print(f"[Calibrate] → {save_path}")
    return rig


def load_rig(path: Path | None = None) -> dict:
    """Load the saved rig model. Raises FileNotFoundError if not built yet."""
    if path is None:
        path = cfg.CALIBRATION_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Rig calibration not found at {path}. "
            "Run: python calibrator.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build fixed camera rig calibration")
    parser.add_argument("--samples", nargs="+", type=int,
                        default=cfg.CALIBRATION_SAMPLES,
                        help="Reference sample IDs with DUSt3R caches")
    args = parser.parse_args()

    setup_logging("calibrate")
    rig = build_rig(sample_ids=args.samples)
    print(f"\nRig model summary:")
    print(f"  Cameras : {rig['n_cameras']}")
    print(f"  Sources : {rig['source_samples']}")
    print(f"  Pose[0] :\n{rig['poses'][0]}")
    print(f"  K[0]    :\n{rig['intrinsics'][0]}")
