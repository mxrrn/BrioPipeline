"""
Run DUSt3R on a set of multi-view images and return camera poses, intrinsics,
and per-frame depth maps. Results are cached to disk so DUSt3R only runs once
per sample.

DUSt3R repo must be installed into the active conda env:
    pip install -e /mnt/c/BA/07-dust3r
"""
import sys
import pickle
import numpy as np
from pathlib import Path

# DUSt3R repo must be on the Python path
sys.path.insert(0, str(Path("/mnt/c/BA/07-dust3r")))


def run_dust3r(image_paths: list[Path], output_dir: Path, device: str = "cuda",
               size: int = 512, niter: int = 300, batch_size: int = 1) -> dict:
    """
    Run DUSt3R global alignment on a list of images.

    Returns dict with keys:
        poses      : np.ndarray (N, 4, 4)  camera-to-world
        intrinsics : np.ndarray (N, 3, 3)  camera intrinsics K
        depths     : list[np.ndarray]       per-image depth map (H, W)
        pts3d      : list[np.ndarray]       per-image 3D points (H, W, 3) world space
        confs      : list[np.ndarray]       per-image confidence map (H, W)
        image_paths: list[str]              ordered image paths matching above arrays
    """
    cache_file = output_dir / "dust3r_cache.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        if "confs" in cached:
            print(f"[DUSt3R] Loading cached results from {cache_file}")
            return cached
        print(f"[DUSt3R] Cache at {cache_file} predates confidence maps — regenerating")

    print(f"[DUSt3R] Running on {len(image_paths)} images (device={device})")
    output_dir.mkdir(parents=True, exist_ok=True)

    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    model = AsymmetricCroCo3DStereo.from_pretrained(
        "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)
    model.eval()

    str_paths = [str(p) for p in image_paths]
    images = load_images(str_paths, size=size)

    # logwin scene graph balances coverage vs compute for ~24 images
    pairs = make_pairs(images, scene_graph="logwin", prefilter=None, symmetrize=True)
    print(f"[DUSt3R] {len(pairs)} image pairs, running pairwise inference...")

    output = inference(pairs, model, device, batch_size=batch_size, verbose=False)

    print("[DUSt3R] Running global alignment...")
    scene = global_aligner(
        output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer
    )
    scene.compute_global_alignment(
        init="mst", niter=niter, schedule="cosine", lr=0.01
    )

    poses      = scene.get_im_poses().detach().cpu().numpy()      # (N, 4, 4)
    intrinsics = scene.get_intrinsics().detach().cpu().numpy()    # (N, 3, 3)
    depths     = [d.detach().cpu().numpy() for d in scene.get_depthmaps()]
    pts3d      = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
    confs      = [c.detach().cpu().numpy() for c in scene.im_conf]

    # Free DUSt3R GPU memory before SAM runs in the same process — otherwise
    # SAM thrashes at the 8 GB VRAM limit (observed: 20+ min hangs).
    import torch
    del scene, output, model
    torch.cuda.empty_cache()

    result = {
        "poses":       poses,
        "intrinsics":  intrinsics,
        "depths":      depths,
        "pts3d":       pts3d,
        "confs":       confs,
        "image_paths": str_paths,
    }

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"[DUSt3R] Cached results to {cache_file}")

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import MULTI_VIEW, IMAGE_ELEVATION, DEVICE, DUST3R_SIZE, DUST3R_NITER
    from pathlib import Path

    sample_id = int(sys.argv[1]) if len(sys.argv) > 1 else 113
    img_dir   = MULTI_VIEW / f"Sample_{sample_id}" / IMAGE_ELEVATION
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.JPG"))
    out_dir   = Path(f"outputs/sample_{sample_id}/dust3r")

    result = run_dust3r(img_paths, out_dir, device=DEVICE, size=DUST3R_SIZE, niter=DUST3R_NITER)
    print(f"Poses shape:      {result['poses'].shape}")
    print(f"Intrinsics shape: {result['intrinsics'].shape}")
    print(f"Depth maps:       {len(result['depths'])} frames")
