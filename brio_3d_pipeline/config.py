"""Central path and settings configuration for the BRIO 3D pipeline."""
from pathlib import Path

# ── Root paths ─────────────────────────────────────────────────────────────
BA_ROOT        = Path("/mnt/c/BA")
DUST3R_REPO    = BA_ROOT / "07-dust3r"
SAM_WEIGHTS    = BA_ROOT / "03-code/sam_weights/sam_vit_b_01ec64.pth"
DATA_ROOT      = BA_ROOT / "02-resources/data"
CONSTRUCTIONS  = DATA_ROOT / "constructions"
MULTI_VIEW     = DATA_ROOT / "multi_view_images"
PROJECT_ROOT   = BA_ROOT / "00-project/brio_3d_pipeline"
OUTPUTS_ROOT   = PROJECT_ROOT / "outputs"
LOGS_DIR       = PROJECT_ROOT / "logs"

# Batch folder containing samples 112–150
BATCH_112_150  = CONSTRUCTIONS / "Sample_112_to_150"

# ── Pipeline settings ───────────────────────────────────────────────────────
IMAGE_ELEVATION   = "Images45"      # which elevation ring to use
DUST3R_SIZE       = 512             # image resize for DUSt3R encoder
DUST3R_BATCH      = 1              # pairs per forward pass (keep at 1 for 8 GB VRAM)
DUST3R_NITER      = 300            # global alignment iterations
SAM_MODEL_TYPE    = "vit_b"        # vit_b fits safely in 8 GB alongside DUSt3R outputs
SAM_POINTS_SIDE   = 16             # automatic mask generator grid density
SAM_IOU_THRESH    = 0.80           # minimum predicted IoU to keep a mask
SAM_STABILITY     = 0.90           # minimum stability score
SAM_MIN_AREA      = 200            # minimum mask area in pixels

DUST3R_SCENE_GRAPH = "complete"   # all N*(N-1)/2 pairs
AUTO_CROP_PADDING  = 0.20         # padding fraction around the LCC bounding box

DEVICE = "cuda"   # set to "cpu" as fallback if CUDA fails
