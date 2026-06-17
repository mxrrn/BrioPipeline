"""Central path and settings configuration for the BRIO 3D pipeline."""
from pathlib import Path

# ── Root paths ─────────────────────────────────────────────────────────────
BA_ROOT        = Path("/mnt/c/BA")
SAM_WEIGHTS    = BA_ROOT / "03-code/sam_weights/sam_vit_b_01ec64.pth"
DATA_ROOT      = BA_ROOT / "02-resources/data"
CONSTRUCTIONS  = DATA_ROOT / "constructions"
MULTI_VIEW     = DATA_ROOT / "multi_view_images"
PROJECT_ROOT   = BA_ROOT / "00-project/brio_pipeline/brio_3d_pipeline"
OUTPUTS_ROOT   = PROJECT_ROOT / "outputs"
LOGS_DIR       = PROJECT_ROOT / "logs"

# Batch folder containing samples 112–150
BATCH_112_150  = CONSTRUCTIONS / "Sample_112_to_150"

# ── Pipeline settings ───────────────────────────────────────────────────────
IMAGE_ELEVATION   = "Images45"      # which elevation ring to use
DUST3R_SIZE       = 512             # image resize for DUSt3R encoder
DUST3R_BATCH      = 1              # pairs per forward pass (keep at 1 for 8 GB VRAM)
DUST3R_NITER      = 100            # global alignment iterations
SAM_MODEL_TYPE    = "vit_b"        # vit_b fits safely in 8 GB alongside DUSt3R outputs
SAM_POINTS_SIDE   = 32             # grid density — 32 needed to hit thin rods (~6 px wide)
SAM_IOU_THRESH    = 0.70           # minimum predicted IoU (thin masks score lower)
SAM_STABILITY     = 0.85           # minimum stability score
SAM_MIN_AREA      = 80             # minimum mask area in pixels (rod ends are small)
SAM_KEEP_FACTOR   = 2              # keep up to KEEP_FACTOR × N masks per view

# ── SAM mask filtering (background / whole-object rejection) ───────────────
SAM_MAX_AREA_FRAC   = 0.50   # drop masks larger than this fraction of the foreground
                             # area (whole-object / scene masks)
SAM_FG_OVERLAP_MIN  = 0.60   # drop masks with less than this fraction of their
                             # pixels inside the foreground (background / table)
SAM_DEDUP_IOU       = 0.80   # masks with pairwise IoU above this are duplicates;
                             # keep the one with higher predicted IoU

# ── DUSt3R point filtering ──────────────────────────────────────────────────
DUST3R_CONF_THRESH  = 2.0    # minimum per-pixel confidence to keep a 3D point
                             # (DUSt3R demo default is 3.0; raise if clouds stay noisy)

AUTO_CROP_PADDING  = 0.20         # padding fraction around the LCC bounding box

DEVICE = "cuda"   # set to "cpu" as fallback if CUDA fails

# ── Component visual classifier ─────────────────────────────────────────────
COMPONENT_DATASET    = Path("/mnt/c/Users/mjmer/OneDrive/Desktop/Work/ISSE/BRIO Constructuion/new_structure_Component")
CLASSIFIER_WEIGHTS   = PROJECT_ROOT / "component_classifier.pth"
CLASSIFIER_CONF_THRESH = 0.65   # minimum confidence to trust a visual prediction

# ── 3D visualisation ─────────────────────────────────────────────────────────
VOXEL_SIZE_BG        = 0.003   # voxel grid size (m) for background scene cloud
VOXEL_SIZE_COMPONENT = 0.003   # voxel grid size (m) for component instance clouds

# ── 3D-overlap instance grouping ─────────────────────────────────────────────
OVERLAP_MIN          = 0.20    # min voxel-overlap ratio (∩ / smaller set) to link two masks
OVERLAP_VOXEL_DIV    = 40      # voxel size = scene bbox diagonal / this divisor

# ── Co-axial rod fragment merging (purely geometric) ─────────────────────────
MAX_ROD_GAP_M        = 0.12    # max gap (m) between co-axial fragments to merge (was 0.08)
COAXIAL_ANGLE_DEG    = 20.0    # max angle (°) between PCA axes to be considered co-axial
COAXIAL_MAX_OFFSET_M = 0.012   # max lateral offset (m) between the two fragment axes
COAXIAL_MIN_ELONG    = 1.8     # min PCA elongation (σ1/σ2) for a cluster to count as a rod (was 2.5)
