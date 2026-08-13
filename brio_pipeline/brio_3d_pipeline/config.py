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
SAM_POINTS_SIDE   = 32             # grid density — only used by mode="auto" (SamAutomaticMaskGenerator);
                                   # the default mode="prompted" path uses SAM_GRID_SPACING below
SAM_IOU_THRESH    = 0.70           # minimum predicted IoU — only used by mode="auto"
SAM_STABILITY     = 0.85           # minimum stability score — only used by mode="auto"
SAM_MIN_AREA      = 50             # minimum mask/remainder area in pixels (was 80) — lowered so small
                                   # leftover fragments survive container/oversized decomposition
                                   # instead of forcing the whole merged blob to be kept
SAM_KEEP_FACTOR   = 2              # keep up to KEEP_FACTOR × N masks per view

# ── SAM prompted-mode grid spacing (mode="prompted", the default) ──────────
# Fixed pixel spacing between grid points — NOT derived from n_components or
# foreground area. The previous formula (density × n_components points spread
# over the whole foreground) meant a small nut/screw covering under 1% of the
# foreground had under a 1% chance per point of ever being prompted at all —
# a lottery, not a guarantee, no matter how "dense" density was set. 
# A fixed small spacing guarantees no object bigger than SAM_GRID_SPACING px in both
# dimensions can fall entirely between grid points. 15px may be  small enough to
# reliably catch nut/screw-sized fasteners at this crop resolution; going much
# smaller sharply increases SAM decoder calls and downstream cloud count.
SAM_GRID_SPACING  = 15

# Edge-aware distance-transform seeds (supplements the uniform grid).
# Canny edges are subtracted from the foreground, splitting it into regions
# roughly matching individual visible part surfaces; each region's distance-
# transform maximum (its centre / medial axis) becomes an extra prompt. So
# every visibly-bounded part gets a prompt at its own centre regardless of
# how the uniform grid aligns with it — including fasteners sitting ON
# other parts, which contribute nothing to the silhouette and therefore
# get nothing from a plain foreground distance transform.
SAM_DT_SEEDS      = True

# ── SAM mask filtering (background / whole-object rejection) ───────────────
SAM_MAX_AREA_FRAC   = 0.35   # flag masks above this fraction of the foreground area as
                             # decomposition candidates (was 0.50) — catches more merged
                             # blobs for the container/oversized decomposition passes
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
CLASSIFIER_WEIGHTS   = PROJECT_ROOT / "classification" / "component_classifier.pth"
CLASSIFIER_CONF_THRESH = 0.65   # minimum confidence to trust a visual prediction

# ── 3D visualisation ─────────────────────────────────────────────────────────
VOXEL_SIZE_BG        = 0.003   # voxel grid size (m) for background scene cloud
VOXEL_SIZE_COMPONENT = 0.003   # voxel grid size (m) for component instance clouds

# ── 3D-overlap instance grouping ─────────────────────────────────────────────
OVERLAP_MIN          = 0.15    # min voxel-overlap ratio (∩ / smaller set) to link two masks
OVERLAP_VOXEL_DIV    = 80      # voxel size = scene bbox diagonal / this divisor

# ── Co-axial rod fragment merging (purely geometric) ─────────────────────────
MAX_ROD_GAP_M        = 0.12    # max gap (m) between co-axial fragments to merge (was 0.08)
COAXIAL_ANGLE_DEG    = 20.0    # max angle (°) between PCA axes to be considered co-axial
COAXIAL_MAX_OFFSET_M = 0.012   # max lateral offset (m) between the two fragment axes
COAXIAL_MIN_ELONG    = 1.5     # min PCA elongation (σ1/σ2) for a cluster to count as a rod (was 1.8)

# ── CLAHE contrast enhancement ────────────────────────────────────────────────
CLAHE_CLIP_LIMIT     = 3.0    # was 2.0 — higher local contrast helps thin rods

# ── Color-distance modifier in 3D grouping ────────────────────────────────────
COLOR_DIST_WEIGHT    = 0.3    # weight for HSV repulsion term in overlap grouping
                              # (0 = off; helps separate differently-coloured touching parts)

# ── Amodal bbox for through-joint rods ────────────────────────────────────────
ROD_PROTRUSION_M     = 0.030  # estimated protrusion beyond each connected component (m)

# ── Image sampling: target this many views per elevation ring ─────────────────
IMAGE_VIEWS_PER_RING = 8      # stride = max(1, len(ring)//this); Images90 (8 imgs) → stride 1
