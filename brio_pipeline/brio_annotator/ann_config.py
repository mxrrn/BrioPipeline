"""
Configuration for the BRIO redesigned annotator pipeline.

Named ann_config (not config) to avoid clashing with brio_3d_pipeline/config.py
when both directories are on sys.path.
"""
import sys
from pathlib import Path

# Make brio_3d_pipeline importable (for component_map, puml_parser, etc.)
_3D_DIR = Path(__file__).parent.parent / "brio_3d_pipeline"
if str(_3D_DIR) not in sys.path:
    sys.path.insert(0, str(_3D_DIR))

from component_map import FOLDER_TO_CLASS

# ── Root paths ────────────────────────────────────────────────────────────────
BA_ROOT        = Path("/mnt/c/BA")
DATA_ROOT      = BA_ROOT / "02-resources/data"
CONSTRUCTIONS  = DATA_ROOT / "constructions"
MULTI_VIEW     = DATA_ROOT / "multi_view_images"
COMPONENT_IMAGES = DATA_ROOT / "component_images"
PROJECT_ROOT   = BA_ROOT / "00-project/brio_pipeline/brio_annotator"
OUTPUTS_ROOT   = PROJECT_ROOT / "outputs"

# ── Model weights ─────────────────────────────────────────────────────────────
SAM_WEIGHTS    = BA_ROOT / "03-code/sam_weights/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
DEVICE         = "cuda"

# ── DINOv2 ───────────────────────────────────────────────────────────────────
DINO_MODEL     = "dinov2_vits14"   # ViT-S/14, 384-dim — smallest, fast on GPU

# ── View selection ────────────────────────────────────────────────────────────
VIEW_ELEVATIONS   = ["Images30", "Images45", "Images60", "Images90"]
VIEW_STRIDE       = 4              # sample every 4th image per elevation ring
TOP_K_VIEWS       = 3             # number of best views to evaluate per sample

# ── SAM prompted mode ─────────────────────────────────────────────────────────
SAM_MIN_MASK_AREA = 80            # minimum mask area to keep (pixels)

# ── Homography propagation ────────────────────────────────────────────────────
HOMOG_MIN_MATCHES  = 8            # minimum ORB inlier matches for a valid homography
HOMOG_MIN_OVERLAP  = 0.30         # discard warped bbox if fg overlap < this fraction

# ── Background threshold (same as brio_3d_pipeline) ──────────────────────────
BG_THRESHOLD = 245

# ── YOLO class registry ───────────────────────────────────────────────────────
CLASSES      = sorted(set(FOLDER_TO_CLASS.values()))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ── Rod classes (amodal bbox merging applies) ─────────────────────────────────
ROD_CLASSES  = {"rolo", "rome", "rosm"}
