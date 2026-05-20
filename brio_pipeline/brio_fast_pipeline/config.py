"""Central configuration for the BRIO fast inference pipeline (Option C)."""
from pathlib import Path

# ── Root paths ─────────────────────────────────────────────────────────────
BA_ROOT        = Path("/mnt/c/BA")
PROJECT_ROOT   = BA_ROOT / "00-project" / "brio_pipeline"
SLOW_PIPELINE  = PROJECT_ROOT / "brio_3d_pipeline"
FAST_PIPELINE  = PROJECT_ROOT / "brio_fast_pipeline"

SLOW_OUTPUTS   = SLOW_PIPELINE / "outputs"
DATASET_DIR    = FAST_PIPELINE / "dataset"
CALIBRATION_FILE = FAST_PIPELINE / "calibration" / "rig_poses.pkl"
RUNS_DIR       = FAST_PIPELINE / "runs"
LOGS_DIR       = FAST_PIPELINE / "logs"

# ── Class vocabulary (29 classes, alphabetical) ────────────────────────────
CLASSES = [
    "blwo11", "blwo21", "bo",    "no",    "nu",
    "pl",     "plpl53", "plwo21","plwo31","plwo33",
    "plwo53", "ro_lo",  "rome",  "rosm",  "sclo",
    "scme",   "scsm",   "sl",    "stpl5", "stwo3",
    "stwo4",  "stwo5",  "stwo6", "stwo7", "stwo9",
    "ti",     "wa",     "whre",  "whwh",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES  = len(CLASSES)

# ── YOLO training ──────────────────────────────────────────────────────────
YOLO_BASE_MODEL  = "yolov8n.pt"    # pretrained COCO nano weights
YOLO_EPOCHS      = 100
YOLO_IMGSZ       = 640
YOLO_BATCH       = 16
TRAIN_VAL_SPLIT  = 0.85            # 85% train, 15% val

# ── Label export ───────────────────────────────────────────────────────────
MIN_MASK_AREA_PX = 100             # masks smaller than this are skipped

# ── Calibration ────────────────────────────────────────────────────────────
CALIBRATION_SAMPLES = [113]        # reference samples used to build rig model
                                   # add more for more robust pose averaging

# ── Triangulation ──────────────────────────────────────────────────────────
MIN_VIEWS = 2                      # minimum views to triangulate a detection
CONF_THRESHOLD = 0.25              # minimum YOLO confidence to use a detection

# ── Connection inference ───────────────────────────────────────────────────
# Maximum 3D distance (world units) between two component centroids
# for them to be considered physically connected.
CONNECTION_DIST_THRESH = 0.12

# Slot compatibility: which class pairs can form a connection.
# Symmetric — order doesn't matter.
SLOT_RULES = {
    frozenset({"nu",    "bo"}),
    frozenset({"nu",    "ro_lo"}),
    frozenset({"nu",    "rome"}),
    frozenset({"nu",    "rosm"}),
    frozenset({"nu",    "sclo"}),
    frozenset({"nu",    "scme"}),
    frozenset({"nu",    "scsm"}),
    frozenset({"bo",    "blwo11"}),
    frozenset({"bo",    "blwo21"}),
    frozenset({"bo",    "plwo21"}),
    frozenset({"bo",    "plwo31"}),
    frozenset({"bo",    "plwo33"}),
    frozenset({"bo",    "plwo53"}),
    frozenset({"bo",    "plpl53"}),
    frozenset({"ro_lo", "blwo11"}),
    frozenset({"ro_lo", "blwo21"}),
    frozenset({"ro_lo", "plwo21"}),
    frozenset({"ro_lo", "plwo31"}),
    frozenset({"ro_lo", "plwo33"}),
    frozenset({"ro_lo", "plwo53"}),
    frozenset({"ro_lo", "plpl53"}),
    frozenset({"rome",  "blwo11"}),
    frozenset({"rome",  "blwo21"}),
    frozenset({"rome",  "plwo21"}),
    frozenset({"rome",  "plwo31"}),
    frozenset({"rome",  "plwo33"}),
    frozenset({"rome",  "plwo53"}),
    frozenset({"rome",  "plpl53"}),
    frozenset({"rosm",  "blwo11"}),
    frozenset({"rosm",  "blwo21"}),
    frozenset({"sclo",  "blwo11"}),
    frozenset({"sclo",  "blwo21"}),
    frozenset({"scme",  "blwo11"}),
    frozenset({"scme",  "blwo21"}),
    frozenset({"scsm",  "blwo11"}),
    frozenset({"scsm",  "blwo21"}),
    frozenset({"whwh",  "ro_lo"}),
    frozenset({"whwh",  "rome"}),
    frozenset({"whre",  "ro_lo"}),
    frozenset({"whre",  "rome"}),
    frozenset({"ti",    "whwh"}),
    frozenset({"ti",    "whre"}),
    frozenset({"sl",    "ro_lo"}),
    frozenset({"sl",    "rome"}),
    frozenset({"no",    "ro_lo"}),
    frozenset({"no",    "rome"}),
    frozenset({"pl",    "blwo11"}),
    frozenset({"pl",    "blwo21"}),
    frozenset({"wa",    "bo"}),
    frozenset({"wa",    "ro_lo"}),
    frozenset({"wa",    "rome"}),
    frozenset({"stwo3", "blwo11"}),
    frozenset({"stwo3", "blwo21"}),
    frozenset({"stwo4", "blwo11"}),
    frozenset({"stwo4", "blwo21"}),
    frozenset({"stwo5", "blwo11"}),
    frozenset({"stwo5", "blwo21"}),
    frozenset({"stwo6", "blwo11"}),
    frozenset({"stwo6", "blwo21"}),
    frozenset({"stwo7", "blwo11"}),
    frozenset({"stwo7", "blwo21"}),
    frozenset({"stwo9", "blwo11"}),
    frozenset({"stwo9", "blwo21"}),
    frozenset({"stpl5", "blwo11"}),
    frozenset({"stpl5", "blwo21"}),
}
