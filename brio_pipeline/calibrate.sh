#!/bin/bash
# NOTE: belongs to the fast pipeline, which is not part of this repository.
#       The script is kept for reference and exits if brio_fast_pipeline/ is absent.
# Build the fixed camera rig calibration from DUSt3R caches.
# Logs are written automatically to brio_fast_pipeline/logs/
# Only needs to be run once (or re-run with more reference samples).
#
# Usage:
#   ./calibrate.sh 113
#   ./calibrate.sh 113 114 115

PY="${PYTHON:-/home/mxrn/miniconda3/envs/brio-3d/bin/python}"
DIR="$(dirname "$0")/brio_fast_pipeline"
if [ ! -d "$DIR" ]; then
    echo "brio_fast_pipeline/ is not included in this repository (development stopped)." >&2
    exit 1
fi
cd "$DIR"
$PY calibrator.py --samples "$@"
