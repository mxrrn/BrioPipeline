#!/bin/bash
# NOTE: belongs to the fast pipeline, which is not part of this repository.
#       The script is kept for reference and exits if brio_fast_pipeline/ is absent.
# Export YOLO training labels from slow-pipeline outputs.
# Logs are written automatically to brio_fast_pipeline/logs/
#
# Usage:
#   ./labels.sh              # export all completed samples
#   ./labels.sh 113 114 115  # export specific samples

PY="${PYTHON:-/home/mxrn/miniconda3/envs/brio-3d/bin/python}"
DIR="$(dirname "$0")/brio_fast_pipeline"
if [ ! -d "$DIR" ]; then
    echo "brio_fast_pipeline/ is not included in this repository (development stopped)." >&2
    exit 1
fi
cd "$DIR"
if [ $# -gt 0 ]; then
    $PY label_exporter.py --samples "$@"
else
    $PY label_exporter.py
fi
