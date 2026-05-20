#!/bin/bash
# Export YOLO training labels from slow-pipeline outputs.
# Logs are written automatically to brio_fast_pipeline/logs/
#
# Usage:
#   ./labels.sh              # export all completed samples
#   ./labels.sh 113 114 115  # export specific samples

PY=/home/mxrn/miniconda3/envs/brio-3d/bin/python
cd "$(dirname "$0")/brio_fast_pipeline"
if [ $# -gt 0 ]; then
    $PY label_exporter.py --samples "$@"
else
    $PY label_exporter.py
fi
