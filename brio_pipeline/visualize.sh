#!/bin/bash
# Visualise the 3D proposal clouds for a completed slow-pipeline sample.
#
# Usage:
#   ./visualize.sh 113                              # latest run
#   ./visualize.sh 113 run_001_20260526_2123        # specific run

PY=/home/mxrn/miniconda3/envs/brio-3d/bin/python
cd "$(dirname "$0")/brio_3d_pipeline"
if [ -n "$2" ]; then
    $PY visualize.py --sample "$1" --run "$2"
else
    $PY visualize.py --sample "$1"
fi
