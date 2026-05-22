#!/bin/bash
# Visualise the 3D proposal clouds for a completed slow-pipeline sample.
#
# Usage:
#   ./visualize.sh 113

PY=/home/mxrn/miniconda3/envs/brio-3d/bin/python
cd "$(dirname "$0")/brio_3d_pipeline"
$PY visualize.py --sample "$1"
