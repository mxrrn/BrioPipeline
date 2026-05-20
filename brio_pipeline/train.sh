#!/bin/bash
# Train YOLOv8n on the exported BRIO dataset.
# Logs are written automatically to brio_fast_pipeline/logs/
#
# Usage:
#   ./train.sh
#   ./train.sh --epochs 150 --batch 8

PY=/home/mxrn/miniconda3/envs/brio-3d/bin/python
cd "$(dirname "$0")/brio_fast_pipeline"
$PY train.py "$@"
