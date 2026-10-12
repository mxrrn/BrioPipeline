#!/bin/bash
# Annotate samples with the slow 3D pipeline.
# Logs are written automatically to brio_3d_pipeline/logs/
#
# Set PYTHON=/path/to/python to override the interpreter.
#
# Usage:
#   ./slow.sh 113
#   ./slow.sh 113 114 115 116 117
#   ./slow.sh 113 114 --device cpu

PY="${PYTHON:-/home/mxrn/miniconda3/envs/brio-3d/bin/python}"
cd "$(dirname "$0")/brio_3d_pipeline"
$PY pipeline.py --samples "$@"
