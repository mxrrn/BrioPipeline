#!/bin/bash
# Build the fixed camera rig calibration from DUSt3R caches.
# Logs are written automatically to brio_fast_pipeline/logs/
# Only needs to be run once (or re-run with more reference samples).
#
# Usage:
#   ./calibrate.sh 113
#   ./calibrate.sh 113 114 115

PY=/home/mxrn/miniconda3/envs/brio-3d/bin/python
cd "$(dirname "$0")/brio_fast_pipeline"
$PY calibrator.py --samples "$@"
