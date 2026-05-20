#!/bin/bash
# Run fast inference on an unseen sample.
# Logs are written automatically to brio_fast_pipeline/logs/
#
# Usage:
#   ./infer.sh 120
#   ./infer.sh 120 --puml /mnt/c/BA/02-resources/data/constructions/Sample_120_InstanceDiagram/InstanceDiagramS120.puml

PY=/home/mxrn/miniconda3/envs/brio-3d/bin/python
cd "$(dirname "$0")/brio_fast_pipeline"
$PY infer.py --sample "$1" "${@:2}"
