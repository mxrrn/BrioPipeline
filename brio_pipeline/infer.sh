#!/bin/bash
# NOTE: belongs to the fast pipeline, which is not part of this repository.
#       The script is kept for reference and exits if brio_fast_pipeline/ is absent.
# Run fast inference on an unseen sample.
# Logs are written automatically to brio_fast_pipeline/logs/
#
# Usage:
#   ./infer.sh 120
#   ./infer.sh 120 --puml /mnt/c/BA/02-resources/data/constructions/Sample_120_InstanceDiagram/InstanceDiagramS120.puml

PY="${PYTHON:-/home/mxrn/miniconda3/envs/brio-3d/bin/python}"
DIR="$(dirname "$0")/brio_fast_pipeline"
if [ ! -d "$DIR" ]; then
    echo "brio_fast_pipeline/ is not included in this repository (development stopped)." >&2
    exit 1
fi
cd "$DIR"
$PY infer.py --sample "$1" "${@:2}"
