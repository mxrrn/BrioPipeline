#!/usr/bin/env bash
# Train the MobileNetV3-small component visual classifier.
# Weights are saved to brio_3d_pipeline/classification/component_classifier.pth
# and picked up automatically by the pipeline on the next run.
#
# Usage:
#   ./train_classifier.sh              # defaults: 40 epochs, batch=32, lr=3e-4
#   ./train_classifier.sh --epochs 60 --batch 16

set -e
cd "$(dirname "$0")"

conda run -n brio-3d --no-capture-output \
    python brio_3d_pipeline/classification/component_classifier.py \
        --train \
        "$@" \
    2>&1 | tee brio_3d_pipeline/logs/train_classifier.log
