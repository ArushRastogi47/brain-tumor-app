#!/usr/bin/env bash
# ----------------------------------------------------------------
# Run the FastAPI app locally.
# Usage:  bash run_local_api.sh [path/to/best.pt]
# ----------------------------------------------------------------
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$(dirname "$SCRIPT_DIR")/results (1)"
cd "$RESULTS_DIR"

MODEL_PT="${1:-swin-evdnet-v2-3d/run/checkpoints/best.pt}"
echo "Using model: $MODEL_PT"
export MODEL_PATH="$SCRIPT_DIR/../results (1)/$MODEL_PT"
export THRESHOLDS_PATH="$SCRIPT_DIR/../results (1)/swin-evdnet-v2-3d/burden_thresholds.json"

echo "Installing deps (first run only)..."
pip install --quiet fastapi uvicorn[standard] python-multipart torch nibabel scipy monai einops

echo "Starting FastAPI server on http://localhost:8000 ..."
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
