#!/usr/bin/env bash
# ----------------------------------------------------------------
# Run the Gradio brain-tumor segmentation app locally.
# Usage:  bash run_gradio.sh
# Optionally set MODEL_WEIGHTS_PATH to skip the HF download:
#   MODEL_WEIGHTS_PATH=/path/to/best_slim.pt bash run_gradio.sh
# ----------------------------------------------------------------
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Use the local slim weights if the env var is set (avoids HF download)
if [ -n "$MODEL_WEIGHTS_PATH" ]; then
  export MODEL_WEIGHTS_PATH
fi

echo "Starting Gradio app on http://localhost:7860 ..."
testenv/bin/python app.py
