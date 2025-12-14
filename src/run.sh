#!/usr/bin/env bash
# run.sh - run the full pipeline scripts in order
# This script is used by the Docker image and local testing to execute the
# main pipeline stages in sequence for demonstration purposes.

set -euo pipefail

echo "[run.sh] Starting full pipeline run at $(date --iso-8601=seconds)"

echo ""
echo "Running data processing..."
python -u data_processing_01.py

echo ""
echo "Running BoW and Logistic Regression training..."
python -u baseline_model_train_02.py

echo ""
echo "Running BoW and Logistic Regression evaluation..."
python -u baseline_model_eval_03.py

echo ""
echo "Running MLP training..."
python -u mlp_train_04.py

echo ""
echo "Running MLP evaluation..."
python -u mlp_eval_05.py

echo ""
echo "Running inference on new data..."
python -u inference_06.py

echo ""
echo "Pipeline finished successfully."

echo "[run.sh] Pipeline finished at $(date --iso-8601=seconds)"