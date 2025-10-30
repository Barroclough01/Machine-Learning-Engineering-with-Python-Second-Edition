#!/usr/bin/env bash
# Start an MLflow tracking server (Linux/WSL).
# Notes:
# - Run this from WSL or a Unix-like shell. If the file lives on Windows (\n), make sure your editor saved it with LF endings.
# - Ensure `mlflow` is installed and on PATH, or activate the environment that provides it before running this script.
# Example activation (uncomment and adapt if you use conda):
#   source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate my-mlflow-env

mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root C:/Users/paxto/Machine-Learning-Engineering-with-Python-Second-Edition/Chapter03/mlflow-advanced/artifacts \
    --host 0.0.0.0 \
    --port 8000

