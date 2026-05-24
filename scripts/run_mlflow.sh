#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8080}"
echo "Starting MLflow UI at http://127.0.0.1:${PORT}"
mlflow server \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns
