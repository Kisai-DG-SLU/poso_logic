#!/bin/bash
# Lancement du serveur MLflow pour le dashboard de métriques PosoLogic
# Usage: bash start_mlflow_ui.sh [port]

MLFLOW_DIR="/mnt/prod/mlruns"
PORT="${1:-5050}"

mkdir -p "$MLFLOW_DIR"

echo "============================================"
echo "  MLflow Dashboard — PosoLogic"
echo "============================================"
echo ""
echo "  Tracking URI : file://$MLFLOW_DIR"
echo "  UI Port      : $PORT"
echo "  URL          : http://localhost:$PORT"
echo ""
echo "============================================"
echo ""

cd /mnt/prod && mlflow ui \
    --backend-store-uri "file://$MLFLOW_DIR" \
    --host 0.0.0.0 \
    --port "$PORT"
