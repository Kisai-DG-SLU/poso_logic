#!/bin/bash
# Docker entrypoint — PosoLogic Triage Médical CHSA
set -e

MODE="${1:-serve}"
PORT="${PORT:-8000}"
LOG_DIR="/app/logs"
MLFLOW_DIR="/app/mlruns"
MODEL_DIR="/app/models/merged_dpo"

mkdir -p "$LOG_DIR" "$MLFLOW_DIR"

case "$MODE" in
    serve)
        echo "============================================"
        echo "  PosoLogic — API de Triage Médical CHSA"
        echo "  Port: $PORT"
        echo "  Modèle: $MODEL_DIR"
        echo "============================================"
        exec python3 /app/api.py
        ;;

    mlflow)
        echo "============================================"
        echo "  PosoLogic — Dashboard MLflow"
        echo "  Port: $PORT"
        echo "  Tracking: file://$MLFLOW_DIR"
        echo "============================================"
        # Reconstruire les runs depuis les logs si disponibles
        python3 /app/mlflow_tracker.py reconstruct 2>/dev/null || true
        exec mlflow ui \
            --backend-store-uri "file://$MLFLOW_DIR" \
            --host 0.0.0.0 \
            --port "$PORT"
        ;;

    evaluate)
        echo "============================================"
        echo "  PosoLogic — Évaluation du Modèle"
        echo "============================================"
        exec python3 -c "
import mlflow_tracker as mt
mt.reconstruct_runs_from_logs()
print('Évaluation terminée. Dashboard MLflow prêt.')
"
        ;;

    *)
        echo "Usage: docker run ... [serve|mlflow|evaluate]"
        echo "  serve     → API de triage médical (défaut)"
        echo "  mlflow    → Dashboard MLflow métriques"
        echo "  evaluate  → Reconstruction des runs + évaluation"
        exit 1
        ;;
esac