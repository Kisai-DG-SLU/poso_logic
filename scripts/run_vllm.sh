#!/bin/bash
# Déploiement vLLM haute performance — Agent de Triage Médical CHSA
# Note: vLLM nécessite un environnement dédié (conflit de dépendances avec pixi)
#   pip install vllm>=0.6.0
set -e

MODEL_NAME="Qwen/Qwen3-1.7B"
ADAPTER_PATH="/mnt/prod/models/checkpoints/dpo_a2_optimized/final"
PORT="${PORT:-8000}"

echo "============================================"
echo "  PosoLogic — API vLLM (haute performance)"
echo "  Modèle: $MODEL_NAME"
echo "  Adapter LoRA: $ADAPTER_PATH"
echo "  Port: $PORT"
echo "============================================"
echo ""
echo "  Installation : pip install vllm>=0.6.0"
echo "  Ou via Docker: cf. Dockerfile (CUDA 12.4 + vLLM)"
echo ""

exec vllm serve "$MODEL_NAME" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --enable-lora \
    --lora-modules "chsa-triage=$ADAPTER_PATH" \
    --api-key "none"
