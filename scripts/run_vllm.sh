#!/bin/bash
# Déploiement vLLM haute performance — Agent de Triage Médical CHSA
# Utilise le modèle merged (LoRA fusionné) pour éviter la compilation CUDA
set -e

MODEL_PATH="/mnt/prod/models/merged_dpo_vllm"
PORT="${PORT:-8000}"

echo "============================================"
echo "  PosoLogic — API vLLM (haute performance)"
echo "  Modèle merged: $MODEL_PATH"
echo "  Port: $PORT"
echo "  vLLM 0.8.5 + torch 2.6.0+cu124"
echo "============================================"

exec python3 scripts/06_api_vllm.py
