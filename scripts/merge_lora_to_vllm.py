"""
Merge les poids LoRA DPO dans Qwen3-1.7B pour utilisation directe avec vLLM
Contourne le besoin de compilation CUDA des kernels LoRA natifs vLLM

Usage: pixi run python scripts/merge_lora_to_vllm.py
"""

import os
import sys
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MODEL_ID = "Qwen/Qwen3-1.7B"
ADAPTER_PATH = "/mnt/prod/models/checkpoints/dpo_a2_optimized/final"
OUTPUT_PATH = "/mnt/prod/models/merged_dpo_vllm"

print("=" * 60)
print("Merge LoRA DPO → Modèle de base (Qwen3-1.7B)")
print("=" * 60)

print(f"\n[1/4] Chargement du modèle de base: {MODEL_ID}")
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

print(f"\n[2/4] Chargement de l'adaptateur LoRA: {ADAPTER_PATH}")
from peft import PeftModel

model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print(f"\n[3/4] Fusion des poids (merge_and_unload)...")
model = model.merge_and_unload()

print(f"\n[4/4] Sauvegarde du modèle merged → {OUTPUT_PATH}")
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)

model_size = sum(f.stat().st_size for f in Path(OUTPUT_PATH).rglob("*")) / (1024**3)
print(f"\n✅ Modèle merged sauvegardé ({model_size:.1f} Go)")
print(f"   Chemin: {OUTPUT_PATH}")
print(f"   Utilisation vLLM: LLM(model='{OUTPUT_PATH}', ...)")
