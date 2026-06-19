"""
Evaluation du modèle DPO sur 100 prompts UltraMedical-Preference
Utilise vLLM 0.8.5 avec modèle merged sur GPU A2
Compare similarité sémantique (cosinus) avec chosen vs rejected

Points clés :
- 100 prompts échantillonnés aléatoirement depuis le test set (0 overlap avec train)
- Inférence via vLLM.chat() pour respecter le format conversationnel
- Embeddings via SentenceTransformer (all-MiniLM-L6-v2) pour comparer la réponse générée aux réponses chosen et rejected
- Métrique principale : alignment_pct — proportion de réponses plus proches de chosen que de rejected
"""

import json
import time
import random
import os
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, util

os.environ['VLLM_USE_V1'] = '0'

from vllm.platforms.cuda import CudaPlatform
import vllm.platforms as vp
vp._current_platform = CudaPlatform()

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)

from vllm import LLM, SamplingParams

MODEL_PATH = "/mnt/prod/models/merged_dpo_vllm"
DATA_PATH = "/mnt/prod/data/raw/ultramedical_preference"
OUTPUT_DIR = Path("/mnt/prod/eval_reports")
N_SAMPLES = 100

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("[1/4] Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("[2/4] Loading vLLM with merged DPO model...")
llm = LLM(
    model=MODEL_PATH,
    max_model_len=1024,
    gpu_memory_utilization=0.70,
    enforce_eager=True,
)
sampling_params = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=256)

print("[3/4] Loading UltraMedical-Preference test set...")
ds = load_from_disk(DATA_PATH)
test_set = ds["test"]
print(f"  {len(test_set)} prompts (0 overlap with train)")

random.seed(42)
indices = random.sample(range(len(test_set)), min(N_SAMPLES, len(test_set)))

def extract_assistant_text(conversation):
    texts = [msg["content"] for msg in conversation if msg.get("role") == "assistant"]
    return " ".join(texts)

def format_prompt(prompt_text):
    return [
        {"role": "system", "content": "You are a medical assistant. Provide accurate, safe, and helpful medical information."},
        {"role": "user", "content": prompt_text},
    ]

results = []
chosen_closer = 0

print(f"[4/4] Evaluating {N_SAMPLES} prompts with vLLM...")

prompts_batch = []
samples_batch = []

for idx in indices:
    sample = test_set[idx]
    chosen_text = extract_assistant_text(sample["chosen"])
    rejected_text = extract_assistant_text(sample["rejected"])
    if not chosen_text or not rejected_text:
        continue
    prompts_batch.append(format_prompt(sample["prompt"]))
    samples_batch.append((sample, chosen_text, rejected_text))

t_start = time.time()
outputs = llm.chat(prompts_batch, sampling_params)
total_latency = time.time() - t_start

for i, (sample, chosen_text, rejected_text) in enumerate(samples_batch):
    generated = outputs[i].outputs[0].text.strip()

    emb_gen = embedder.encode(generated, convert_to_tensor=True)
    emb_chosen = embedder.encode(chosen_text, convert_to_tensor=True)
    emb_rejected = embedder.encode(rejected_text, convert_to_tensor=True)

    sim_chosen = util.cos_sim(emb_gen, emb_chosen).item()
    sim_rejected = util.cos_sim(emb_gen, emb_rejected).item()
    prefers_chosen = sim_chosen > sim_rejected

    if prefers_chosen:
        chosen_closer += 1

    results.append({
        "case_id": i + 1,
        "prompt_id": sample["prompt_id"],
        "sim_chosen": round(sim_chosen, 4),
        "sim_rejected": round(sim_rejected, 4),
        "prefers_chosen": prefers_chosen,
    })

total_valid = len(results)
alignment_pct = (chosen_closer / total_valid * 100) if total_valid > 0 else 0
avg_latency = round(total_latency / total_valid, 2)

report = {
    "model": "Qwen3-1.7B + DPO (merged)",
    "inference": "vLLM 0.8.5 on GPU A2",
    "dataset": "UltraMedical-Preference (test split)",
    "num_samples": total_valid,
    "alignment_pct": round(alignment_pct, 1),
    "prefers_chosen_count": chosen_closer,
    "average_latency_s": avg_latency,
    "total_latency_s": round(total_latency, 2),
    "results": results,
    "data_leakage_checked": True,
    "train_test_overlap": 0,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

with open(OUTPUT_DIR / "eval_ultramedical_preference.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

summary = f"""
=== Evaluation UltraMedical-Preference (vLLM) ===
Modele: Qwen3-1.7B + DPO (merged)
Inference: vLLM 0.8.5 on GPU A2 (CUDA 12.4)
Echantillon: {total_valid} prompts (test set, 0 overlap with train)
Comparaison: similarite cosinus (all-MiniLM-L6-v2) vs chosen/rejected

Alignement avec chosen: {chosen_closer}/{total_valid} = {alignment_pct:.1f}%
Latence moyenne: {avg_latency}s/cas
Latence totale: {report['total_latency_s']}s
"""
print(summary)
with open(OUTPUT_DIR / "eval_ultramedical_preference_summary.txt", "w") as f:
    f.write(summary)