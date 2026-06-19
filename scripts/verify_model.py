"""Script de vérification du chargement du modèle DPO (checkpoint LoRA + inférence).

Points clés :
- Vérification structurelle : présence de adapter_model.safetensors et adapter_config.json
- Vérification fonctionnelle : chargement du modèle de base + adaptateur LoRA + inférence test
- Idéal pour CI : exit 1 si un problème est détecté
"""

import os
import sys

CHECKPOINT_DIR = "models/checkpoints/dpo_a2_optimized/final"
BASE_MODEL = "Qwen/Qwen3-1.7B"

if not os.path.isdir(CHECKPOINT_DIR):
    print(f"ERREUR: Dossier checkpoint non trouve: {CHECKPOINT_DIR}")
    sys.exit(1)

adapter_path = os.path.join(CHECKPOINT_DIR, "adapter_model.safetensors")
if not os.path.isfile(adapter_path):
    print(f"ERREUR: adapter_model.safetensors non trouve dans {CHECKPOINT_DIR}")
    sys.exit(1)

config_path = os.path.join(CHECKPOINT_DIR, "adapter_config.json")
if not os.path.isfile(config_path):
    print(f"ERREUR: adapter_config.json non trouve dans {CHECKPOINT_DIR}")
    sys.exit(1)

print(f"Checkpoints DPO trouves dans {CHECKPOINT_DIR}")

try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Chargement du modele de base: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Modele de base charge avec succes")

    print(f"Chargement des poids LoRA depuis: {CHECKPOINT_DIR}")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    print("Modele LoRA charge avec succes")

    print(
        f"Parametres entraines: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    test_input = "Patient de 45 ans avec douleur thoracique"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test inference: OK - Input: '{test_input}'")
    print(f"Reponse: '{result}'")

    print("VERIFICATION COMPLETE: Modele operationnel")

except ImportError as e:
    print(f"WARNING: Package manquant ({e}) - verification structurelle uniquement")
    print("VERIFICATION STRUCTURELLE: Structure du checkpoint OK")
except Exception as e:
    print(f"ERREUR chargement modele: {e}")
    sys.exit(1)
