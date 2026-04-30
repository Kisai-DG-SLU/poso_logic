#!/usr/bin/env python3
"""
Script d'évaluation clinique pour le modèle DPO
À exécuter sur les checkpoints intermédiaires
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json
from pathlib import Path
import time

# Chemin de base
CHECKPOINT_DIR = Path("/mnt/prod/models/checkpoints/dpo_a2_optimized")

# Cas cliniques de test pour le triage
TEST_CASES = [
    {
        "triage_level": "Urgence vitale",
        "instruction": "Patient de 45 ans, douleur thoracique intense irradiant à la mâchoire, sueurs, nausées. ATCD: HTA, tabagisme.",
        "expected_priority": "Immédiat - niveau 1",
        "key_signs": ["douleur thoracique", "irradiation", "sueurs", "nausées"]
    },
    {
        "triage_level": "Urgence relative",
        "instruction": "Enfant 3 ans, fièvre 39.5°C depuis 2 jours, vomissements occasionnels, boit normalement.",
        "expected_priority": "Dans les 2 heures - niveau 3",
        "key_signs": ["fièvre", "vomissements", "enfant"]
    },
    {
        "triage_level": "Urgence bénigne",
        "instruction": "Adulte 30 ans, entorse de la cheville lors d'une activité sportive, douleur modérée, capable de marcher.",
        "expected_priority": "Dans les 4 heures - niveau 4",
        "key_signs": ["entorse", "douleur modérée", "marche possible"]
    },
    {
        "triage_level": "Non urgent",
        "instruction": "Patient 50 ans, renouvellement d'ordonnance pour antihypertenseur, pas de symptômes particuliers.",
        "expected_priority": "Délai possible - niveau 5",
        "key_signs": ["renouvellement", "ordonnance", "pas de symptômes"]
    },
    {
        "triage_level": "Urgence vitale",
        "instruction": "Patient 70 ans, parole confuse, déviation de la commissure labiale, bras droit faible. Début il y a 30 minutes.",
        "expected_priority": "Immédiat - niveau 1 (AVC)",
        "key_signs": ["parole confuse", "déviation", "faiblesse", "brusque"]
    }
]

def load_model_for_inference(checkpoint_path):
    """Charge le modèle pour l'inférence"""
    print(f"Chargement du modèle depuis {checkpoint_path}...")
    
    # Charger la configuration Peft
    config = PeftConfig.from_pretrained(checkpoint_path)
    
    # Charger le modèle de base
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Charger les poids LoRA
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_triage_decision(model, tokenizer, instruction, device="cuda"):
    """Génère une décision de triage"""
    prompt = instruction + "\n\nQuelle est votre décision de triage ?"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraire juste la réponse
    if "Quelle est votre décision de triage ?" in response:
        response = response.split("Quelle est votre décision de triage ?")[-1].strip()
    
    return response

def evaluate_model_on_cases(model, tokenizer, test_cases):
    """Évalue le modèle sur les cas cliniques"""
    results = []
    
    print("\n" + "="*60)
    print("ÉVALUATION CLINIQUE DU MODÈLE DPO")
    print("="*60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nCas {i}: {case['triage_level']}")
        print(f"Instruction: {case['instruction'][:100]}...")
        
        start = time.time()
        response = generate_triage_decision(model, tokenizer, case['instruction'])
        elapsed = time.time() - start
        
        print(f"Réponse générée: {response[:200]}...")
        print(f"Attendu: {case['expected_priority']}")
        print(f"Temps: {elapsed:.2f}s")
        
        # Analyse simple
        priority_lower = case['expected_priority'].lower()
        response_lower = response.lower()
        
        # Vérifier si la priorité est correctement identifiée
        correct = any(word in response_lower for word in priority_lower.split())
        
        results.append({
            "case_id": i,
            "triage_level": case['triage_level'],
            "expected": case['expected_priority'],
            "generated": response[:200],
            "correct_identified": correct,
            "latency_seconds": elapsed
        })
    
    # Calculer les métriques
    total = len(results)
    correct_count = sum(1 for r in results if r['correct_identified'])
    
    print("\n" + "="*60)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("="*60)
    print(f"Cas évalués: {total}")
    print(f"Identifications correctes: {correct_count}/{total} ({100*correct_count/total:.1f}%)")
    
    latency_sum = sum(r['latency_seconds'] for r in results)
    print(f"Latence moyenne: {latency_sum/total:.2f}s")
    
    return results

def evaluate_checkpoint(checkpoint_path, output_dir):
    """Évalue un checkpoint spécifique"""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Checkpoint introuvable: {checkpoint_path}")
        return None
    
    print(f"\n🔍 Évaluation du checkpoint: {checkpoint_path.name}")
    
    try:
        # Charger le modèle
        model, tokenizer = load_model_for_inference(checkpoint_path)
        
        # Évaluer
        results = evaluate_model_on_cases(model, tokenizer, TEST_CASES)
        
        # Sauvegarder les résultats
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results_file = output_dir / f"eval_{checkpoint_path.name}.json"
        with open(results_file, "w") as f:
            json.dump({
                "checkpoint": str(checkpoint_path),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
                "summary": {
                    "total_cases": len(results),
                    "correct_identifications": sum(1 for r in results if r['correct_identified']),
                    "average_latency": sum(r['latency_seconds'] for r in results) / len(results)
                }
            }, f, indent=2)
        
        print(f"\n✅ Résultats sauvegardés: {results_file}")
        
        # Libérer la mémoire
        del model
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"Erreur lors de l'évaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_latest_checkpoint():
    """Évalue le dernier checkpoint disponible"""
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint-*"))
    
    if not checkpoints:
        print("Aucun checkpoint trouvé.")
        return
    
    # Prendre le dernier checkpoint
    latest = checkpoints[-1]
    
    # Répertoire de sortie
    output_dir = Path("/mnt/memory/dpo_evaluations")
    
    evaluate_checkpoint(latest, output_dir)

if __name__ == "__main__":
    # Évaluer le dernier checkpoint
    evaluate_latest_checkpoint()
    
    # Ou évaluer un checkpoint spécifique
    # evaluate_checkpoint("/mnt/prod/models/checkpoints/dpo_a2_optimized/checkpoint-1000", "/mnt/memory/dpo_evaluations")
