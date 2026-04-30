"""
Script DPO minimal - contournant tous les problèmes connus.
Se concentre sur une seule étape d'entraînement réussie.
"""
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gc
import torch
import torch.nn.functional as F
from pathlib import Path
import json
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constantes
MODEL_PATH = "/mnt/prod/models/checkpoints/sft_final"
REF_MODEL_PATH = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "/mnt/prod/models/checkpoints/dpo_minimal"
DEBUG_MODE = True
MAX_LENGTH = 256  # Valeur très basse pour réduire la mémoire

def debug(msg):
    """Affiche un message de debug si DEBUG_MODE est activé"""
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}")

def free_memory():
    """Libère la mémoire GPU de manière agressive"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force collecte mémoire Python
    for _ in range(3):
        gc.collect()
    
    # Affiche l'utilisation mémoire
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"Mémoire GPU: {allocated:.2f} GB allouée / {reserved:.2f} GB réservée")

def log_gpu_usage(stage):
    """Enregistre l'utilisation du GPU à une étape donnée"""
    if torch.cuda.is_available():
        print(f"--- Mémoire GPU ({stage}) ---")
        for i in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(i) / (1024 ** 3)
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"GPU {i}: {mem_used:.2f} GB / {mem_total:.2f} GB")

def create_example_batch():
    """Crée un exemple de batch simple pour test DPO"""
    return {
        "instruction": ["Comment traiter une migraine?"],
        "chosen": ["Pour traiter une migraine, prenez du repos dans une pièce calme et sombre, buvez beaucoup d'eau et prenez un analgésique si nécessaire."],
        "rejected": ["Prenez n'importe quel médicament que vous trouvez et continuez vos activités normalement."]
    }

def compute_logps(logits, input_ids, attention_mask):
    """Calcule les log probabilités pour les séquences"""
    # Décaler les logits et les tokens pour l'autoregression
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()
    
    # Récupérer les probabilités des tokens suivants
    token_logprobs = torch.gather(
        F.log_softmax(shift_logits, dim=-1),
        dim=-1,
        index=shift_tokens.unsqueeze(-1)
    ).squeeze(-1)
    
    # Masquer les valeurs padding
    token_logprobs = token_logprobs * shift_mask
    
    # Sommer les log probs
    seq_logprobs = token_logprobs.sum(dim=-1)
    
    return seq_logprobs

def train_dpo_step():
    """Effectue une seule étape d'entraînement DPO"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n=== Configuration DPO ===")
        print(f"Modèle: {MODEL_PATH}")
        print(f"Référence: {REF_MODEL_PATH}")
        print(f"Device: {device}")
        
        # Vérification GPU
        if device == "cuda":
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"Mémoire GPU totale: {total_mem:.2f} GB")
        
        # Chargement des modèles
        print("\n1. Chargement tokenizer...")
        # Utiliser le tokenizer du modèle de base plutôt que celui du SFT (qui a un problème)
        tokenizer = AutoTokenizer.from_pretrained(REF_MODEL_PATH, trust_remote_code=True)
        
        log_gpu_usage("après tokenizer")
        
        print("\n2. Chargement du modèle policy...")
        policy_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,  # Réduit pour économiser mémoire
            device_map="auto"
        )
        
        # Configuration LoRA
        print("\n3. Configuration LoRA...")
        peft_config = LoraConfig(
            r=4,  # Valeur très basse pour réduire la mémoire
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        policy_model = get_peft_model(policy_model, peft_config)
        policy_model.print_trainable_parameters()
        
        log_gpu_usage("après policy model")
        free_memory()
        
        print("\n4. Chargement du modèle référence...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            REF_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        ref_model.eval()  # Référence toujours en mode évaluation
        
        log_gpu_usage("après référence")
        free_memory()
        
        # Créer un optimiseur simple
        print("\n5. Configuration de l'optimiseur...")
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
        
        # Créer un batch de test simple
        print("\n6. Création du batch de test...")
        batch = create_example_batch()
        
        # Mode entraînement
        policy_model.train()
        
        # DPO étape unique
        print("\n7. Exécution d'une étape DPO...")
        
        # Tokeniser les entrées
        print("   7.1 Tokenisation...")
        instruction = batch["instruction"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        
        chosen_seqs = [f"{ins}\n{ch}" for ins, ch in zip(instruction, chosen)]
        rejected_seqs = [f"{ins}\n{rej}" for ins, rej in zip(instruction, rejected)]
        
        chosen_tokens = tokenizer(
            chosen_seqs, padding=True, truncation=True, max_length=MAX_LENGTH, 
            return_tensors="pt"
        ).to(device)
        
        rejected_tokens = tokenizer(
            rejected_seqs, padding=True, truncation=True, max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)
        
        log_gpu_usage("après tokenisation entrées")
        free_memory()
        
        # Forward pass policy
        print("   7.2 Forward pass policy model...")
        with torch.cuda.amp.autocast():  # Utilisation de la précision mixte
            chosen_policy_output = policy_model(**chosen_tokens)
            chosen_policy_logits = chosen_policy_output.logits
            
            log_gpu_usage("après chosen policy")
            free_memory()
            
            rejected_policy_output = policy_model(**rejected_tokens)
            rejected_policy_logits = rejected_policy_output.logits
            
            log_gpu_usage("après rejected policy")
            free_memory()
        
        # Forward pass reference
        print("   7.3 Forward pass reference model...")
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                chosen_ref_output = ref_model(**chosen_tokens)
                chosen_ref_logits = chosen_ref_output.logits
                
                log_gpu_usage("après chosen ref")
                free_memory()
                
                rejected_ref_output = ref_model(**rejected_tokens)
                rejected_ref_logits = rejected_ref_output.logits
                
                log_gpu_usage("après rejected ref")
                free_memory()
        
        # Calcul des log probs
        print("   7.4 Calcul des log probs...")
        policy_chosen_logps = compute_logps(
            chosen_policy_logits, chosen_tokens["input_ids"], 
            chosen_tokens["attention_mask"]
        )
        
        policy_rejected_logps = compute_logps(
            rejected_policy_logits, rejected_tokens["input_ids"],
            rejected_tokens["attention_mask"]
        )
        
        ref_chosen_logps = compute_logps(
            chosen_ref_logits, chosen_tokens["input_ids"],
            chosen_tokens["attention_mask"]
        )
        
        ref_rejected_logps = compute_logps(
            rejected_ref_logits, rejected_tokens["input_ids"],
            rejected_tokens["attention_mask"]
        )
        
        # Calcul des ratios et perte DPO
        print("   7.5 Calcul de la perte DPO...")
        chosen_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_ratio = policy_rejected_logps - ref_rejected_logps
        
        beta = 0.1
        logits = beta * (chosen_ratio - rejected_ratio)
        loss = -F.logsigmoid(logits).mean()
        
        print(f"   Perte DPO: {loss.item()}")
        
        # Rétropropagation
        print("   7.6 Rétropropagation...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        log_gpu_usage("après backward")
        free_memory()
        
        # Sauvegarde
        print("\n8. Sauvegarde du modèle...")
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        policy_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Sauvegarde métadonnées
        with open(output_dir / "dpo_minimal_config.json", 'w') as f:
            json.dump({
                "loss": loss.item(),
                "status": "success",
                "max_length": MAX_LENGTH,
                "model_path": MODEL_PATH,
                "ref_model_path": REF_MODEL_PATH,
                "example_batch": batch
            }, f, indent=4)
        
        print("\n=== ÉTAPE DPO RÉUSSIE ===")
        print(f"Modèle sauvegardé dans: {output_dir}")
        return True
        
    except Exception as e:
        import traceback
        print("\n==== ERREUR CRITIQUE DURANT L'ÉTAPE DPO ====")
        print(f"Type d'erreur: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nTraceback complet:")
        traceback.print_exc()
        
        if torch.cuda.is_available():
            print("\nInformations GPU:")
            print(f"Mémoire allouée: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"Mémoire réservée: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        return False

if __name__ == "__main__":
    print("=== DÉBUT SCRIPT DPO MINIMAL ===")
    success = train_dpo_step()
    
    if success:
        print("\n✅ Script DPO minimal exécuté avec succès!")
    else:
        print("\n❌ Échec du script DPO minimal!")
