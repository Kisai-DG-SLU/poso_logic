"""
Script DPO complet basé sur l'approche minimale qui a fonctionné.
Conçu pour un entraînement DPO stable sur le dataset complet.
"""
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from pathlib import Path
import json
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import time

# Constantes
SFT_MODEL_PATH = "/mnt/prod/models/checkpoints/sft_final"
REF_MODEL_PATH = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "/mnt/prod/models/checkpoints/dpo_final_working"
DATASET_PATH = "/mnt/prod/data/processed/dpo_dataset"
MAX_LENGTH = 384  # Réduit pour économiser la mémoire
DEBUG_MODE = True

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

def get_config():
    """Configuration DPO complète"""
    return {
        "sft_model_path": SFT_MODEL_PATH,
        "ref_model_path": REF_MODEL_PATH,
        "dataset_path": DATASET_PATH,
        "output_dir": OUTPUT_DIR,
        "max_length": MAX_LENGTH,
        "batch_size": 1,
        "gradient_accumulation_steps": 32,
        "learning_rate": 1e-5,
        "max_steps": 50,  # Nombre total d'étapes (ajustable)
        "warmup_steps": 5,
        "beta": 0.1,  # Coefficient DPO
        "lora_r": 8,  # Réduit pour économiser la mémoire
        "lora_alpha": 16,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "eval_steps": 10,
        "save_steps": 5,
        "logging_steps": 1,
        "eval_size": 100,
        "seed": 42
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

def compute_loss(policy_model, ref_model, tokenizer, batch, beta, device):
    """
    Calcule la perte DPO pour un lot de données
    """
    # Extraire les données
    instruction = batch["instruction"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]
    
    # Tokeniser les séquences complètes
    chosen_seqs = [f"{instruction}\n{chosen}" for instruction, chosen in zip(instruction, chosen)]
    rejected_seqs = [f"{instruction}\n{rejected}" for instruction, rejected in zip(instruction, rejected)]
    
    # Tokeniser avec truncation pour contrôler la longueur
    chosen_tokens = tokenizer(
        chosen_seqs, padding=True, truncation=True, max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)
    
    rejected_tokens = tokenizer(
        rejected_seqs, padding=True, truncation=True, max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)
    
    # Forward pass pour policy model (avec gradients)
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        chosen_policy_output = policy_model(**chosen_tokens)
        chosen_policy_logits = chosen_policy_output.logits
        
        # Libération mémoire intermédiaire
        free_memory()
        
        rejected_policy_output = policy_model(**rejected_tokens)
        rejected_policy_logits = rejected_policy_output.logits
    
    # Forward pass pour reference model (sans gradients)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            chosen_ref_output = ref_model(**chosen_tokens)
            chosen_ref_logits = chosen_ref_output.logits
            
            # Libération mémoire intermédiaire
            free_memory()
            
            rejected_ref_output = ref_model(**rejected_tokens)
            rejected_ref_logits = rejected_ref_output.logits
    
    # Libérer mémoire après forward passes
    free_memory()
    
    # Calculer les log probs
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
    
    # Calculer le ratio d'importance
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps
    
    # Calculer les logits pour la perte DPO
    logits = beta * (chosen_ratio - rejected_ratio)
    
    # Perte sigmoid
    loss = -F.logsigmoid(logits).mean()
    
    # Métriques supplémentaires
    chosen_rewards = chosen_ratio.detach()
    rejected_rewards = rejected_ratio.detach()
    reward_acc = (chosen_rewards > rejected_rewards).float().mean()
    
    metrics = {
        "loss": loss.item(),
        "reward_acc": reward_acc.item(),
        "chosen_rewards": chosen_rewards.mean().item(),
        "rejected_rewards": rejected_rewards.mean().item(),
    }
    
    return loss, metrics

def train_dpo():
    """Fonction d'entraînement DPO principale"""
    try:
        config = get_config()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configurer torch pour maximiser la stabilité
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print("\n=== Configuration DPO ===")
        print(f"Modèle SFT: {config['sft_model_path']}")
        print(f"Modèle référence: {config['ref_model_path']}")
        print(f"Beta: {config['beta']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
        print(f"Longueur max de séquence: {config['max_length']}")
        print(f"Nombre maximum d'étapes: {config['max_steps']}")
        print(f"LoRA r: {config['lora_r']}, alpha: {config['lora_alpha']}")
        print(f"Device: {device}")
        
        # Vérification de la mémoire GPU disponible
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"Mémoire GPU totale: {total_mem:.2f} GB")
        
        # Chargement des modèles
        print("\n1. Chargement tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config['ref_model_path'], trust_remote_code=True)
        
        # Chargement du modèle policy
        print("\n2. Chargement du modèle policy...")
        policy_model = AutoModelForCausalLM.from_pretrained(
            config['sft_model_path'],
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Configuration LoRA
        print("\n3. Configuration LoRA...")
        peft_config = LoraConfig(
            r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            target_modules=config['lora_target_modules'],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        policy_model = get_peft_model(policy_model, peft_config)
        policy_model.print_trainable_parameters()
        
        # Chargement du modèle référence
        print("\n4. Chargement du modèle référence...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            config['ref_model_path'],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        ref_model.eval()  # Référence toujours en mode évaluation
        
        free_memory()
        
        # Chargement des données
        print(f"\n5. Chargement dataset depuis {config['dataset_path']}...")
        dataset_full = load_from_disk(config['dataset_path'])
        
        train_dataset = dataset_full["train"]
        eval_dataset_full = dataset_full["validation"]
        eval_size = min(config['eval_size'], len(eval_dataset_full))
        eval_dataset = eval_dataset_full.select(range(eval_size))
        
        print(f"\nStatistiques datasets:")
        print(f"- Train: {len(train_dataset)} exemples")
        print(f"- Eval: {len(eval_dataset)} exemples")
        
        # DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # Optimiseur
        optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=config['learning_rate'],
        )
        
        # Learning rate scheduler
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['max_steps'],
        )
        
        # Initialisation des variables d'entraînement
        policy_model.train()
        accumulated_loss = 0
        global_step = 0
        start_time = time.time()
        
        # Créer dossier de sortie
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Sauvegarde config DPO
        with open(output_dir / "dpo_config.json", 'w') as f:
            json.dump(config, f, indent=4)
        
        # Sauvegarde tokenizer
        tokenizer.save_pretrained(output_dir)
        
        print("\n=== DÉBUT ENTRAÎNEMENT DPO ===")
        
        # Boucle d'entraînement
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training", total=config['max_steps'])):
            # Limiter le nombre d'étapes
            if global_step >= config['max_steps']:
                break
            
            try:
                # Afficher l'étape actuelle
                print(f"\n=== Étape {global_step + 1}/{config['max_steps']} ===")
                free_memory()
                
                # Calculer la perte
                loss, metrics = compute_loss(
                    policy_model, ref_model, tokenizer, batch, config['beta'], device
                )
                
                # Normaliser la perte pour l'accumulation de gradients
                loss = loss / config['gradient_accumulation_steps']
                
                # Rétropropagation
                loss.backward()
                
                # Libérer mémoire après backward
                free_memory()
                
                # Accumuler la perte pour le logging
                accumulated_loss += loss.item()
                
                # Mise à jour des poids
                if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0 or batch_idx == len(train_dataloader) - 1:
                    # Clip des gradients pour stabiliser l'entraînement
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                    
                    # Update des poids
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # Incrémenter le compteur global
                    global_step += 1
                    
                    # Logging des métriques
                    if global_step % config['logging_steps'] == 0:
                        avg_loss = accumulated_loss / config['gradient_accumulation_steps']
                        
                        # Afficher les métriques
                        print(f"Step {global_step} | Loss: {avg_loss:.5f} | "
                              f"Reward Acc: {metrics['reward_acc']:.4f} | "
                              f"LR: {lr_scheduler.get_last_lr()[0]:.8f}")
                        
                        accumulated_loss = 0
                    
                    # Sauvegarde périodique
                    if global_step % config['save_steps'] == 0:
                        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                        checkpoint_dir.mkdir(exist_ok=True, parents=True)
                        
                        print(f"\nSauvegarde du checkpoint {global_step}...")
                        policy_model.save_pretrained(checkpoint_dir)
                        
                        # Sauvegarder métriques
                        with open(checkpoint_dir / "metrics.json", 'w') as f:
                            json.dump({
                                "step": global_step,
                                "loss": avg_loss,
                                "reward_acc": metrics['reward_acc'],
                                "chosen_rewards": metrics['chosen_rewards'],
                                "rejected_rewards": metrics['rejected_rewards'],
                            }, f, indent=4)
                        
                        print(f"Checkpoint {global_step} sauvegardé avec succès")
                
                # Libérer mémoire à la fin de chaque étape
                free_memory()
                
            except Exception as e:
                import traceback
                print(f"\n⚠️ Erreur durant l'étape {global_step}: {str(e)}")
                traceback.print_exc()
                print("Tentative de continuer avec la prochaine étape...")
                optimizer.zero_grad()
                free_memory()
                continue
        
        # Sauvegarde finale
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True, parents=True)
        
        print("\nSauvegarde du modèle final...")
        policy_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        # Durée totale
        total_time = time.time() - start_time
        print(f"\n=== ENTRAÎNEMENT TERMINÉ ===")
        print(f"Étapes complétées: {global_step}/{config['max_steps']}")
        print(f"Durée totale: {total_time:.2f} secondes")
        print(f"Modèle sauvegardé dans: {final_dir}")
        
        return config
        
    except Exception as e:
        import traceback
        print("\n==== ERREUR CRITIQUE DURANT L'ENTRAÎNEMENT ====")
        print(f"Type d'erreur: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nTraceback complet:")
        traceback.print_exc()
        
        if torch.cuda.is_available():
            print("\nInformations GPU:")
            print(f"Mémoire allouée: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"Mémoire réservée: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        print("\nEntraînement arrêté prématurément. Vérifiez les checkpoints intermédiaires.")

def main():
    """Fonction principale"""
    print("\n=== SCRIPT DPO - VERSION FINALE ===")
    config = train_dpo()
    
    print("\nPour le mentorat, vous pouvez démontrer:")
    print("1. Le modèle de base (Qwen/Qwen3-1.7B)")
    print(f"2. Le modèle SFT (sft_final)")
    print(f"3. Le modèle DPO ({OUTPUT_DIR}/final)")
    
    print("\nUtilisez scripts/compare_models.py pour comparer les résultats.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n==== ERREUR CRITIQUE ====")
        print(f"Type d'erreur: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nTraceback complet:")
        traceback.print_exc()
        
        if torch.cuda.is_available():
            print("\nInformations GPU:")
            print(f"Mémoire allouée: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"Mémoire réservée: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
