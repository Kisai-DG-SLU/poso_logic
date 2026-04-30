"""
Script d'entraînement DPO minimal et robuste pour Qwen3-1.7B
Conçu pour tester l'essentiel de la solution sur un petit échantillon
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.disable(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from pathlib import Path
import json
import time
import gc
from tqdm import tqdm

# Chemin des dossiers
PROCESSED_DIR = Path("/mnt/prod/data/processed")
MODEL_DIR = Path("/mnt/prod/models")
SFT_CHECKPOINT = MODEL_DIR / "checkpoints" / "sft_final"
LOGS_DIR = Path("/mnt/prod/logs")
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# Configuration
config = {
    "model_name": "Qwen/Qwen3-1.7B", 
    "max_seq_length": 256,              # Réduit au minimum
    "batch_size": 1,                    # Batch minimal
    "gradient_accumulation_steps": 8,   # Accumulation réduite pour le test
    "learning_rate": 2e-6,
    "num_train_epochs": 1,
    "beta": 0.1,                        # Coefficient DPO standard
    "lora_r": 4,                        # Rang LoRA minimal
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], # Minimum viable
    "max_steps": 10,                    # Limité pour test
    "output_dir": str(MODEL_DIR / "checkpoints" / "dpo_minimal"),
    "sample_size": 100                  # Taille de l'échantillon pour test
}

# Fonction pour libérer la mémoire GPU
def free_memory():
    """Libérer la mémoire GPU de manière agressive"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Force collecte mémoire Python
    for _ in range(3):
        gc.collect()
        
    # Affiche l'utilisation mémoire
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"Mémoire GPU: {allocated:.2f} GB allouée / {reserved:.2f} GB réservée")

# Calcul des log probabilités
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

# Calcul de la perte DPO
def compute_loss(policy_model, ref_model, tokenizer, batch, beta, device):
    """Calcule la perte DPO pour un lot de données"""
    try:
        # Extraire les données
        instruction = batch["instruction"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        
        # Tokeniser les séquences en réduisant la longueur pour économiser la mémoire
        max_length = config["max_seq_length"]
        
        # Tokeniser les séquences complètes
        chosen_seqs = [f"{inst}\n{ch}" for inst, ch in zip(instruction, chosen)]
        rejected_seqs = [f"{inst}\n{rej}" for inst, rej in zip(instruction, rejected)]
        
        # Tokeniser
        chosen_tokens = tokenizer(
            chosen_seqs, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        rejected_tokens = tokenizer(
            rejected_seqs, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        # Forward pass pour policy model (avec gradients)
        with torch.set_grad_enabled(True):
            chosen_policy_output = policy_model(**chosen_tokens)
            chosen_policy_logits = chosen_policy_output.logits
            
            # Libérer mémoire
            free_memory()
            
            rejected_policy_output = policy_model(**rejected_tokens)
            rejected_policy_logits = rejected_policy_output.logits
        
        # Forward pass pour reference model (sans gradients)
        with torch.no_grad():
            chosen_ref_output = ref_model(**chosen_tokens)
            chosen_ref_logits = chosen_ref_output.logits
            
            # Libérer mémoire
            free_memory()
            
            rejected_ref_output = ref_model(**rejected_tokens)
            rejected_ref_logits = rejected_ref_output.logits
        
        # Libérer mémoire
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
    except Exception as e:
        print(f"Erreur dans compute_loss: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

# Fonction principale d'entraînement DPO
def train_dpo_minimal():
    """Fonction d'entraînement DPO minimale"""
    print("=== Configuration DPO Minimale ===")
    for k, v in config.items():
        print(f"{k}: {v}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Mémoire GPU totale: {total_mem:.2f} GB")
    
    # Tokenizer
    print("\nChargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'], 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Chargement modèle
    print("\nChargement du modèle policy...")
    try:
        policy_model = AutoModelForCausalLM.from_pretrained(
            config['model_name'], 
            torch_dtype=torch.float16,  # Utiliser float16 au lieu de bfloat16
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        print("Tentative avec options réduites...")
        policy_model = AutoModelForCausalLM.from_pretrained(
            config['model_name'], 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Modèle de référence
    print("\nChargement du modèle de référence (identique à policy)...")
    ref_model = policy_model  # Utiliser le même modèle pour économiser la mémoire
    
    # S'assurer que ref_model est en eval mode et ses paramètres sont gelés
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # LoRA pour modèle policy
    print("\nConfiguration LoRA...")
    lora_config = LoraConfig(
        r=config['lora_r'], 
        lora_alpha=config['lora_alpha'],
        target_modules=config['target_modules'], 
        lora_dropout=config['lora_dropout'], 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    
    policy_model = get_peft_model(policy_model, lora_config)
    
    # Mettre policy_model en mode train
    policy_model.train()
    
    print(f"\nParamètres entraînables: {policy_model.print_trainable_parameters()}")
    
    # Chargement dataset
    print("\nChargement dataset...")
    dataset_dir = PROCESSED_DIR / "dpo_dataset_trl" / "train"
    try:
        train_dataset = datasets.load_from_disk(str(dataset_dir))
        # Échantillon pour test
        sample_size = min(config["sample_size"], len(train_dataset))
        train_dataset = train_dataset.select(range(sample_size))
        print(f"Dataset chargé: {len(train_dataset)} exemples")
    except Exception as e:
        print(f"Erreur chargement dataset: {e}")
        raise e
    
    # DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,  # Sans multiprocessing
        pin_memory=False
    )
    
    # Créer le répertoire de sortie
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(config['max_steps'] * 0.1)),
        num_training_steps=config['max_steps'],
    )
    
    # Entraînement
    print("\n=== DÉBUT ENTRAÎNEMENT DPO ===")
    global_step = 0
    accumulated_loss = 0
    
    # Portez les modèles sur le device
    policy_model.to(device)
    if ref_model != policy_model:  # Seulement si ce sont deux modèles différents
        ref_model.to(device)
    
    # Boucle d'entraînement
    progress_bar = tqdm(total=config['max_steps'], desc="Training")
    
    try:
        for batch in train_dataloader:
            if global_step >= config['max_steps']:
                break
                
            # Libérer la mémoire avant calcul
            free_memory()
            
            # Calculer la perte
            loss, metrics = compute_loss(
                policy_model, ref_model, tokenizer, batch, config['beta'], device
            )
            
            # Normaliser la perte
            loss = loss / config['gradient_accumulation_steps']
            
            # Rétropropagation
            loss.backward()
            
            # Accumuler la perte
            accumulated_loss += loss.item()
            
            # Mise à jour des poids
            if (global_step + 1) % config['gradient_accumulation_steps'] == 0:
                # Clip des gradients
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                
                # Mise à jour
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Affichage
                print(f"\nStep {global_step+1} | "
                      f"Loss: {accumulated_loss:.5f} | "
                      f"Reward Acc: {metrics['reward_acc']:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.8f}")
                
                # Réinitialiser la perte
                accumulated_loss = 0
                
                # Sauvegarde intermédiaire tous les 5 steps
                if (global_step + 1) % 5 == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step+1}"
                    checkpoint_dir.mkdir(exist_ok=True, parents=True)
                    
                    policy_model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    print(f"Sauvegarde intermédiaire: {checkpoint_dir}")
            
            # Incrémenter le compteur
            global_step += 1
            progress_bar.update(1)
            
    except KeyboardInterrupt:
        print("\nInterruption manuelle. Sauvegarde du modèle...")
    except Exception as e:
        print(f"\nErreur: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        progress_bar.close()
        
        # Sauvegarde finale
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            print(f"\nSauvegarde du modèle final dans {final_dir}...")
            policy_model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            
            # Sauvegarde config
            with open(output_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
                
            print("\n✅ ENTRAÎNEMENT MINIMAL TERMINÉ!")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde finale: {e}")

if __name__ == "__main__":
    train_dpo_minimal()