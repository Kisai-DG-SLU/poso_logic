"""
Script d'entraînement DPO production - implémentation personnalisée
qui contourne les problèmes d'incompatibilité TRL/transformers
"""
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.disable(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    get_linear_schedule_with_warmup, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from pathlib import Path
import json
from tqdm import tqdm
import time

# Chemin des dossiers
PROCESSED_DIR = Path("/mnt/prod/data/processed")
MODEL_DIR = Path("/mnt/prod/models")
SFT_CHECKPOINT = MODEL_DIR / "checkpoints" / "sft_final"

def get_dpo_config():
    """Configuration DPO complète"""
    return {
        "model_name": "Qwen/Qwen3-1.7B",
        "max_seq_length": 1024,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-5,
        "num_train_epochs": 1,
        "beta": 0.1,                 # Coefficient DPO
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "output_dir": str(MODEL_DIR / "checkpoints" / "dpo_final"),
        "save_steps": 500,
        "logging_steps": 25,
        "eval_steps": 500,
        "warmup_steps": 100,
        "save_total_limit": 2,
        "eval_size": 1000,           # Nombre d'exemples d'évaluation
        "max_steps": None,           # Si None, utilise num_train_epochs
        "checkpoint_dir": str(MODEL_DIR / "checkpoints" / "dpo_final"),
    }

def compute_logps(logits, input_ids, attention_mask):
    """
    Calcule les log probabilités pour les séquences
    """
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
    
    # Sommer les log probs (gérer les lots)
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
    
    # Tokeniser
    chosen_tokens = tokenizer(
        chosen_seqs, padding=True, truncation=True, max_length=1024,
        return_tensors="pt"
    ).to(device)
    
    rejected_tokens = tokenizer(
        rejected_seqs, padding=True, truncation=True, max_length=1024,
        return_tensors="pt"
    ).to(device)
    
    # Forward pass pour policy model (avec gradients)
    with torch.set_grad_enabled(True):
        chosen_policy_output = policy_model(**chosen_tokens)
        chosen_policy_logits = chosen_policy_output.logits
        
        rejected_policy_output = policy_model(**rejected_tokens)
        rejected_policy_logits = rejected_policy_output.logits
    
    # Forward pass pour reference model (sans gradients)
    with torch.no_grad():
        chosen_ref_output = ref_model(**chosen_tokens)
        chosen_ref_logits = chosen_ref_output.logits
        
        rejected_ref_output = ref_model(**rejected_tokens)
        rejected_ref_logits = rejected_ref_output.logits
    
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
    """
    Fonction d'entraînement DPO principale
    """
    config = get_dpo_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=== Configuration DPO ===")
    print(f"Modèle: {config['model_name']}")
    print(f"Beta: {config['beta']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"Nombre d'exemples d'évaluation: {config['eval_size']}")
    print(f"Device: {device}")
    
    # Tokenizer
    print("\nChargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Modèle policy (modèle à entraîner, utilise SFT comme départ)
    model_path = str(SFT_CHECKPOINT) if SFT_CHECKPOINT.exists() else config['model_name']
    print(f"\nChargement modèle policy depuis: {model_path}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    
    # Modèle de référence (frozen, utilise base model)
    print("\nChargement modèle référence...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    
    # S'assurer que ref_model est en eval mode et ses paramètres sont gelés
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # LoRA pour modèle policy
    print("Configuration LoRA pour le modèle policy...")
    lora_config = LoraConfig(
        r=config['lora_r'], lora_alpha=config['lora_alpha'],
        target_modules=config['target_modules'], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    policy_model = get_peft_model(policy_model, lora_config)
    
    # S'assurer que policy_model est bien en mode train
    policy_model.train()
    
    print(f"Paramètres entraînables: {policy_model.print_trainable_parameters()}")
    
    # Dataset DPO
    processed_dir = PROCESSED_DIR / "dpo_dataset_trl"
    print(f"\nChargement dataset depuis {processed_dir}...")
    
    if processed_dir.exists():
        # Charger les datasets
        train_dataset = load_from_disk(str(processed_dir / "train"))
        eval_dataset = load_from_disk(str(processed_dir / "eval"))
        
        # Limiter le nombre d'exemples d'évaluation
        eval_size = min(config['eval_size'], len(eval_dataset))
        eval_dataset = eval_dataset.select(range(eval_size))
        
        # DataLoader
        train_dataloader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True,
            num_workers=1, pin_memory=True
        )
        
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=1, pin_memory=True
        )
        
        print(f"\nStatistiques datasets:")
        print(f"- Train: {len(train_dataset)} exemples")
        print(f"- Eval: {len(eval_dataset)} exemples")
    else:
        raise ValueError(f"Dataset introuvable: {processed_dir}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Scheduler
    if config['max_steps'] is not None:
        num_training_steps = config['max_steps']
    else:
        num_training_steps = len(train_dataloader) * config['num_train_epochs'] // config['gradient_accumulation_steps']
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_training_steps,
    )
    
    # Fonction d'évaluation
    def evaluate(policy_model, ref_model, eval_dataloader, beta, device):
        policy_model.eval()
        total_loss = 0
        total_reward_acc = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                # Calculer la perte sur ce lot
                _, metrics = compute_loss(
                    policy_model, ref_model, tokenizer, batch, beta, device
                )
                
                batch_size = 1  # Fixé à 1 dans notre cas
                total_loss += metrics["loss"] * batch_size
                total_reward_acc += metrics["reward_acc"] * batch_size
                total_samples += batch_size
                
                # Limiter la durée d'évaluation
                if batch_idx >= 100:
                    break
        
        # Calculer les moyennes
        avg_loss = total_loss / total_samples
        avg_reward_acc = total_reward_acc / total_samples
        
        return {
            "eval_loss": avg_loss,
            "eval_reward_acc": avg_reward_acc,
        }
    
    # Ensurer que le dossier output existe
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # État de l'entraînement
    state = {
        "best_eval_loss": float("inf"),
        "step": 0,
        "epoch": 0,
    }
    
    # Boucle d'entraînement
    print("\n=== DÉBUT ENTRAÎNEMENT DPO ===")
    
    accumulated_loss = 0
    global_step = 0
    
    start_time = time.time()
    log_history = []
    
    epochs = config['num_train_epochs'] if config['max_steps'] is None else 1
    
    for epoch in range(epochs):
        state["epoch"] = epoch
        
        print(f"\nÉpoque {epoch+1}/{epochs}")
        policy_model.train()
        
        # Initialiser la barre de progression
        progress_bar = tqdm(total=len(train_dataloader))
        
        # Sauvegarder les métriques à chaque étape pour les graphiques
        for batch_idx, batch in enumerate(train_dataloader):
            # Calculer la perte
            loss, metrics = compute_loss(
                policy_model, ref_model, tokenizer, batch, config['beta'], device
            )
            
            # Normaliser la perte pour l'accumulation de gradients
            loss = loss / config['gradient_accumulation_steps']
            
            # Rétropropagation
            loss.backward()
            
            # Accumuler la perte pour le logging
            accumulated_loss += loss.item()
            
            # Mise à jour des poids
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0 or batch_idx == len(train_dataloader) - 1:
                # Clip des gradients pour stabiliser l'entraînement
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                
                # Mise à jour
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Logging
                global_step += 1
                state["step"] = global_step
                
                # Mise à jour de la barre de progression
                progress_bar.update(config['gradient_accumulation_steps'])
                progress_bar.set_description(
                    f"Step {global_step} | Loss: {accumulated_loss:.5f} | Reward Acc: {metrics['reward_acc']:.4f}"
                )
                
                # Logging détaillé
                if global_step % config['logging_steps'] == 0:
                    avg_loss = accumulated_loss
                    elapsed = time.time() - start_time
                    
                    log_info = {
                        "step": global_step,
                        "loss": avg_loss,
                        "reward_acc": metrics["reward_acc"],
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "seconds_elapsed": elapsed,
                    }
                    
                    log_history.append(log_info)
                    
                    print(f"\nStep {global_step} | Loss: {avg_loss:.5f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.8f} | "
                          f"Reward Acc: {metrics['reward_acc']:.4f} | "
                          f"Chosen reward: {metrics['chosen_rewards']:.4f} | "
                          f"Rejected reward: {metrics['rejected_rewards']:.4f}")
                    
                    # Réinitialiser la perte accumulée
                    accumulated_loss = 0
                
                # Évaluation
                if global_step % config['eval_steps'] == 0:
                    print(f"\nÉvaluation à l'étape {global_step}...")
                    eval_results = evaluate(policy_model, ref_model, eval_dataloader, config['beta'], device)
                    
                    print(f"Eval Loss: {eval_results['eval_loss']:.5f} | "
                          f"Eval Reward Acc: {eval_results['eval_reward_acc']:.4f}")
                    
                    # Sauvegarde si meilleur modèle
                    if eval_results["eval_loss"] < state["best_eval_loss"]:
                        state["best_eval_loss"] = eval_results["eval_loss"]
                        print(f"Nouveau meilleur modèle! Perte d'évaluation: {eval_results['eval_loss']:.5f}")
                        
                        # Sauvegarde
                        best_model_dir = output_dir / "best_model"
                        best_model_dir.mkdir(exist_ok=True, parents=True)
                        policy_model.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)
                        
                        # Sauvegarder l'état
                        with open(best_model_dir / "trainer_state.json", "w") as f:
                            json.dump(state, f, indent=2)
                
                # Sauvegarde régulière
                if global_step % config['save_steps'] == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Sauvegarde modèle
                    policy_model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    # Sauvegarde état et config
                    with open(checkpoint_dir / "trainer_state.json", "w") as f:
                        json.dump(state, f, indent=2)
                    
                    with open(checkpoint_dir / "training_args.json", "w") as f:
                        json.dump(config, f, indent=2)
                
                # Limite d'étapes
                if config['max_steps'] is not None and global_step >= config['max_steps']:
                    progress_bar.close()
                    break
            
            # Mise à jour incrémentale pour les étapes intermédiaires
            else:
                progress_bar.update(1)
        
        # Fermer la barre de progression
        progress_bar.close()
        
        # Arrêter si max_steps atteint
        if config['max_steps'] is not None and global_step >= config['max_steps']:
            print(f"\nNombre maximum d'étapes ({config['max_steps']}) atteint.")
            break
    
    # Évaluation finale
    print("\nÉvaluation finale...")
    final_eval_results = evaluate(policy_model, ref_model, eval_dataloader, config['beta'], device)
    
    print(f"Eval Loss final: {final_eval_results['eval_loss']:.5f} | "
          f"Eval Reward Acc final: {final_eval_results['eval_reward_acc']:.4f}")
    
    # Sauvegarde finale
    final_output_dir = output_dir / "final"
    final_output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nSauvegarde du modèle final dans: {final_output_dir}")
    policy_model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    # Sauvegarder l'historique et la config
    with open(output_dir / "log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)
    
    with open(output_dir / "dpo_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ ENTRAÎNEMENT DPO TERMINÉ AVEC SUCCÈS!")
    return config

def main():
    config = train_dpo()
    print("\n=== INFORMATIONS FINALES ===")
    print(f"- Modèle entraîné sauvegardé dans: {config['output_dir']}")
    print(f"- Configuration sauvegardée dans: {config['output_dir']}/dpo_config.json")
    print(f"- Pour charger le modèle: policy_model = AutoModelForCausalLM.from_pretrained('{config['output_dir']}')")
    
    # Derniers conseils
    print("\n💡 CONSEIL: Pour le mentorat, vous pouvez présenter:")
    print("1. Le modèle SFT base (/mnt/prod/models/checkpoints/sft_final)")
    print("2. Le modèle DPO final (/mnt/prod/models/checkpoints/dpo_final/final)")
    print("3. Les améliorations apportées par DPO sur des exemples médicaux")

if __name__ == "__main__":
    main()
