"""
Script d'entraînement DPO ultra-optimisé pour GPU A2 (15GB)
Objectif : Entraîner sur l'ensemble du dataset de manière réaliste
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
from datasets import load_from_disk
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
import math

# Chemins
PROCESSED_DIR = Path("/mnt/prod/data/processed")
MODEL_DIR = Path("/mnt/prod/models")
OUTPUT_DIR = MODEL_DIR / "checkpoints" / "dpo_a2_optimized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration ultra-optimisée pour A2
config = {
    "model_name": "Qwen/Qwen3-1.7B",
    "max_seq_length": 128,      # Drastiquement réduit
    "batch_size": 1,
    "gradient_accumulation_steps": 8,  # Réduit de 64 à 8
    "learning_rate": 1e-6,      # Plus conservateur
    "num_train_epochs": 1,
    "beta": 0.1,
    "lora_r": 2,               # Très réduit (de 4 à 2)
    "lora_alpha": 8,           # Ajusté en conséquence
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj"],  # Seulement 2 modules
    "warmup_steps": 100,
    "save_every_n_steps": 500,
    "eval_steps": 1000,
    "max_steps": None,
    "seed": 42,
    "fp16": True,              # Mixed precision
    "gradient_checkpointing": True,
    "optim": "adamw_torch_fused",  # Optimiseur fusionné plus rapide
}

def print_gpu_utilization():
    """Afficher l'utilisation GPU de manière concise"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {allocated:.1f}/{reserved:.1f}GB"
    return "GPU: N/A"

def clear_cache():
    """Nettoyer le cache GPU de manière minimale"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def compute_logps(logits, input_ids, attention_mask):
    """Calcul optimisé des log probabilités"""
    with torch.cuda.amp.autocast():
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        token_logprobs = torch.gather(
            F.log_softmax(shift_logits, dim=-1),
            dim=-1,
            index=shift_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        token_logprobs = token_logprobs * shift_mask
        return token_logprobs.sum(dim=-1)

def compute_dpo_loss_optimized(
    policy_model, ref_model, batch, tokenizer, 
    beta, device, max_length, scaler
):
    """Calcul de perte DPO optimisé avec mixed precision"""
    # Préparation des séquences
    instruction = batch["instruction"][0]
    chosen = batch["chosen"][0]
    rejected = batch["rejected"][0]
    
    # Tokenisation unique
    chosen_seq = f"{instruction}\n{chosen}"
    rejected_seq = f"{instruction}\n{rejected}"
    
    # Limiter encore plus la longueur si nécessaire
    if len(chosen_seq) > 200:
        chosen_seq = chosen_seq[:200]
    if len(rejected_seq) > 200:
        rejected_seq = rejected_seq[:200]
    
    # Tokenizer avec longueur fixe
    tokens_chosen = tokenizer(
        chosen_seq, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt"
    ).to(device)
    
    tokens_rejected = tokenizer(
        rejected_seq, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt"
    ).to(device)
    
    # Mixed precision pour policy forward
    with torch.cuda.amp.autocast():
        chosen_policy_out = policy_model(**tokens_chosen)
        rejected_policy_out = policy_model(**tokens_rejected)
    
    # Ref model forward (sans gradients)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            chosen_ref_out = ref_model(**tokens_chosen)
            rejected_ref_out = ref_model(**tokens_rejected)
    
    # Calcul des log probs
    policy_chosen_logps = compute_logps(
        chosen_policy_out.logits, 
        tokens_chosen.input_ids,
        tokens_chosen.attention_mask
    )
    
    policy_rejected_logps = compute_logps(
        rejected_policy_out.logits,
        tokens_rejected.input_ids,
        tokens_rejected.attention_mask
    )
    
    ref_chosen_logps = compute_logps(
        chosen_ref_out.logits,
        tokens_chosen.input_ids,
        tokens_chosen.attention_mask
    )
    
    ref_rejected_logps = compute_logps(
        rejected_ref_out.logits,
        tokens_rejected.input_ids,
        tokens_rejected.attention_mask
    )
    
    # DPO loss
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -F.logsigmoid(logits).mean()
    
    # Métriques
    reward_acc = (chosen_ratio > rejected_ratio).float().mean().item()
    
    return loss, {
        "loss": loss.item(),
        "reward_acc": reward_acc,
        "chosen_ratio": chosen_ratio.mean().item(),
        "rejected_ratio": rejected_ratio.mean().item()
    }

def train_dpo_optimized():
    """Entraînement DPO optimisé pour A2"""
    print("=== Configuration DPO Optimisée pour A2 ===")
    for k, v in config.items():
        print(f"{k}: {v}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True  # Optimisation CUDA
    
    # Tokenizer
    print("\nChargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Modèles
    print("Chargement modèles...")
    print(f"Mémoire avant: {print_gpu_utilization()}")
    
    # Policy model avec gradient checkpointing
    policy_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if config['gradient_checkpointing']:
        policy_model.gradient_checkpointing_enable()
    
    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    ref_model.eval()
    
    print(f"Mémoire après chargement: {print_gpu_utilization()}")
    
    # LoRA configuration minimale
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=config['target_modules'],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.print_trainable_parameters()
    
    # Dataset
    print("\nChargement dataset...")
    train_dataset = load_from_disk(str(PROCESSED_DIR / "dpo_dataset_trl" / "train"))
    print(f"Dataset: {len(train_dataset)} exemples")
    
    # DataLoader simple
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # Optimiseur avec fused optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
        fused=True  # Optimiseur fusionné CUDA
    )
    
    # Scheduler
    num_training_steps = len(train_loader) // config['gradient_accumulation_steps']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    print("\n=== DÉBUT ENTRAÎNEMENT OPTIMISÉ ===")
    print(f"Steps par epoch: {num_training_steps}")
    print(f"Temps estimé par epoch: {num_training_steps * 10 / 3600:.1f} heures (si 10s/step)")
    
    # Boucle d'entraînement
    global_step = 0
    accumulated_loss = 0
    start_time = time.time()
    
    policy_model.train()
    optimizer.zero_grad()
    
    # Progress bar
    pbar = tqdm(total=num_training_steps, desc="Training")
    
    for step, batch in enumerate(train_loader):
        # Calcul de la perte avec mixed precision
        loss, metrics = compute_dpo_loss_optimized(
            policy_model, ref_model, batch, tokenizer,
            config['beta'], device, config['max_seq_length'], scaler
        )
        
        # Normalisation pour gradient accumulation
        loss = loss / config['gradient_accumulation_steps']
        
        # Backward avec mixed precision
        scaler.scale(loss).backward()
        accumulated_loss += loss.item()
        
        # Update weights
        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Logging
            if global_step % 10 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                eta_seconds = (num_training_steps - global_step) / steps_per_sec
                eta_hours = eta_seconds / 3600
                
                pbar.set_description(
                    f"Loss: {accumulated_loss:.4f} | "
                    f"Acc: {metrics['reward_acc']:.2f} | "
                    f"Speed: {steps_per_sec:.1f} steps/s | "
                    f"ETA: {eta_hours:.1f}h | "
                    f"{print_gpu_utilization()}"
                )
                accumulated_loss = 0
            
            # Checkpoint
            if global_step % config['save_every_n_steps'] == 0:
                checkpoint_path = OUTPUT_DIR / f"checkpoint-{global_step}"
                checkpoint_path.mkdir(exist_ok=True, parents=True)
                
                policy_model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                # Sauvegarder les statistiques
                stats = {
                    "global_step": global_step,
                    "elapsed_time": elapsed,
                    "steps_per_second": steps_per_sec,
                    "estimated_total_hours": num_training_steps / steps_per_sec / 3600,
                    "last_metrics": metrics
                }
                with open(checkpoint_path / "training_stats.json", "w") as f:
                    json.dump(stats, f, indent=2)
                
                print(f"\n✓ Checkpoint sauvegardé: {checkpoint_path}")
            
            pbar.update(1)
            
            # Clear cache occasionnellement
            if global_step % 100 == 0:
                clear_cache()
            
            # Limite optionnelle de steps
            if config['max_steps'] and global_step >= config['max_steps']:
                break
    
    pbar.close()
    
    # Sauvegarde finale
    print("\nSauvegarde du modèle final...")
    final_path = OUTPUT_DIR / "final"
    final_path.mkdir(exist_ok=True, parents=True)
    
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Statistiques finales
    total_time = time.time() - start_time
    print(f"\n✅ Entraînement terminé!")
    print(f"Temps total: {total_time/3600:.1f} heures")
    print(f"Steps totaux: {global_step}")
    print(f"Vitesse moyenne: {global_step/total_time:.1f} steps/sec")
    
    with open(OUTPUT_DIR / "final_results.json", "w") as f:
        json.dump({
            "total_time_hours": total_time / 3600,
            "total_steps": global_step,
            "average_speed_steps_per_sec": global_step / total_time,
            "config": config
        }, f, indent=2)
    
    return OUTPUT_DIR

if __name__ == "__main__":
    output_path = train_dpo_optimized()
    print(f"\nModèle disponible dans: {output_path}/final")