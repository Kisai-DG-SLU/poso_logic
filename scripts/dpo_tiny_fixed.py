"""
Solution DPO ultra simplifiée corrigée - contourne complètement les API de TRL
et implémente directement l'algorithme DPO
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

# Chemin des dossiers
PROCESSED_DIR = Path("/mnt/prod/data/processed")
MODEL_DIR = Path("/mnt/prod/models")
SFT_CHECKPOINT = MODEL_DIR / "checkpoints" / "sft_final"

def get_config():
    """Configuration DPO simplifiée"""
    return {
        "model_name": "Qwen/Qwen3-1.7B",
        "max_seq_length": 512,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
        "beta": 0.1,                 # Coefficient DPO
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "output_dir": str(MODEL_DIR / "checkpoints" / "dpo_tiny"),
        "max_samples": 20,           # Limiter les samples pour test
        "max_steps": 5,              # Nombre d'étapes d'entraînement
        "warmup_steps": 1,
        "log_freq": 1,               # Frequency de log
    }

def compute_policy_logps(model, tokenizer, batch, device):
    """
    Calcule les log probabilités pour chosen et rejected
    """
    # Extraire les données
    instruction = batch["instruction"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]
    
    # Tokeniser les séquence complètes
    chosen_seqs = [f"{instruction}\n{chosen}" for instruction, chosen in zip(instruction, chosen)]
    rejected_seqs = [f"{instruction}\n{rejected}" for instruction, rejected in zip(instruction, rejected)]
    
    # Tokeniser
    chosen_tokens = tokenizer(
        chosen_seqs, padding=True, truncation=True, max_length=512,
        return_tensors="pt"
    ).to(device)
    
    rejected_tokens = tokenizer(
        rejected_seqs, padding=True, truncation=True, max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Forward pass pour chosen
    chosen_output = model(**chosen_tokens)
    chosen_logits = chosen_output.logits
        
    # Forward pass pour rejected
    rejected_output = model(**rejected_tokens)
    rejected_logits = rejected_output.logits
    
    # Calculer les log probs pour chosen
    chosen_logps = compute_logps(
        chosen_logits, chosen_tokens["input_ids"],
        chosen_tokens["attention_mask"]
    )
    
    # Calculer les log probs pour rejected
    rejected_logps = compute_logps(
        rejected_logits, rejected_tokens["input_ids"],
        rejected_tokens["attention_mask"]
    )
    
    return chosen_logps, rejected_logps

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

def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps, 
                     reference_chosen_logps, reference_rejected_logps, 
                     beta):
    """
    Calcule la perte DPO
    """
    # Calculer le ratio d'importance pour chosen et rejected
    chosen_ratio = policy_chosen_logps - reference_chosen_logps
    rejected_ratio = policy_rejected_logps - reference_rejected_logps
    
    # Calculer les logits pour la perte DPO
    logits = beta * (chosen_ratio - rejected_ratio)
    
    # Perte sigmoid
    loss = -F.logsigmoid(logits).mean()
    
    return loss

def train_dpo_tiny():
    """
    Implémentation simplifiée de DPO
    """
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=== Configuration DPO Tiny ===")
    print(f"Modèle: {config['model_name']}")
    print(f"Beta: {config['beta']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"Max samples: {config['max_samples']}")
    print(f"Max steps: {config['max_steps']}")
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
        # Charger le dataset
        train_dataset_full = load_from_disk(str(processed_dir / "train"))
        
        # Limiter le nombre d'exemples
        max_samples = config['max_samples']
        train_dataset = train_dataset_full.select(range(min(max_samples, len(train_dataset_full))))
        
        # Utiliser un simple DataLoader
        data_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True
        )
        
        print(f"\nStatistiques dataset:")
        print(f"- Train original: {len(train_dataset_full)} exemples")
        print(f"- Train réduit: {len(train_dataset)} exemples")
    else:
        raise ValueError(f"Dataset introuvable: {processed_dir}")
    
    # Optimizer et scheduler
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    # Simple linear warmup, constant learning rate afterwards
    num_steps = min(config['max_steps'], len(data_loader) * config['max_samples'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_steps
    )
    
    # Fonction pour évaluation
    def evaluate_step(policy_model, ref_model, batch, device, config):
        """Évalue la perte DPO sur un lot"""
        # Batch sur device
        batch = {k: v for k, v in batch.items() if k in ["instruction", "chosen", "rejected"]}
        
        # Calculer les logps pour policy
        with torch.set_grad_enabled(True):
            policy_chosen_logps, policy_rejected_logps = compute_policy_logps(
                policy_model, tokenizer, batch, device
            )
        
        # Calculer les logps pour reference sans gradients
        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps = compute_policy_logps(
                ref_model, tokenizer, batch, device
            )
        
        # Détacher les tenseurs de référence du graphe de calcul
        ref_chosen_logps = ref_chosen_logps.detach()
        ref_rejected_logps = ref_rejected_logps.detach()
        
        # Calculer la perte DPO
        loss = compute_dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            config['beta']
        )
        
        return loss
    
    # Boucle d'entraînement
    print("\n=== DÉBUT ENTRAÎNEMENT DPO ===")
    
    policy_model.train()
    ref_model.eval()
    
    step = 0
    accumulation_steps = config['gradient_accumulation_steps']
    progress_bar = tqdm(total=config['max_steps'], desc="Training")
    
    for epoch in range(1):
        for batch_idx, batch in enumerate(data_loader):
            # Limite de steps
            if step >= config['max_steps']:
                break
            
            # Reset gradients à zéro
            optimizer.zero_grad()
            
            # Forward pass et calcul de la perte
            loss = evaluate_step(policy_model, ref_model, batch, device, config)
            
            # Normaliser la perte pour l'accumulation de gradient
            loss = loss / accumulation_steps
            
            # Backpropagation
            loss.backward()
            
            # Mise à jour des poids
            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(data_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Log
                if step % config['log_freq'] == 0:
                    progress_bar.set_description(f"Step {step}, Loss: {loss.item() * accumulation_steps:.5f}")
                
                step += 1
                progress_bar.update(1)
                
                # Limite de steps
                if step >= config['max_steps']:
                    break
    
    # Sauvegarder le modèle
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSauvegarde du modèle dans: {output_dir}")
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Sauvegarder la config
    with open(output_dir / "dpo_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Entraînement DPO Tiny terminé avec succès!")
    return config

if __name__ == "__main__":
    train_dpo_tiny()
