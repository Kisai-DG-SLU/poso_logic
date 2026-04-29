"""
Script Fine-tuning DPO
Étape 2 - Alignement par préférences
"""

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from pathlib import Path
import json

PROCESSED_DIR = Path("/mnt/prod/data/processed")
MODEL_DIR = Path("/mnt/prod/models")
SFT_CHECKPOINT = MODEL_DIR / "checkpoints" / "sft_final"

def get_dpo_config():
    """Configuration DPO"""
    return {
        "model_name": "Qwen/Qwen3-1.7B",
        "max_seq_length": 1024,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-5,
        "num_train_epochs": 1,
        "beta": 0.1,
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "logging_steps": 25,
        "save_steps": 500,
        "eval_steps": 500,
        "output_dir": str(MODEL_DIR / "checkpoints" / "dpo_final"),
    }

def prepare_dpo_dataset():
    """Prépare le dataset DPO"""
    print("Chargement dataset DPO...")
    ds = load_from_disk(str(PROCESSED_DIR / "dpo_dataset")
    print(f"Dataset chargé: {ds}")
    return ds

def train_dpo():
    """Lance l'entraînement DPO"""
    config = get_dpo_config()
    
    print("=== Configuration DPO ===")
    print(f"Modèle: {config['model_name']}")
    print(f"Beta: {config['beta']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['per_device_batch_size']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    
    # Charger le tokenizer
    print("\nChargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Charger le modèle (utiliser SFT fine-tuné si disponible)
    model_path = str(SFT_CHECKPOINT) if SFT_CHECKPOINT.exists() else config['model_name']
    print(f"\nChargement modèle depuis: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configurer LoRA
    print("Configuration LoRA...")
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=config['target_modules'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Préparer le dataset
    ds = prepare_dpo_dataset()
    
    # Séparer train/val si nécessaire
    if 'validation' not in ds:
        ds = ds['train'].train_test_split(test_size=0.1)
        train_dataset = ds['train']
        eval_dataset = ds['test']
    else:
        train_dataset = ds['train']
        eval_dataset = ds['validation']
    
    # Convertir au format attendu par TRL (instruction -> prompt)
    print("\nConversion dataset au format TRL...")
    def convert_to_prompt_format(example):
        return {
            'prompt': example['instruction'],
            'chosen': example['chosen'],
            'rejected': example['rejected']
        }
    
    train_dataset = train_dataset.map(convert_to_prompt_format)
    eval_dataset = eval_dataset.map(convert_to_prompt_format)
    
    # Sauvegarde le dataset converti pour éviter re-tokenization
    processed_dir = PROCESSED_DIR / "dpo_dataset_trl"
    if not processed_dir.exists():
        print(f"\nSauvegarde dataset TRL dans {processed_dir}...")
        train_dataset.save_to_disk(str(processed_dir / "train")
        eval_dataset.save_to_disk(str(processed_dir / "eval")
    else:
        print(f"\nChargement dataset TRL depuis {processed_dir}...")
        train_dataset = load_from_disk(str(processed_dir / "train")
        eval_dataset = load_from_disk(str(processed_dir / "eval")
    
    # Configuration DPO avec DPOConfig (nouvelle API TRL)
    print("\nInitialisation DPOConfig...")
    dpo_config = DPOConfig(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_batch_size'],
        per_device_eval_batch_size=config['per_device_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        max_length=config['max_seq_length'],
        max_prompt_length=config['max_seq_length'] // 2,
        beta=config['beta'],
    )
    
    # Créer le DPOTrainer
    print("\nInitialisation DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Pas de modèle de référence séparé
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Lancer l'entraînement
    print("\n=== DÉBUT ENTRAÎNEMENT DPO ===")
    dpo_trainer.train()
    
    # Sauvegarder le modèle final
    print("\nSauvegarde modèle final...")
    dpo_trainer.save_model(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])
    
    print(f"\n✅ Entraînement DPO terminé!")
    print(f"Modèle sauvegardé dans: {config['output_dir']}")
    
    return config

def main():
    config = train_dpo()
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "dpo_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\nConfiguration sauvegardée dans models/dpo_config.json")

if __name__ == "__main__":
    main()