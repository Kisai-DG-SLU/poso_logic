"""
Script Fine-tuning DPO
Etape 2 - Alignement par préférences
"""
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.disable(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
from pathlib import Path
import json

PROCESSED_DIR = Path("/mnt/prod/data/processed")
MODEL_DIR = Path("/mnt/prod/models")
SFT_CHECKPOINT = MODEL_DIR / "checkpoints" / "sft_final"

def get_dpo_config():
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

def train_dpo():
    config = get_dpo_config()
    
    print("=== Configuration DPO ===")
    print(f"Modèle: {config['model_name']}")
    print(f"Beta: {config['beta']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['per_device_batch_size']}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    
    # Tokenizer
    print("\nChargement tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Modèle
    model_path = str(SFT_CHECKPOINT) if SFT_CHECKPOINT.exists() else config['model_name']
    print(f"\nChargement modèle depuis: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    
    # LoRA
    print("Configuration LoRA...")
    lora_config = LoraConfig(
        r=config['lora_r'], lora_alpha=config['lora_alpha'],
        target_modules=config['target_modules'], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Dataset TRL déjà préparé
    processed_dir = PROCESSED_DIR / "dpo_dataset_trl"
    if processed_dir.exists():
        print(f"\nChargement dataset TRL depuis {processed_dir}...")
        train_dataset = load_from_disk(str(processed_dir / "train"))
        eval_dataset = load_from_disk(str(processed_dir / "eval"))
    else:
        print("\nPréparation dataset DPO...")
        ds = load_from_disk(str(PROCESSED_DIR / "dpo_dataset"))
        if 'validation' not in ds:
            ds = ds['train'].train_test_split(test_size=0.1)
            train_dataset = ds['train']
            eval_dataset = ds['test']
        else:
            train_dataset = ds['train']
            eval_dataset = ds['validation']
        def convert_to_prompt_format(example):
            return {'prompt': example['instruction'], 'chosen': example['chosen'], 'rejected': example['rejected']}
        train_dataset = train_dataset.map(convert_to_prompt_format)
        eval_dataset = eval_dataset.map(convert_to_prompt_format)
        print(f"\nSauvegarde dataset TRL dans {processed_dir}...")
        train_dataset.save_to_disk(str(processed_dir / "train"))
        eval_dataset.save_to_disk(str(processed_dir / "eval"))
    
    # TrainingArguments avec l'attribut manquant
    print("\nInitialisation TrainingArguments...")
    training_args = TrainingArguments(
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
        report_to="none"
    )
    # Patch : ajouter l'attribut manquant
    training_args.model_init_kwargs = None
    training_args.ref_model_init_kwargs = None
    
    # DPOTrainer
    print("\nInitialisation DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        beta=config['beta'],
        max_length=config['max_seq_length'],
        max_prompt_length=config['max_seq_length'] // 2,
    )
    
    # Entraînement
    print("\n=== DÉBUT ENTRAÎNEMENT DPO ===")
    dpo_trainer.train()
    
    # Sauvegarde
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
