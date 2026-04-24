#!/usr/bin/env python3
"""
Fine-tuning SFT LoRA complet - À exécuter sur machine avec GPU
Étape 2: Entraînement SFT + DPO
"""

import os
import torch
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import json
from pathlib import Path

PROCESSED_DIR = Path("/mnt/prod/data/processed")
MODEL_DIR = Path("/mnt/prod/models")
OUTPUT_DIR = MODEL_DIR / "checkpoints"

def setup_model(model_name):
    """Charge le modèle et tokenizer"""
    print(f"Chargement {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    model.config.use_cache = False
    
    return model, tokenizer

def setup_lora(model, config):
    """Applique LoRA au modèle"""
    print("Configuration LoRA...")
    
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def load_sft_data():
    """Charge le dataset SFT"""
    print("Chargement dataset SFT...")
    ds = load_from_disk(str(PROCESSED_DIR / "sft_dataset"))
    
    def format_chat(example):
        instruction = example.get("instruction", "")
        response = example.get("response", "")
        
        return {
            "text": f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        }
    
    ds = ds.map(format_chat, remove_columns=ds.column_names)
    return ds

def train_sft(model_name=None, config_path=None):
    """Entraînement SFT complet"""
    
    if config_path is None:
        config_path = MODEL_DIR / "sft_config.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    if model_name is None:
        model_name = config.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer = setup_model(model_name)
    model = setup_lora(model, config)
    
    train_ds = load_sft_data()
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=config.get("per_device_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        num_train_epochs=config.get("num_train_epochs", 3),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        logging_steps=config.get("logging_steps", 25),
        save_steps=config.get("save_steps", 200),
        save_total_limit=3,
        bf16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds["train"],
        eval_dataset=train_ds.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Démarrage entraînement SFT...")
    trainer.train()
    
    print(f"Sauvegarde modèle dans {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR / "sft_final")
    tokenizer.save_pretrained(OUTPUT_DIR / "sft_final")
    
    return OUTPUT_DIR / "sft_final"

def train_dpo(model_path=None):
    """Entraînement DPO (après SFT)"""
    
    if model_path is None:
        model_path = OUTPUT_DIR / "sft_final"
    
    config_path = MODEL_DIR / "dpo_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"=== DPO à partir de {model_path} ===")
    print("DPO nécessite: pip install trl")
    print("Code à implémenter avec DPOTrainer detrl")
    
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sft", "dpo"], default="sft")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    
    if args.mode == "sft":
        train_sft(model_name=args.model)
    else:
        train_dpo()