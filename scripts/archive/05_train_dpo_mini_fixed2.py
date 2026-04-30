"""
Version réduite du script d'entraînement DPO avec un jeu de données limité
pour accélérer le test et réduire les temps d'exécution
"""
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.disable(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

# Appliquer le monkey patch AVANT d'importer TRL
from importlib.machinery import SourceFileLoader
patch_module = SourceFileLoader("monkey_patch_direct", 
                                "/mnt/prod/scripts/monkey_patch_direct.py").load_module()
patch_module.apply_trl_patches()

import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
from pathlib import Path
import json

# Chemin des dossiers
PROCESSED_DIR = Path("/mnt/prod/data/processed")
MODEL_DIR = Path("/mnt/prod/models")
SFT_CHECKPOINT = MODEL_DIR / "checkpoints" / "sft_final"

def add_missing_attributes(args, **kwargs):
    """
    Ajoute les attributs manquants à TrainingArguments pour DPO
    """
    required_attrs = {
        "model_init_kwargs": None,
        "ref_model_init_kwargs": None,
        "generate_during_eval": False,
        "fsdp": [],
        "precompute_ref_log_probs": False,
        "remove_unused_columns": False,
        "include_tokens_per_second": False,
        "label_names": [],
        "beta": 0.1,
        "model_adapter_name": None,
        "ref_adapter_name": None,
        "reference_free": False,
        "max_length": 1024,
        "max_prompt_length": 512,
        "max_target_length": 512,
        "max_completion_length": 512,
        "is_encoder_decoder": False,
        "truncation_mode": "keep_end",
        "optimization_method": "offline",
        "loss_type": "sigmoid",
        "label_pad_token_id": -100,
        "disable_dropout": True,
        "label_smoothing": 0,
        "dataset_num_proc": 1,
        "sync_ref_model": False,
        "f_divergence_type": "kl",
        "f_alpha_divergence_coef": 1.0,
        "include_num_input_tokens_seen": False,
        "lm_head_name": "lm_head",
        "force_use_ref_model": False,
    }
    
    # Mise à jour avec les valeurs fournies
    required_attrs.update(kwargs)
    
    # Ajout des attributs manquants
    for attr, value in required_attrs.items():
        if not hasattr(args, attr):
            setattr(args, attr, value)
    
    return args

def get_dpo_config():
    """Configuration DPO"""
    return {
        "model_name": "Qwen/Qwen3-1.7B",
        "max_seq_length": 512,  # Réduit pour accélérer
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 4,  # Réduit pour accélérer
        "learning_rate": 1e-5,
        "num_train_epochs": 1,
        "beta": 0.1,
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "logging_steps": 5,  # Réduit pour voir plus de logs
        "save_steps": 50,  # Réduit pour sauvegarder plus souvent
        "eval_steps": 50,  # Réduit pour évaluer plus souvent
        "output_dir": str(MODEL_DIR / "checkpoints" / "dpo_mini_test"),
        "save_total_limit": 1,
        "warmup_steps": 10,
        "max_samples": 100,  # Limiter le nombre d'exemples
    }

def train_dpo():
    """
    Fonction principale d'entraînement DPO avec dataset réduit
    """
    config = get_dpo_config()
    
    print("=== Configuration DPO Mini Test ===")
    print(f"Modèle: {config['model_name']}")
    print(f"Beta: {config['beta']}")
    print(f"Max samples: {config['max_samples']}")
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
    
    # Référence explicite
    print("\nCréation modèle référence (même poids que pré-SFT)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    
    # LoRA
    print("Configuration LoRA...")
    lora_config = LoraConfig(
        r=config['lora_r'], lora_alpha=config['lora_alpha'],
        target_modules=config['target_modules'], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Dataset TRL - avec limitation
    processed_dir = PROCESSED_DIR / "dpo_dataset_trl"
    if processed_dir.exists():
        print(f"\nChargement dataset TRL depuis {processed_dir}...")
        
        # Charger les datasets complets
        train_dataset_full = load_from_disk(str(processed_dir / "train"))
        eval_dataset_full = load_from_disk(str(processed_dir / "eval"))
        
        # Limiter le nombre d'exemples pour accélérer
        max_samples = config['max_samples']
        train_dataset = train_dataset_full.select(range(min(max_samples, len(train_dataset_full))))
        eval_dataset = eval_dataset_full.select(range(min(max_samples//10, len(eval_dataset_full))))
        
        print(f"\nStatistiques datasets:")
        print(f"- Train original: {len(train_dataset_full)} exemples")
        print(f"- Train réduit: {len(train_dataset)} exemples")
        print(f"- Eval réduit: {len(eval_dataset)} exemples")
    else:
        raise ValueError(f"Dataset TRL introuvable dans {processed_dir}")
    
    # Arguments
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
        report_to="none",
        save_total_limit=config.get("save_total_limit", 1),
        warmup_steps=config.get("warmup_steps", 10),
        # Options pour accélérer
        max_steps=20,  # Limiter à 20 steps max
        dataloader_num_workers=1,
    )
    
    # Ajout attributs
    print("\nAjout des attributs manquants à TrainingArguments...")
    training_args = add_missing_attributes(
        training_args,
        beta=config['beta'],
        max_length=config['max_seq_length'],
        max_prompt_length=config['max_seq_length'] // 2,
        max_target_length=config['max_seq_length'] // 2,
    )
    
    # DPOTrainer patché
    print("\nInitialisation DPOTrainer (avec patch)...")
    try:
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            beta=config['beta'],
            max_length=config['max_seq_length'],
            max_prompt_length=config['max_seq_length'] // 2,
            loss_type="sigmoid",
        )
        print("✅ DPOTrainer créé avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation du DPOTrainer: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Entraînement
    print("\n=== DÉBUT ENTRAÎNEMENT DPO ===")
    try:
        # Forcer l'usage de _inner_training_loop patché
        result = dpo_trainer.train()
        print(f"✅ Entraînement terminé: {result}")
    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Sauvegarde
    print("\nSauvegarde modèle final...")
    try:
        dpo_trainer.save_model(config['output_dir'])
        tokenizer.save_pretrained(config['output_dir'])
        print(f"✅ Modèle sauvegardé dans: {config['output_dir']}")
    except Exception as e:
        print(f"❌ Erreur pendant la sauvegarde: {type(e).__name__}: {e}")
        try:
            model.save_pretrained(config['output_dir'])
            print(f"✅ Modèle sauvegardé via méthode alternative")
        except:
            pass
    
    return config

def main():
    config = train_dpo()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "dpo_mini_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nConfiguration mini sauvegardée dans models/dpo_mini_config.json")

if __name__ == "__main__":
    main()
