"""
Script Fine-tuning DPO FINAL CORRIGÉ
Etape 2 - Alignement par préférences (DPO)
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
        "save_total_limit": 2,
        "warmup_steps": 100,
    }

def add_missing_attributes(args, **kwargs):
    """
    Ajoute tous les attributs manquants à TrainingArguments
    pour garantir la compatibilité avec DPOTrainer
    """
    # Liste exhaustive des attributs requis par DPOTrainer
    # et absents de TrainingArguments
    required_attrs = {
        "model_init_kwargs": None,
        "ref_model_init_kwargs": None,
        "generate_during_eval": False,
        "fsdp": [],  # Liste vide, pas None
        "precompute_ref_log_probs": False,
        "remove_unused_columns": False,
        "include_tokens_per_second": False,
        "include_num_input_tokens_seen": False,
        "label_names": [],
        "lm_head_name": "lm_head",
        "beta": 0.1,
        "model_adapter_name": None,
        "ref_adapter_name": None,
        "reference_free": False,
        "force_use_ref_model": False,
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
        "f_divergence_type": "kl",
        "f_alpha_divergence_coef": 1.0,
        "dataset_num_proc": 1,
        "sync_ref_model": False
    }
    
    # Update avec les valeurs passées explicitement
    required_attrs.update(kwargs)
    
    # Ajouter tous les attributs manquants
    for attr, value in required_attrs.items():
        if not hasattr(args, attr):
            setattr(args, attr, value)
    
    return args

# Sous-classe de DPOTrainer qui corrige le problème de get_batch_samples
class FixedDPOTrainer(DPOTrainer):
    def get_batch_samples(self, batch, num_items=None, device=None):
        """
        Version corrigée de get_batch_samples qui gère l'incompatibilité
        entre TRL et transformers.
        """
        # Si on reçoit (batch, num_items, device) ou similaire
        if device is None and num_items is not None and isinstance(batch, (list, tuple)) and len(batch) >= 1:
            # Adaptation aux différentes versions
            print(f"Debug: Adaptation arguments get_batch_samples: {len(batch)} éléments")
            if len(batch) >= 3:
                # Forme: (batch, num_items, device) ou similaire
                batch, num_items, device = batch[:3]
        
        # Implémentation originale avec 2 arguments
        return super().get_batch_samples(batch, device)

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
    
    # Créer un modèle de référence explicite
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
    
    print(f"\nStatistiques datasets:")
    print(f"- Train: {len(train_dataset)} exemples")
    print(f"- Eval: {len(eval_dataset)} exemples")
    
    # TrainingArguments et patch des attributs manquants
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
        save_total_limit=config.get("save_total_limit", 2),
        warmup_steps=config.get("warmup_steps", 100)
    )
    
    # Ajouter TOUS les attributs manquants avec nos valeurs personnalisées
    print("\nAjout des attributs manquants à TrainingArguments...")
    training_args = add_missing_attributes(
        training_args,
        beta=config['beta'],
        max_length=config['max_seq_length'],
        max_prompt_length=config['max_seq_length'] // 2,
        max_target_length=config['max_seq_length'] // 2,
    )
    
    # DPOTrainer avec gestion d'exceptions et retry
    print("\nInitialisation DPOTrainer...")
    try:
        dpo_trainer = FixedDPOTrainer(
            model=model,
            ref_model=ref_model,  # Référence explicite
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
        print("Tentative avec une configuration alternative...")
        
        # Tentative de configuration alternative
        dpo_trainer = None
        try:
            # Version en dernier recours
            from trl.trainer.dpo_config import DPOConfig
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
                evaluation_strategy="steps",
                save_strategy="steps",
                bf16=True,
                gradient_checkpointing=True,
                beta=config['beta'],
                max_length=config['max_seq_length'],
                max_prompt_length=config['max_seq_length'] // 2,
                max_target_length=config['max_seq_length'] // 2,
            )
            dpo_trainer = FixedDPOTrainer(
                model=model,
                ref_model=ref_model,
                args=dpo_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
            )
            print("✅ DPOTrainer créé avec configuration alternative!")
        except Exception as e2:
            print(f"❌ Échec de la tentative alternative: {type(e2).__name__}: {e2}")
            raise ValueError("Impossible de créer le DPOTrainer. Vérifiez les versions de TRL et transformers.")
    
    # Entraînement
    print("\n=== DÉBUT ENTRAÎNEMENT DPO ===")
    try:
        dpo_trainer.train()
    except Exception as e:
        print(f"\n❌ Erreur pendant l'entraînement: {type(e).__name__}: {e}")
        raise
    
    # Sauvegarde
    print("\nSauvegarde modèle final...")
    try:
        dpo_trainer.save_model(config['output_dir'])
        tokenizer.save_pretrained(config['output_dir'])
        print(f"\n✅ Modèle sauvegardé dans: {config['output_dir']}")
    except Exception as e:
        print(f"\n❌ Erreur pendant la sauvegarde: {type(e).__name__}: {e}")
        # Sauvegarde alternative
        try:
            model.save_pretrained(config['output_dir'])
            print(f"\n✅ Modèle sauvegardé via méthode alternative")
        except Exception as e2:
            print(f"\n❌ Échec de la sauvegarde alternative: {type(e2).__name__}: {e2}")
    
    print(f"\n✅ Entraînement DPO terminé!")
    return config

def main():
    config = train_dpo()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "dpo_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nConfiguration sauvegardée dans models/dpo_config.json")

if __name__ == "__main__":
    main()
