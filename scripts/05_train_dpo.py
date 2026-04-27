"""
Script Fine-tuning DPO
Étape 2 - Alignement par préférences
"""

from datasets import load_from_disk
from pathlib import Path
import json

PROCESSED_DIR = Path("/mnt/prod/data/processed")
MODEL_DIR = Path("/mnt/prod/models")

def get_dpo_config():
    """Configuration DPO"""
    return {
        "model_name": "Qwen/Qwen3-1.7B",
        "ref_model_name": "Qwen/Qwen3-1.7B",
        "max_seq_length": 2048,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
        "num_train_epochs": 2,
        "beta": 0.1,
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "logging_steps": 25,
        "save_steps": 200,
    }

def prepare_dpo_dataset():
    """Prépare le dataset DPO"""
    print("Préparation dataset DPO...")
    ds = load_from_disk(str(PROCESSED_DIR / "dpo_dataset"))
    
    return ds

def train_dpo():
    """Lance l'entraînement DPO"""
    config = get_dpo_config()
    
    print("=== Configuration DPO ===")
    print(f"Modèle: {config['model_name']}")
    print(f"Beta: {config['beta']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    print("\n⚠Ce script nécessite:")
    print("- GPU avec CUDA")
    print("- Libraries: transformers, peft, trl, torch")
    
    return config

def main():
    config = train_dpo()
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "dpo_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\nConfiguration sauvegardée dans models/dpo_config.json")

if __name__ == "__main__":
    main()