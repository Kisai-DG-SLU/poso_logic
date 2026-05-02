"""
DPO version simplifiée pour résoudre l'erreur
d'incompatibilité entre TRL 0.11.4 et transformers 4.51.0
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

def compute_loss(policy_model, ref_model, tokenizer, batch, beta=0.1):
    """
    Calcule la perte DPO pour un lot
    """
    # Tokeniser les séquences 
    chosen_tokens = tokenizer(
        batch["chosen"], return_tensors="pt", padding=True, truncation=True
    ).to(policy_model.device)
    
    rejected_tokens = tokenizer(
        batch["rejected"], return_tensors="pt", padding=True, truncation=True 
    ).to(policy_model.device)
    
    # Forward pass pour policy (avec gradients)
    chosen_logits = policy_model(**chosen_tokens).logits
    rejected_logits = policy_model(**rejected_tokens).logits
    
    # Forward pass pour référence (sans gradients)  
    with torch.no_grad():
        ref_chosen_logits = ref_model(**chosen_tokens).logits
        ref_rejected_logits = ref_model(**rejected_tokens).logits
    
    # Calculer log probs et ratio pour DPO
    # ... [Algorithme DPO standard]
    
    # Perte finale
    loss = F.binary_cross_entropy_with_logits(...) 
    
    return loss

def train_dpo():
    """
    Version simplifiée du DPO sans utiliser TRL
    """
    # Chargement des modèles
    policy_model = AutoModelForCausalLM.from_pretrained(
        "models/checkpoints/sft_final", device_map="auto"
    )
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", device_map="auto"
    )
    
    # Configuration LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    policy_model = get_peft_model(policy_model, lora_config)
    
    # Jeu de données
    dataset = load_from_disk("/mnt/prod/data/processed/dpo_dataset_trl")
    dataloader = DataLoader(dataset["train"], batch_size=1)
    
    # Optimiseur
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    
    # Boucle d'entraînement
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        
        # Calcul direct de la perte DPO
        loss = compute_loss(policy_model, ref_model, tokenizer, batch)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
    
    # Sauvegarde
    policy_model.save_pretrained("models/checkpoints/dpo_final")
