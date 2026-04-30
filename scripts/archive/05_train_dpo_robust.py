"""
Script d'entraînement DPO robuste pour Qwen3-1.7B
Conçu pour fonctionner avec l'ensemble du dataset (196k+ exemples) et gérer
les problèmes courants : incompatibilité de tokenizer, modules LoRA incorrects 
et problèmes de mémoire GPU.
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.disable(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from datasets import load_from_disk, Dataset as HFDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    get_linear_schedule_with_warmup, set_seed
)
from peft import (
    LoraConfig, get_peft_model, PeftModel,
    prepare_model_for_kbit_training
)
from pathlib import Path
import json
import time
import gc
import os
from tqdm import tqdm
import traceback
import random
import shutil
import signal
from datetime import datetime
import math
import argparse

# Chemin des dossiers
PROCESSED_DIR = Path("/mnt/prod/data/processed")
MODEL_DIR = Path("/mnt/prod/models")
SFT_CHECKPOINT = MODEL_DIR / "checkpoints" / "sft_final"
LOGS_DIR = Path("/mnt/prod/logs")
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# Configuration globale
def get_dpo_config():
    """Configuration DPO optimisée"""
    return {
        "model_name": "Qwen/Qwen3-1.7B",
        "max_seq_length": 256,            # Considérablement réduit pour économiser la mémoire
        "batch_size": 1,                  # Batch size minimal pour stabilité
        "gradient_accumulation_steps": 64, # Accumulation accrue pour économiser la mémoire
        "learning_rate": 2e-6,            # Taux d'apprentissage encore plus conservateur
        "num_train_epochs": 1,
        "beta": 0.1,                      # Coefficient DPO standard
        "lora_r": 4,                      # Rang LoRA réduit pour économiser la mémoire
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        # Modules cibles corrects pour Qwen3 - uniquement les plus importants pour économiser la mémoire
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "warmup_ratio": 0.03,            # % des steps pour warmup
        "logging_steps": 10,             # Logging plus fréquent
        "save_steps": 25,                # Sauvegarde plus fréquente
        "eval_steps": 100,               # Évaluation moins fréquente pour économiser du temps
        "eval_size": 50,                 # Taille de l'échantillon d'évaluation
        "output_dir": str(MODEL_DIR / "checkpoints" / "dpo_robust"),
        "seed": 42,
        "sharding_size": 5000,          # Taille des shards de données pour la reprise
        "resume_from_checkpoint": True,  # Activer la reprise par défaut
        "max_steps": None               # Si défini, limite le nombre d'étapes total
    }

# Journalisation améliorée
def setup_logging(config):
    """Configurer la journalisation pour le suivi de l'entraînement"""
    log_file = LOGS_DIR / f"dpo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configuration du logger principal
    logger = logging.getLogger("dpo_trainer")
    logger.setLevel(logging.INFO)
    
    # Gestionnaire pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Gestionnaire pour le fichier
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Ajouter les gestionnaires
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Désactiver la propagation pour éviter la double journalisation
    logger.propagate = False
    
    # Journaliser la configuration
    logger.info("=== Configuration DPO ===")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    return logger

# Fonction pour libérer la mémoire GPU
def free_memory():
    """Libérer la mémoire GPU de manière agressive"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Force collecte mémoire Python
    for _ in range(3):
        gc.collect()
    
    # Affiche l'utilisation mémoire
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"Mémoire GPU: {allocated:.2f} GB allouée / {reserved:.2f} GB réservée")

# Calcul des log probabilités
def compute_logps(logits, input_ids, attention_mask):
    """Calcule les log probabilités pour les séquences"""
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

# Calcul de la perte DPO
def compute_loss(policy_model, ref_model, tokenizer, batch, beta, device, max_length=384):
    """Calcule la perte DPO pour un lot de données"""
    try:
        # Extraire les données
        instruction = batch["instruction"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        
        # Tokeniser les séquences complètes
        chosen_seqs = [f"{instruction}\n{chosen}" for instruction, chosen in zip(instruction, chosen)]
        rejected_seqs = [f"{instruction}\n{rejected}" for instruction, rejected in zip(instruction, rejected)]
        
        # Tokeniser avec longueur maximale constante
        chosen_tokens = tokenizer(
            chosen_seqs, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        rejected_tokens = tokenizer(
            rejected_seqs, padding=True, truncation=True, max_length=max_length,
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
        
        # Libérer mémoire après forward pass
        free_memory()
        
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
    except Exception as e:
        print(f"Erreur dans compute_loss: {str(e)}")
        traceback.print_exc()
        raise e

# Fonction d'évaluation
def evaluate(policy_model, ref_model, eval_dataloader, tokenizer, beta, device, max_length=384):
    """Fonction d'évaluation"""
    policy_model.eval()
    total_loss = 0
    total_reward_acc = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            # Calculer la perte sur ce lot
            _, metrics = compute_loss(
                policy_model, ref_model, tokenizer, batch, beta, device, max_length
            )
            
            batch_size = len(batch["instruction"])
            total_loss += metrics["loss"] * batch_size
            total_reward_acc += metrics["reward_acc"] * batch_size
            total_samples += batch_size
            
            # Limiter la durée d'évaluation
            if batch_idx >= 50:
                break
            
            # Libérer la mémoire
            free_memory()
    
    # Remettre policy_model en mode train
    policy_model.train()
    
    # Calculer les moyennes
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_reward_acc = total_reward_acc / total_samples if total_samples > 0 else 0
    
    return {
        "eval_loss": avg_loss,
        "eval_reward_acc": avg_reward_acc,
    }

# Gestionnaire de signal d'interruption
def setup_interruption_handler():
    """Configure un gestionnaire pour les signaux d'interruption"""
    handler_data = {"interrupted": False}
    
    def handler(sig, frame):
        print("\n⚠️ Signal d'interruption reçu. Arrêt propre à la fin de cette étape...")
        handler_data["interrupted"] = True
    
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    
    return handler_data

# Fonction pour charger le dernier checkpoint si disponible
def get_last_checkpoint(checkpoint_dir):
    """Récupère le dernier checkpoint valide"""
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = [
        str(cp) for cp in checkpoint_dir.glob("checkpoint-*")
        if (cp / "adapter_model.safetensors").exists() and (cp / "training_state.json").exists()
        and not "error" in cp.name  # Exclure les checkpoints d'erreur
    ]
    
    if not checkpoints:
        return None
    
    # Trier par numéro d'étape
    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(os.path.basename(x).replace("checkpoint-", ""))
    )
    
    return checkpoints[-1]  # Retourner le plus récent

# Préparation du modèle pour l'entraînement
def prepare_model(config, device, logger):
    """Prépare les modèles (policy et reference) pour l'entraînement"""
    # Tokenizer
    logger.info("Chargement du tokenizer directement depuis le modèle de base...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'], 
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Modèle policy (modèle à entraîner, utilise SFT comme départ)
    model_path = str(SFT_CHECKPOINT) if SFT_CHECKPOINT.exists() else config['model_name']
    logger.info(f"Chargement du modèle policy depuis: {model_path}")
    
    # Utiliser 8bit pour économiser encore plus de mémoire
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,  # Désactiver temporairement la quantification 8 bits
        device_map="auto", 
        trust_remote_code=True,
        low_cpu_mem_usage=True  # Réduire l'utilisation de mémoire CPU
    )
    
    # Modèle de référence (frozen, utilise le modèle de base)
    logger.info("Chargement du modèle de référence...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'], 
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,  # Désactiver temporairement la quantification 8 bits
        device_map="auto", 
        trust_remote_code=True,
        low_cpu_mem_usage=True  # Réduire l'utilisation de mémoire CPU
    )
    
    # S'assurer que ref_model est en eval mode et ses paramètres sont gelés
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # LoRA pour modèle policy
    logger.info(f"Configuration LoRA avec R={config['lora_r']}, Alpha={config['lora_alpha']}...")
    lora_config = LoraConfig(
        r=config['lora_r'], 
        lora_alpha=config['lora_alpha'],
        target_modules=config['target_modules'], 
        lora_dropout=config['lora_dropout'], 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    
    # Préparation pour QLoRA si nécessaire
    if hasattr(policy_model, "enable_input_require_grads"):
        policy_model.enable_input_require_grads()
    else:
        policy_model = prepare_model_for_kbit_training(policy_model)
        
    policy_model = get_peft_model(policy_model, lora_config)
    
    # S'assurer que policy_model est bien en mode train
    policy_model.train()
    
    trainable_params, all_params = policy_model.get_nb_trainable_parameters()
    logger.info(
        f"Paramètres entraînables: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}% "
        f"du total de {all_params:,d} paramètres)"
    )
    
    return tokenizer, policy_model, ref_model

# Préparation des datasets
def prepare_datasets(config, logger):
    """Prépare les datasets d'entraînement et d'évaluation"""
    processed_dir = PROCESSED_DIR / "dpo_dataset_trl"
    logger.info(f"Chargement des datasets depuis {processed_dir}...")
    
    if processed_dir.exists():
        # Charger les datasets
        train_dataset_full = load_from_disk(str(processed_dir / "train"))
        eval_dataset_full = load_from_disk(str(processed_dir / "eval"))
        
        # Limiter encore plus le nombre d'exemples d'évaluation pour économiser la mémoire
        eval_size = min(config['eval_size'], len(eval_dataset_full))
        # Sélectionner le début de l'ensemble d'évaluation
        eval_dataset = eval_dataset_full.select(range(min(20, eval_size)))
        
        logger.info(f"Statistiques des datasets:")
        logger.info(f"- Train: {len(train_dataset_full)} exemples")
        logger.info(f"- Eval: {len(eval_dataset)} exemples")
        
        return train_dataset_full, eval_dataset
    else:
        # Tenter de charger depuis le format de base et convertir
        basic_dir = PROCESSED_DIR / "dpo_dataset"
        logger.info(f"Dataset TRL non trouvé. Tentative depuis {basic_dir}...")
        
        if basic_dir.exists():
            ds = load_from_disk(str(basic_dir))
            
            # Vérifier si le dataset contient la partition validation
            if 'validation' not in ds:
                logger.info("Split train-test de 90-10...")
                ds = ds['train'].train_test_split(test_size=0.1, seed=42)
                train_dataset = ds['train']
                eval_dataset = ds['test']
            else:
                train_dataset = ds['train']
                eval_dataset = ds['validation']
            
            # Conversion au format DPO (trl)
            def convert_to_dpo_format(example):
                return {
                    'instruction': example['instruction'],
                    'chosen': example['chosen'],
                    'rejected': example['rejected']
                }
            
            train_dataset = train_dataset.map(convert_to_dpo_format)
            eval_dataset = eval_dataset.map(convert_to_dpo_format)
            
            # Créer le répertoire TRL et sauvegarder
            logger.info(f"Sauvegarde du dataset au format TRL dans {processed_dir}...")
            processed_dir.mkdir(exist_ok=True, parents=True)
            train_dataset.save_to_disk(str(processed_dir / "train"))
            eval_dataset.save_to_disk(str(processed_dir / "eval"))
            
            # Limitation pour l'évaluation
            eval_size = min(config['eval_size'], len(eval_dataset))
            eval_dataset = eval_dataset.select(range(eval_size))
            
            logger.info(f"Conversion terminée. Statistiques:")
            logger.info(f"- Train: {len(train_dataset)} exemples")
            logger.info(f"- Eval: {len(eval_dataset)} exemples")
            
            return train_dataset, eval_dataset
        else:
            raise ValueError(f"Aucun dataset trouvé à {basic_dir} ou {processed_dir}")

# Fonction principale d'entraînement DPO
def train_dpo(args=None):
    """Fonction d'entraînement DPO robuste avec reprise et gestion des erreurs"""
    # Obtenir la configuration
    config = get_dpo_config()
    
    # Mettre à jour la config avec les arguments de ligne de commande
    if args:
        for k, v in vars(args).items():
            if v is not None and k != 'test_mode':
                config[k] = v
    
    # Configuration de la journalisation
    logger = setup_logging(config)
    
    # Définir la graine pour la reproductibilité
    set_seed(config['seed'])
    
    # Déterminer le device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Utilisation du device: {device}")
    
    # Vérifier les informations GPU
    if device == "cuda":
        gpu_info = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {gpu_info.name}")
        logger.info(f"Mémoire GPU totale: {gpu_info.total_memory / 1024 ** 3:.2f} GB")
    
    # Gestionnaire d'interruption
    interrupt_handler = setup_interruption_handler()
    
    # Préparer les modèles
    tokenizer, policy_model, ref_model = prepare_model(config, device, logger)
    
    # Préparer les datasets
    train_dataset_full, eval_dataset = prepare_datasets(config, logger)
    
    # Mode test : vérifier que tout se charge correctement
    if args and hasattr(args, 'test_mode') and args.test_mode:
        logger.info("\n=== MODE TEST ===")
        logger.info("✅ Modèles chargés correctement")
        logger.info("✅ Datasets chargés correctement")
        logger.info(f"✅ Train dataset: {len(train_dataset_full)} exemples")
        logger.info(f"✅ Eval dataset: {len(eval_dataset)} exemples")
        
        # Test d'un calcul de perte sur un échantillon
        logger.info("\nTest du calcul de perte sur un échantillon...")
        test_batch = {
            "instruction": [train_dataset_full[0]["instruction"]],
            "chosen": [train_dataset_full[0]["chosen"]],
            "rejected": [train_dataset_full[0]["rejected"]]
        }
        
        try:
            loss, metrics = compute_loss(
                policy_model, ref_model, tokenizer, test_batch, 
                config['beta'], device, config['max_seq_length']
            )
            logger.info(f"✅ Calcul de perte réussi - Loss: {loss.item():.4f}")
            logger.info(f"✅ Métriques: {metrics}")
        except Exception as e:
            logger.error(f"❌ Erreur lors du test de calcul de perte: {e}")
            raise
        
        logger.info("\n✅ TOUS LES TESTS PASSENT - LE SCRIPT EST PRÊT POUR L'ENTRAÎNEMENT")
        return config
    
    # Créer le répertoire de sortie
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DataLoader pour l'évaluation (fixe) - sans multi-processing pour éviter les problèmes de mémoire
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,  # Éviter le multiprocessing qui peut causer des OOM
        pin_memory=False  # Désactiver pin_memory pour réduire la consommation mémoire
    )
    
    # État de l'entraînement et variables de contrôle
    start_epoch = 0
    start_step = 0
    best_eval_loss = float("inf")
    global_step = 0
    
    # Utilisation de shards pour gérer les reprises et les lots importants
    shard_size = config["sharding_size"]  # Taille fixe de chaque shard
    num_shards = math.ceil(len(train_dataset_full) / config["sharding_size"])
    current_shard = 0
    
    # Vérifier s'il faut reprendre depuis un checkpoint existant
    if config['resume_from_checkpoint']:
        checkpoint_path = get_last_checkpoint(output_dir)
        
        if checkpoint_path:
            logger.info(f"Reprise depuis le checkpoint: {checkpoint_path}")
            
            # Charger l'état d'entraînement
            try:
                with open(os.path.join(checkpoint_path, "training_state.json"), "r") as f:
                    state = json.load(f)
                
                start_step = state.get("global_step", 0)
                start_epoch = state.get("epoch", 0)
                current_shard = state.get("current_shard", 0)
                best_eval_loss = state.get("best_eval_loss", float('inf'))
                
                # Charger les poids LoRA
                logger.info(f"Chargement des poids LoRA depuis {checkpoint_path}...")
                policy_model = PeftModel.from_pretrained(
                    policy_model,
                    checkpoint_path,
                    is_trainable=True
                )
                policy_model.print_trainable_parameters()
                
                global_step = start_step
                
                logger.info(f"Reprise à l'époque {start_epoch}, étape {global_step}, shard {current_shard}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du checkpoint: {e}")
                logger.warning("Démarrage d'un nouvel entraînement...")
                start_step = 0
                start_epoch = 0
    
    # Paramètres d'entraînement
    num_epochs = config['num_train_epochs']
    
    # Calcul des étapes totales
    total_steps = int(len(train_dataset_full) / config['batch_size'] / config['gradient_accumulation_steps'] * num_epochs)
    if config['max_steps'] and config['max_steps'] < total_steps:
        total_steps = config['max_steps']
    
    logger.info(f"Entraînement pour: {total_steps} étapes au total")
    
    # Warmup steps basé sur ratio
    warmup_steps = max(1, int(total_steps * config['warmup_ratio']))
    logger.info(f"Warmup: {warmup_steps} étapes ({config['warmup_ratio'] * 100:.1f}% du total)")
    
    # Boucle d'entraînement principale
    epochs_to_process = range(start_epoch, num_epochs)
    logger.info(f"Démarrage de l'entraînement DPO sur {num_epochs} époques")
    
    for epoch in epochs_to_process:
        # Traitement par shards pour permettre la reprise à un point intermédiaire
        for shard_idx in range(current_shard, num_shards):
            # Créer un sous-ensemble pour ce shard
            start_idx = shard_idx * shard_size
            end_idx = min((shard_idx + 1) * shard_size, len(train_dataset_full))
            
            # Sélectionner les indices pour ce shard
            indices = list(range(start_idx, end_idx))
            
            # Mélanger les indices dans ce shard
            random.shuffle(indices)
            current_shard_dataset = train_dataset_full.select(indices)
            
            logger.info(f"Époque {epoch+1}/{num_epochs}, Shard {shard_idx+1}/{num_shards}: "
                        f"{start_idx}-{end_idx} ({len(indices)} exemples)")
            
            # Créer le DataLoader pour ce shard - sans multi-processing pour éviter les problèmes de mémoire
            train_dataloader = DataLoader(
                current_shard_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=0,  # Désactiver le multiprocessing pour éviter les OOM
                pin_memory=False  # Désactiver pin_memory pour réduire la consommation mémoire
            )
            
            # Optimizer avec les optimisations pour la mémoire
            optimizer = torch.optim.AdamW(
                policy_model.parameters(),
                lr=config['learning_rate'],
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # Scheduler avec warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            
            # Sauter les étapes déjà complétées pour ce shard (en cas de reprise)
            steps_in_this_shard = 0
            steps_to_skip = 0
            
            if shard_idx == current_shard and global_step > 0:
                steps_to_skip = global_step
                
                # Avancer le scheduler au bon endroit
                for _ in range(steps_to_skip):
                    scheduler.step()
                
                logger.info(f"Saut de {steps_to_skip} étapes déjà complétées")
            
            # Boucle d'entraînement sur les lots
            accumulated_loss = 0
            log_history = []
            
            policy_model.train()
            
            # Préparation de la barre de progression
            pbar = tqdm(total=len(train_dataloader), desc=f"Shard {shard_idx+1} Entraînement")
            start_time = time.time()
            
            try:
                for batch_idx, batch in enumerate(train_dataloader):
                    # Vérifier l'interruption
                    if interrupt_handler["interrupted"]:
                        logger.info("Interruption détectée. Sauvegarde et arrêt...")
                        break
                    
                    # Sauter les étapes déjà complétées (en cas de reprise)
                    if steps_to_skip > 0 and batch_idx < steps_to_skip:
                        continue
                    
                    # Gestion de la mémoire avant calcul
                    free_memory()
                    
                    # Calcul de la perte
                    loss, metrics = compute_loss(
                        policy_model, ref_model, tokenizer, batch, config['beta'], device,
                        max_length=config['max_seq_length']
                    )
                    
                    # Normaliser la perte pour l'accumulation de gradients
                    loss = loss / config['gradient_accumulation_steps']
                    
                    # Rétropropagation
                    loss.backward()
                    
                    # Accumuler la perte pour le logging
                    accumulated_loss += loss.item()
                    
                    # Libérer mémoire après backward
                    free_memory()
                    
                    # Mise à jour des poids
                    if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0 or batch_idx == len(train_dataloader) - 1:
                        # Clip des gradients pour stabiliser l'entraînement
                        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                        
                        # Mise à jour des poids
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        # Incrémenter le compteur global
                        global_step += 1
                        steps_in_this_shard += 1
                        
                        # Mettre à jour la barre de progression
                        pbar.update(1)
                        pbar.set_description(
                            f"E{epoch+1}S{shard_idx+1} | Step {global_step} | "
                            f"Loss: {accumulated_loss:.5f} | Acc: {metrics['reward_acc']:.4f}"
                        )
                        
                        # Journalisation détaillée
                        if global_step % config['logging_steps'] == 0:
                            elapsed = time.time() - start_time
                            elapsed_per_step = elapsed / max(1, steps_in_this_shard)
                            
                            log_info = {
                                "step": global_step,
                                "epoch": epoch,
                                "shard": shard_idx,
                                "loss": accumulated_loss,
                                "reward_acc": metrics["reward_acc"],
                                "chosen_rewards": metrics["chosen_rewards"],
                                "rejected_rewards": metrics["rejected_rewards"],
                                "learning_rate": scheduler.get_last_lr()[0],
                                "elapsed_seconds": elapsed,
                                "seconds_per_step": elapsed_per_step,
                            }
                            
                            log_history.append(log_info)
                            
                            logger.info(
                                f"Step {global_step} | Loss: {accumulated_loss:.5f} | "
                                f"LR: {scheduler.get_last_lr()[0]:.8f} | "
                                f"Reward Acc: {metrics['reward_acc']:.4f} | "
                                f"{elapsed_per_step:.1f} sec/step"
                            )
                            
                            # Réinitialiser la perte accumulée
                            accumulated_loss = 0
                        
                        # Évaluation périodique
                        if global_step % config['eval_steps'] == 0:
                            logger.info(f"\nÉvaluation à l'étape {global_step}...")
                            eval_results = evaluate(
                                policy_model, ref_model, eval_dataloader, tokenizer, 
                                config['beta'], device, config['max_seq_length']
                            )
                            
                            logger.info(
                                f"Eval Loss: {eval_results['eval_loss']:.5f} | "
                                f"Eval Reward Acc: {eval_results['eval_reward_acc']:.4f}"
                            )
                            
                            # Sauvegarde si meilleur modèle
                            if eval_results["eval_loss"] < best_eval_loss:
                                best_eval_loss = eval_results["eval_loss"]
                                logger.info(
                                    f"Nouveau meilleur modèle! "
                                    f"Perte d'évaluation: {eval_results['eval_loss']:.5f}"
                                )
                                
                                # Sauvegarde du meilleur modèle
                                best_model_dir = output_dir / "best_model"
                                best_model_dir.mkdir(exist_ok=True, parents=True)
                                policy_model.save_pretrained(best_model_dir)
                                tokenizer.save_pretrained(best_model_dir)
                                
                                # Sauvegarder l'état
                                training_state = {
                                    "global_step": global_step,
                                    "epoch": epoch,
                                    "current_shard": shard_idx,
                                    "best_eval_loss": best_eval_loss,
                                    "evaluation": eval_results,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                }
                                
                                with open(best_model_dir / "training_state.json", "w") as f:
                                    json.dump(training_state, f, indent=2)
                        
                        # Sauvegarde régulière (checkpoint)
                        if global_step % config['save_steps'] == 0:
                            checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                            checkpoint_dir.mkdir(exist_ok=True, parents=True)
                            
                            # Sauvegarde du modèle
                            policy_model.save_pretrained(checkpoint_dir)
                            tokenizer.save_pretrained(checkpoint_dir)
                            
                            # Sauvegarde de l'état
                            training_state = {
                                "global_step": global_step,
                                "epoch": epoch,
                                "current_shard": shard_idx,
                                "best_eval_loss": best_eval_loss,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            
                            with open(checkpoint_dir / "training_state.json", "w") as f:
                                json.dump(training_state, f, indent=2)
                            
                            # Sauvegarder la configuration
                            with open(checkpoint_dir / "config.json", "w") as f:
                                json.dump(config, f, indent=2)
                            
                            logger.info(f"Checkpoint sauvegardé: {checkpoint_dir}")
                            
                            # Nettoyage des anciens checkpoints si nécessaire
                            checkpoints = sorted(
                                [cp for cp in output_dir.glob("checkpoint-*")],
                                key=lambda x: int(x.name.replace("checkpoint-", ""))
                            )
                            
                            # Conserver uniquement les 3 derniers checkpoints pour économiser l'espace
                            if len(checkpoints) > 3:
                                for old_checkpoint in checkpoints[:-3]:
                                    try:
                                        logger.info(f"Suppression de l'ancien checkpoint: {old_checkpoint}")
                                        shutil.rmtree(old_checkpoint)
                                    except Exception as e:
                                        logger.warning(f"Erreur lors de la suppression de {old_checkpoint}: {e}")
                        
                        # Vérifier si le nombre maximum d'étapes a été atteint
                        if config['max_steps'] and global_step >= config['max_steps']:
                            logger.info(f"Nombre maximum d'étapes atteint ({config['max_steps']}). Arrêt...")
                            break
            
            except Exception as e:
                logger.error(f"\nErreur pendant l'entraînement: {type(e).__name__}: {e}")
                logger.error(traceback.format_exc())
                
                # Sauvegarder le modèle dans l'état actuel pour reprise
                error_checkpoint_dir = output_dir / f"checkpoint-error-{global_step}"
                error_checkpoint_dir.mkdir(exist_ok=True, parents=True)
                
                try:
                    # Sauvegarde du modèle
                    policy_model.save_pretrained(error_checkpoint_dir)
                    tokenizer.save_pretrained(error_checkpoint_dir)
                    
                    # Sauvegarde de l'état
                    training_state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "current_shard": shard_idx,
                        "best_eval_loss": best_eval_loss,
                        "error": str(e),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    with open(error_checkpoint_dir / "training_state.json", "w") as f:
                        json.dump(training_state, f, indent=2)
                    
                    logger.info(f"État sauvegardé après erreur: {error_checkpoint_dir}")
                    logger.info("Vous pouvez reprendre l'entraînement à partir de ce point avec --resume_from_checkpoint")
                except Exception as save_error:
                    logger.error(f"Erreur lors de la sauvegarde après échec: {save_error}")
                
                # Tenter de recommencer avec la configuration actuelle après une pause
                logger.info("Pause de 10 secondes avant de tenter de poursuivre...")
                time.sleep(10)
                continue  # Continuer avec le shard suivant
            
            finally:
                # Fermer la barre de progression
                pbar.close()
            
            # Mettre à jour le point de reprise pour le shard suivant
            current_shard += 1
            
            # Vérifier l'interruption ou l'atteinte du nombre maximum d'étapes
            if (interrupt_handler["interrupted"] or 
                (config['max_steps'] and global_step >= config['max_steps'])):
                break
        
        # Réinitialiser l'indice de shard pour la prochaine époque
        current_shard = 0
        
        # Vérifier l'interruption ou l'atteinte du nombre maximum d'étapes
        if (interrupt_handler["interrupted"] or 
            (config['max_steps'] and global_step >= config['max_steps'])):
            break
    
    # Évaluation finale
    logger.info("\nÉvaluation finale...")
    final_eval_results = evaluate(
        policy_model, ref_model, eval_dataloader, tokenizer, 
        config['beta'], device, config['max_seq_length']
    )
    
    logger.info(
        f"Eval Loss final: {final_eval_results['eval_loss']:.5f} | "
        f"Eval Reward Acc final: {final_eval_results['eval_reward_acc']:.4f}"
    )
    
    # Sauvegarde finale
    final_output_dir = output_dir / "final"
    final_output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"\nSauvegarde du modèle final dans: {final_output_dir}")
    policy_model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    # Sauvegarder l'historique et la config
    with open(output_dir / "log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)
    
    with open(output_dir / "dpo_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Sauvegarder les résultats finaux
    final_results = {
        "final_eval_loss": final_eval_results["eval_loss"],
        "final_eval_reward_acc": final_eval_results["eval_reward_acc"],
        "best_eval_loss": best_eval_loss,
        "total_steps": global_step,
        "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_training_time_seconds": time.time() - start_time
    }
    
    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    logger.info("\n✅ ENTRAÎNEMENT DPO TERMINÉ AVEC SUCCÈS!")
    
    return config

def main():
    """Point d'entrée principal avec parsing des arguments"""
    parser = argparse.ArgumentParser(description='Entraînement DPO robuste pour Qwen3')
    
    # Arguments pour remplacer les valeurs de configuration
    parser.add_argument('--batch_size', type=int, help='Taille du batch')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Étapes d\'accumulation du gradient')
    parser.add_argument('--learning_rate', type=float, help='Taux d\'apprentissage')
    parser.add_argument('--max_steps', type=int, help='Nombre maximum d\'étapes d\'entraînement')
    parser.add_argument('--max_seq_length', type=int, help='Longueur maximale des séquences')
    parser.add_argument('--sharding_size', type=int, help='Taille des shards de données')
    parser.add_argument('--resume_from_checkpoint', type=bool, help='Reprendre depuis un checkpoint')
    parser.add_argument('--save_steps', type=int, help='Fréquence de sauvegarde')
    parser.add_argument('--seed', type=int, help='Graine aléatoire')
    parser.add_argument('--test_mode', action='store_true', help='Mode test - vérifier que tout se charge correctement sans entraînement')
    
    args = parser.parse_args()
    
    try:
        config = train_dpo(args)
        print("\n=== INFORMATIONS FINALES ===")
        print(f"- Modèle entraîné sauvegardé dans: {config['output_dir']}")
        print(f"- Configuration sauvegardée dans: {config['output_dir']}/dpo_config.json")
        print(f"- Résultats finaux dans: {config['output_dir']}/final_results.json")
        
        # Derniers conseils
        print("\n💡 CONSEILS POUR LE DÉPLOIEMENT:")
        print("1. Utiliser vLLM pour servir le modèle final: `/mnt/prod/models/checkpoints/dpo_robust/final`")
        print("2. Tester avec des exemples cliniques pour vérifier si l'alignement a fonctionné")
        print("3. Mesurer la latence et les performances pour le rapport technique")
        
    except Exception as e:
        print("\n==== ERREUR CRITIQUE DURANT L'ENTRAÎNEMENT ====")
        print(f"Type d'erreur: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nTraceback complet:")
        traceback.print_exc()
        
        if torch.cuda.is_available():
            print("\nInformations GPU:")
            print(f"Mémoire allouée: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"Mémoire réservée: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        print("\nCessation de l'entraînement. Utilisez les checkpoints intermédiaires si disponibles.")
        sys.exit(1)

if __name__ == "__main__":
    main()