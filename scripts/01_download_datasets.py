"""
Script de téléchargement et traitement des datasets médicaux
Étape 1 - Préparation des données pour Fine-Tuning
"""

from datasets import load_dataset, load_from_disk
import json
import os
from pathlib import Path

DATA_DIR = Path("/mnt/prod/data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def download_frenchmedical():
    """Télécharge alternatives médicales françaises"""
    print("Téléchargement datasets médicaux français...")
    
    alternatives = [
        ("illiade/medical_qa_french", "french_qa"),
    ]
    
    for ds_id, name in alternatives:
        try:
            print(f"  → {ds_id}...")
            ds = load_dataset(ds_id)
            ds.save_to_disk(str(RAW_DIR / name))
            print(f"    → {len(ds.get('train', ds))} exemples")
        except Exception as e:
            print(f"    → Erreur: {e}")

def download_medquad():
    """Télécharge MedQuAD - dataset anglais"""
    print("Téléchargement MedQuAD...")
    ds = load_dataset("lavita/MedQuAD")
    ds.save_to_disk(str(RAW_DIR / "medquad"))
    print(f"  → {ds}")
    return ds

def download_ultramedical():
    """Télécharge UltraMedical - dataset bilingue"""
    print("Téléchargement UltraMedical...")
    ds = load_dataset("TsinghuaC3I/UltraMedical")
    ds.save_to_disk(str(RAW_DIR / "ultramedical"))
    print(f"  → {ds}")
    return ds

def download_ultramedical_preference():
    """Télécharge UltraMedical-Preference - dataset DPO"""
    print("Téléchargement UltraMedical-Preference...")
    ds = load_dataset("TsinghuaC3I/UltraMedical-Preference")
    ds.save_to_disk(str(RAW_DIR / "ultramedical_preference"))
    print(f"  → {ds}")
    return ds

def download_all():
    """Télécharge tous les datasets"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "french_medical": download_frenchmedical,
        "medquad": download_medquad,
        "ultramedical": download_ultramedical,
        "ultramedical_preference": download_ultramedical_preference,
    }
    
    for name, func in datasets.items():
        try:
            func()
        except Exception as e:
            print(f"Erreur pour {name}: {e}")

if __name__ == "__main__":
    download_all()