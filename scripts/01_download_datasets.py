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

def download_frenchmedmcqa():
    """Télécharge FrenchMedMCQA - dataset français"""
    print("Téléchargement FrenchMedMCQA...")
    ds = load_dataset("qanastek/FrenchMedMCQA")
    ds.save_to_disk(str(RAW_DIR / "frenchmedmcqa"))
    print(f"  → {ds}")
    return ds

def download_mediqa():
    """Télécharge MediQA - dataset anglais"""
    print("Téléchargement MediQA...")
    ds = load_dataset("bigbio/mediqa_qa")
    ds.save_to_disk(str(RAW_DIR / "mediqa_qa"))
    print(f"  → {ds}")
    return ds

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
        "frenchmedmcqa": download_frenchmedmcqa,
        "mediqa": download_mediqa,
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