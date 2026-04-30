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
    
    try:
        print("  → ANR-MALADES/MediQAl (MCQU - 32k questions FR)...")
        ds = load_dataset("ANR-MALADES/MediQAl", "mcqu")
        ds.save_to_disk(str(RAW_DIR / "mediqal_mcqu"))
        print(f"    → {len(ds['train'])} exemples train")
    except Exception as e:
        print(f"    → Erreur MediQAl MCQU: {e}")
    
    try:
        print("  → ANR-MALADES/MediQAl (MCQM)...")
        ds = load_dataset("ANR-MALADES/MediQAl", "mcqm")
        ds.save_to_disk(str(RAW_DIR / "mediqal_mcqm"))
        print(f"    → {len(ds['train'])} exemples train")
    except Exception as e:
        print(f"    → Erreur MediQAl MCQM: {e}")
    
    try:
        print("  → ANR-MALADES/MediQAl (OEQ)...")
        ds = load_dataset("ANR-MALADES/MediQAl", "oeq")
        ds.save_to_disk(str(RAW_DIR / "mediqal_oeq"))
        print(f"    → {len(ds['test'])} exemples")
    except Exception as e:
        print(f"    → Erreur MediQAl OEQ: {e}")
    
    try:
        print("  → qanastek/FrenchMedMCQA (3,105 questions)...")
        ds = load_dataset("qanastek/FrenchMedMCQA")
        ds.save_to_disk(str(RAW_DIR / "frenchmedmcqa"))
        print(f"    → {len(ds['train'])} exemples")
    except Exception as e:
        print(f"    → Erreur FrenchMedMCQA: {e}")
    
    try:
        print("  → abdellahennajari/FrBMedQA (41k biomedical FR)...")
        ds = load_dataset("abdellahennajari/FrBMedQA")
        ds.save_to_disk(str(RAW_DIR / "frbmedqa"))
        print(f"    → {len(ds['train'])} exemples")
    except Exception as e:
        print(f"    → Erreur FrBMedQA: {e}")

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