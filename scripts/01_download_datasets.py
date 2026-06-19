"""
Script de téléchargement et traitement des datasets médicaux
Étape 1 - Préparation des données pour Fine-Tuning
Inclut le calcul SHA256 pour la traçabilité et le versionnement
"""

from datasets import load_dataset, load_from_disk
import hashlib
import json
import os
from pathlib import Path

# Répertoires de stockage
DATA_DIR = Path("/mnt/prod/data")
RAW_DIR = DATA_DIR / "raw"          # Datasets bruts téléchargés depuis HuggingFace
PROCESSED_DIR = DATA_DIR / "processed"  # Datasets anonymisés prêts pour l'entraînement

# Fichier de registre des versions (SHA256) pour traçabilité
CHECKSUMS_FILE = RAW_DIR / "datasets_checksums.json"


def compute_sha256(filepath: Path) -> str:
    """Calcule le hash SHA256 d'un fichier pour tracer la version exacte du dataset"""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def save_checksums(checksums: dict):
    """Sauvegarde les checksums dans le registre JSON (datasets_checksums.json)"""
    CHECKSUMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKSUMS_FILE, "w") as f:
        json.dump(checksums, f, indent=2)
    print(f"  → Checksums sauvegardés dans {CHECKSUMS_FILE}")

def log_dataset_version(name: str, path: Path, metadata: dict):
    """Enregistre la version, le SHA256 et la date d'un dataset pour la traçabilité"""
    checksums = {}
    if CHECKSUMS_FILE.exists():
        with open(CHECKSUMS_FILE) as f:
            checksums = json.load(f)
    
    checksums[name] = {
        "dataset": name,
        "version": metadata.get("version", "unknown"),
        "date": metadata.get("date", ""),
        "num_examples": metadata.get("num_examples", 0),
        "sha256": metadata.get("sha256", "non calculé"),
    }
    save_checksums(checksums)

def get_dataset_metadata(ds, name: str, save_path: Path) -> dict:
    """Extrait les métadonnées (taille, hash SHA256, splits) d'un dataset sauvegardé"""
    num_examples = len(ds.get("train", ds)) if isinstance(ds, dict) else len(ds)
    
    sha256 = ""
    arrow_files = list(save_path.rglob("data.arrow")) + list(save_path.rglob("*.arrow"))
    if arrow_files:
        sha256 = compute_sha256(arrow_files[0])
    
    return {
        "dataset": name,
        "version": "1.0",
        "date": __import__("datetime").datetime.now().strftime("%Y-%m-%d"),
        "num_examples": num_examples,
        "sha256": sha256,
        "splits": list(ds.keys()) if isinstance(ds, dict) else ["train"],
    }


def download_frenchmedical():
    """Télécharge tous les datasets médicaux français depuis HuggingFace"""
    print("Téléchargement datasets médicaux français...")
    
    # 5 sources françaises : questions médicales (MediQAl, FrenchMedMCQA, FrBMedQA)
    datasets_fr = {
        "mediqal_mcqu": ("ANR-MALADES/MediQAl", "mcqu"),
        "mediqal_mcqm": ("ANR-MALADES/MediQAl", "mcqm"),
        "mediqal_oeq": ("ANR-MALADES/MediQAl", "oeq"),
        "frenchmedmcqa": ("qanastek/FrenchMedMCQA", None),
        "frbmedqa": ("abdellahennajari/FrBMedQA", None),
    }
    
    for local_name, (hf_name, config) in datasets_fr.items():
        try:
            save_path = RAW_DIR / local_name
            print(f"  → {hf_name} ({local_name})...")
            # Téléchargement depuis HF (avec config si nécessaire)
            if config:
                ds = load_dataset(hf_name, config)
            else:
                ds = load_dataset(hf_name)
            ds.save_to_disk(str(save_path))
            num = len(ds.get("train", ds)) if isinstance(ds, dict) else len(ds)
            metadata = get_dataset_metadata(ds, local_name, save_path)
            log_dataset_version(local_name, save_path, metadata)
            print(f"    → {num} exemples, SHA256: {metadata['sha256'][:16]}...")
        except Exception as e:
            print(f"    → Erreur {local_name}: {e}")


def download_medquad():
    """Télécharge MedQuAD (anglais) — questions/réponses médicales depuis HF"""
    local_name = "medquad"
    print("Téléchargement MedQuAD...")
    ds = load_dataset("lavita/MedQuAD")
    save_path = RAW_DIR / local_name
    ds.save_to_disk(str(save_path))
    metadata = get_dataset_metadata(ds, local_name, save_path)
    log_dataset_version(local_name, save_path, metadata)
    print(f"  → {ds}, SHA256: {metadata['sha256'][:16]}...")
    return ds


def download_ultramedical():
    """Télécharge UltraMedical (anglais) — corpus médical général pour SFT"""
    local_name = "ultramedical"
    print("Téléchargement UltraMedical...")
    ds = load_dataset("TsinghuaC3I/UltraMedical")
    save_path = RAW_DIR / local_name
    ds.save_to_disk(str(save_path))
    metadata = get_dataset_metadata(ds, local_name, save_path)
    log_dataset_version(local_name, save_path, metadata)
    print(f"  → {ds}, SHA256: {metadata['sha256'][:16]}...")
    return ds


def download_ultramedical_preference():
    """Télécharge UltraMedical-Preference (anglais) — paires préférée/rejetée pour DPO"""
    local_name = "ultramedical_preference"
    print("Téléchargement UltraMedical-Preference...")
    ds = load_dataset("TsinghuaC3I/UltraMedical-Preference")
    save_path = RAW_DIR / local_name
    ds.save_to_disk(str(save_path))
    metadata = get_dataset_metadata(ds, local_name, save_path)
    log_dataset_version(local_name, save_path, metadata)
    print(f"  → {ds}, SHA256: {metadata['sha256'][:16]}...")
    return ds


def download_all():
    """Orchestrateur : télécharge tous les datasets en séquence"""
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