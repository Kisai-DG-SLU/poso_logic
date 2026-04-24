"""
Script création dataset SFT et DPO - Version optimisée
"""

from datasets import load_from_disk, Dataset, DatasetDict
from pathlib import Path

PROCESSED_DIR = Path("/mnt/prod/data/processed")

def create_sft():
    """Crée dataset SFT ~5000 paires"""
    print("Création SFT...")
    
    sft_data = []
    
    sources = [
        ("medquad_anon", "medquad"),
        ("ultramedical_anon", "ultramedical"),
    ]
    
    for folder, name in sources:
        path = PROCESSED_DIR / folder
        if path.exists():
            ds = load_from_disk(str(path))
            print(f"  → {name}: {len(ds)} exemples")
            
            for ex in ds:
                q = ex.get("question", "")
                a = ex.get("answer", "")
                
                if q and a:
                    sft_data.append({
                        "instruction": f"Vous êtes un assistant de triage médical. Question: {q}",
                        "response": a,
                        "source": name,
                        "language": "en",
                        "priority_level": "medium"
                    })
    
    TARGET = 5000
    if len(sft_data) > TARGET:
        sft_data = sft_data[:TARGET]
    
    print(f"  → Total: {len(sft_data)} paires SFT")
    
    ds = Dataset.from_list(sft_data)
    splits = ds.train_test_split(test_size=0.15, seed=42)
    val_test = splits["test"].train_test_split(test_size=0.5, seed=42)
    
    result = DatasetDict({
        "train": splits["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })
    
    result.save_to_disk(str(PROCESSED_DIR / "sft_dataset"))
    print(f"  → Sauvegardé: train={len(result['train'])}, val={len(result['validation'])}, test={len(result['test'])}")
    
    return result

def create_dpo():
    """Crée dataset DPO"""
    print("Création DPO...")
    
    raw_path = PROCESSED_DIR.parent / "raw" / "ultramedical_preference"
    
    if not raw_path.exists():
        print("  → Dataset brut non trouvé, création manuelle...")
        dpo_data = [
            {
                "instruction": "Patient présente douleur thoracique, que faire?",
                "chosen": "Appeler le 15 immédiatement. Urgence vitale possible.",
                "rejected": "Donner un paracétamol.",
                "source": "manual",
                "language": "fr"
            }
        ] * 100
        
        for i in range(400):
            dpo_data.append({
                "instruction": f"Question médicale {i}: Symptômes gripaux",
                "chosen": "Repos, hydratation, paracétamol",
                "rejected": "Antibiotiques immédiat",
                "source": "synthetic",
                "language": "fr"
            })
    else:
        ds = load_from_disk(str(raw_path))
        train = ds.get("train", ds["train"]).select(range(min(5000, len(ds["train"]))))
        
        dpo_data = []
        for ex in train:
            chosen = ex.get("chosen", "")
            rejected = ex.get("rejected", "")
            prompt = ex.get("prompt", "")
            
            if chosen and rejected:
                dpo_data.append({
                    "instruction": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": "ultramedical_pref",
                    "language": "en"
                })
    
    print(f"  → {len(dpo_data)} paires DPO")
    
    ds = Dataset.from_list(dpo_data)
    splits = ds.train_test_split(test_size=0.1, seed=42)
    
    result = DatasetDict({
        "train": splits["train"],
        "validation": splits["test"]
    })
    
    result.save_to_disk(str(PROCESSED_DIR / "dpo_dataset"))
    print(f"  → Sauvegardé")
    
    return result

def main():
    create_sft()
    create_dpo()
    print("\n=== Étape 1 terminée ===")

if __name__ == "__main__":
    main()