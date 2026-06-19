"""
Création des datasets SFT et DPO depuis les données anonymisées
Produit data/processed/sft_dataset et data/processed/dpo_dataset
"""

from datasets import Dataset, DatasetDict, load_from_disk
from pathlib import Path
import random

# Répertoire contenant les datasets anonymisés (input) et cible des datasets SFT/DPO (output)
PROCESSED_DIR = Path("/mnt/prod/data/processed")
OUT_DIR = Path("/mnt/prod/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Nombre maximal d'exemples pour le dataset SFT (limitation mémoire / temps d'entraînement)
N_SFT_MAX = 5000
# Nombre de scénarios de triage synthétiques FR ajoutés au dataset DPO
N_DPO_SYNTHETIC = 100


def create_sft():
    """Construit le dataset SFT : paires instruction/réponse à partir des données anonymisées"""
    print("Création dataset SFT...")
    sft_data = []

    # MedQuAD (EN, question/answer) — questions médicales générales
    p = PROCESSED_DIR / "medquad_anon"
    if p.exists():
        ds = load_from_disk(str(p))
        print(f"  MedQuAD: {len(ds)}")
        for ex in ds:
            q, a = str(ex.get("question", "")), str(ex.get("answer", ""))
            if q and a:
                sft_data.append({
                    "instruction": f"You are a medical triage assistant. Question: {q[:1000]}",
                    "response": a[:500],
                    "source": "medquad", "language": "en",
                })

    # UltraMedical (EN, human/gpt) — conversations médicales générales
    p = PROCESSED_DIR / "ultramedical_anon"
    if p.exists():
        ds = load_from_disk(str(p))
        print(f"  UltraMedical: {len(ds)}")
        for ex in ds:
            h, g = str(ex.get("human", "")), str(ex.get("gpt", ""))
            if h and g:
                sft_data.append({
                    "instruction": f"You are a medical triage assistant. Question: {h[:1000]}",
                    "response": g[:500],
                    "source": "ultramedical", "language": "en",
                })

    # FrBMedQA (FR, question/answer) — données médicales en français
    p = PROCESSED_DIR / "frbmedqa_anon"
    if p.exists():
        ds = load_from_disk(str(p))
        print(f"  FrBMedQA: {len(ds)}")
        for ex in ds:
            q, a = str(ex.get("question", "")), str(ex.get("answer", ""))
            if q and a:
                sft_data.append({
                    "instruction": f"Vous êtes un assistant de triage médical. Question: {q[:1000]}",
                    "response": a[:500],
                    "source": "frbmedqa", "language": "fr",
                })

    print(f"  Total SFT brutes: {len(sft_data)}")
    sft_data = sft_data[:N_SFT_MAX]  # Limite à 5000 exemples

    # Split train/validation/test (80/10/10)
    ds = Dataset.from_list(sft_data)
    splits = ds.train_test_split(test_size=0.1, seed=42)
    val_test = splits["test"].train_test_split(test_size=0.5, seed=42)
    result = DatasetDict({
        "train": splits["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })
    result.save_to_disk(str(OUT_DIR / "sft_dataset"))
    print(f"  SFT sauvegardé: train={len(result['train'])}, val={len(result['validation'])}, test={len(result['test'])}")


def create_dpo():
    """Construit le dataset DPO : paires (instruction, chosen, rejected) pour l'alignement par préférences"""
    print("Création dataset DPO...")
    dpo_data = []

    # UltraMedical-Preference (EN) — paires préférée/rejetée issues de feedback expert
    p = PROCESSED_DIR / "ultramedical_preference_anon"
    if p.exists():
        ds = load_from_disk(str(p))
        print(f"  UltraMedical-Preference: {len(ds)}")
        for ex in ds:
            prompt = str(ex.get("prompt", "")) or ""
            chosen_text = str(ex.get("chosen", ""))
            rejected_text = str(ex.get("rejected", ""))
            if prompt and chosen_text and rejected_text and len(chosen_text) > 5:
                dpo_data.append({
                    "instruction": prompt,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "source": "ultramedical_pref", "language": "en",
                })

    # Scénarios synthétiques FR (démonstration de triage médical pour le POC)
    scenarios = [
        ("douleur thoracique", "Appeler le 15 immédiatement. Urgence vitale possible.", "Donner un paracétamol et attendre."),
        ("difficulté respiratoire", "Appeler le 15. Monitorer la saturation O2.", "Prescrire des antibiotiques sans examen."),
        ("perte de conscience", "Appeler le 15 et vérifier les constantes vitales.", "Demander au patient de rentrer chez lui."),
        ("fièvre + raideur nuque", "Appeler le 15. Suspicion méningite.", "Donner du repos et surveiller."),
        ("saignement abondant", "Appeler le 15. Appliquer compression directe.", "Attendre sans intervention."),
    ]
    random.seed(42)
    for s in random.choices(scenarios, k=N_DPO_SYNTHETIC):
        dpo_data.append({
            "instruction": f"Patient présente: {s[0]}",
            "chosen": s[1],
            "rejected": s[2],
            "source": "synthetic_fr", "language": "fr",
        })

    print(f"  Total DPO: {len(dpo_data)}")
    ds = Dataset.from_list(dpo_data)
    splits = ds.train_test_split(test_size=0.1, seed=42)
    result = DatasetDict({
        "train": splits["train"],
        "validation": splits["test"],
    })
    result.save_to_disk(str(OUT_DIR / "dpo_dataset"))
    print(f"  DPO sauvegardé: train={len(result['train'])}, val={len(result['validation'])}")


if __name__ == "__main__":
    create_sft()
    create_dpo()
    print("Terminé.")
