"""
Création des datasets SFT et DPO depuis les données brutes HuggingFace
Produit data/processed/sft_dataset et data/processed/dpo_dataset
"""

from datasets import Dataset, DatasetDict, load_from_disk
from pathlib import Path
import random

RAW_DIR = Path("/mnt/prod/data/raw")
OUT_DIR = Path("/mnt/prod/data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SFT_MAX = 5000
N_DPO_SYNTHETIC = 100


def create_sft():
    print("Création dataset SFT...")
    sft_data = []

    # MedQuAD (EN, question/answer)
    p = RAW_DIR / "medquad"
    if p.exists():
        ds = load_from_disk(str(p))["train"]
        print(f"  MedQuAD: {len(ds)}")
        for ex in ds:
            q, a = str(ex.get("question", "")), str(ex.get("answer", ""))
            if q and a:
                sft_data.append({
                    "instruction": f"You are a medical triage assistant. Question: {q[:1000]}",
                    "response": a[:500],
                    "source": "medquad", "language": "en",
                })

    # UltraMedical (EN, conversations)
    p = RAW_DIR / "ultramedical"
    if p.exists():
        ds = load_from_disk(str(p))["train"]
        print(f"  UltraMedical: {len(ds)}")
        for ex in ds:
            convs = ex.get("conversations", [])
            if isinstance(convs, list) and len(convs) >= 2:
                h = convs[0].get("value", "") if isinstance(convs[0], dict) else ""
                g = convs[1].get("value", "") if isinstance(convs[1], dict) else ""
                if h and g:
                    sft_data.append({
                        "instruction": f"You are a medical triage assistant. Question: {h[:1000]}",
                        "response": g[:500],
                        "source": "ultramedical", "language": "en",
                    })

    # FrBMedQA (FR, question/answer)
    p = RAW_DIR / "frbmedqa"
    if p.exists():
        ds = load_from_disk(str(p))["train"]
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
    sft_data = sft_data[:N_SFT_MAX]

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
    print("Création dataset DPO...")
    dpo_data = []

    # UltraMedical-Preference (EN)
    p = RAW_DIR / "ultramedical_preference"
    if p.exists():
        ds = load_from_disk(str(p))["train"]
        print(f"  UltraMedical-Preference: {len(ds)}")
        for ex in ds:
            prompt = str(ex.get("prompt", "")) or ""
            chosen_conv = ex.get("chosen", [])
            rejected_conv = ex.get("rejected", [])
            if isinstance(chosen_conv, list) and isinstance(rejected_conv, list):
                chosen_text = " ".join(
                    m.get("content", "") for m in chosen_conv
                    if isinstance(m, dict) and m.get("role") == "assistant"
                )
                rejected_text = " ".join(
                    m.get("content", "") for m in rejected_conv
                    if isinstance(m, dict) and m.get("role") == "assistant"
                )
            else:
                chosen_text = str(chosen_conv) if chosen_conv else ""
                rejected_text = str(rejected_conv) if rejected_conv else ""
            if prompt and chosen_text and rejected_text and len(chosen_text) > 5:
                dpo_data.append({
                    "instruction": prompt,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "source": "ultramedical_pref", "language": "en",
                })

    # Scénarios synthétiques FR (démonstration triage)
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
