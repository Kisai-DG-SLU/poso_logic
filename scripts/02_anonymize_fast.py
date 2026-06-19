"""
Script d'anonymisation optimisé - Version batchée avec sampling et reprise
Étape 1 - Conformité RGPD avec Microsoft Presidio
Sauvegarde incrémentale en JSONL pour reprise après interruption
"""

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from datasets import load_from_disk, Dataset
from pathlib import Path
from tqdm import tqdm
import json
import sys
import warnings
warnings.filterwarnings("ignore")

# Répertoires de données
RAW_DIR = Path("/mnt/prod/data/raw")
PROCESSED_DIR = Path("/mnt/prod/data/processed")

# Configuration : comment remplacer chaque type d'entité détectée
ANONYMIZER_CONFIG = {
    "PERSON": OperatorConfig("replace", {"new_value": "<ANONYMIZED_PERSON>"}),
    "PHONE_NUMBER": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 12, "from_end": True}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<ANONYMIZED_EMAIL>"}),
    "LOCATION": OperatorConfig("replace", {"new_value": "<ANONYMIZED_LOCATION>"}),
    "DATE_TIME": OperatorConfig("redact", {}),
    "US_SSN": OperatorConfig("redact", {}),
    "CREDIT_CARD": OperatorConfig("redact", {}),
    "IP_ADDRESS": OperatorConfig("redact", {}),
    "DEFAULT": OperatorConfig("replace", {"new_value": "<ANONYMIZED>"})
}

# Types d'entités personnelles que Presidio doit détecter
ANALYZER_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", "DATE_TIME", "US_SSN", "CREDIT_CARD", "IP_ADDRESS"]

# Limite d'échantillons par dataset (pour rester dans les limites mémoire et temps)
MAX_SAMPLES_PER_DATASET_EN = 50000
MAX_SAMPLES_PER_DATASET_FR = 50000


def extract_fields(ex, dataset_name):
    """Extrait les champs texte à anonymiser selon la structure du dataset"""
    if dataset_name == "medquad":
        return {
            "question": str(ex.get("question", "")),
            "answer": str(ex.get("answer", "")),
        }
    elif dataset_name == "ultramedical":
        convs = ex.get("conversations", [])
        human = str(convs[0].get("value", "")) if len(convs) > 0 and isinstance(convs[0], dict) else ""
        gpt = str(convs[1].get("value", "")) if len(convs) > 1 and isinstance(convs[1], dict) else ""
        return {"human": human, "gpt": gpt}
    elif dataset_name == "ultramedical_preference":
        prompt = str(ex.get("prompt", ""))
        chosen = " ".join(
            str(m.get("content", "")) for m in ex.get("chosen", [])
            if isinstance(m, dict)
        )
        rejected = " ".join(
            str(m.get("content", "")) for m in ex.get("rejected", [])
            if isinstance(m, dict)
        )
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    elif dataset_name == "frbmedqa":
        return {
            "passage": str(ex.get("passage", "")),
            "question": str(ex.get("question", "")),
            "answer": str(ex.get("answer", "")),
        }
    elif dataset_name in ("mediqal_mcqu", "mediqal_mcqm"):
        fields = {"clinical_case": str(ex.get("clinical_case", ""))}
        for k in ["question", "answer_a", "answer_b", "answer_c", "answer_d", "answer_e"]:
            if k in ex:
                fields[k] = str(ex.get(k, ""))
        return fields
    elif dataset_name == "mediqal_oeq":
        return {
            "clinical_case": str(ex.get("clinical_case", "")),
            "question": str(ex.get("question", "")),
            "answer": str(ex.get("answer", "")),
        }
    else:
        return {"question": str(ex.get("question", "")), "answer": str(ex.get("answer", ""))}


class Anonymizer:
    """Wrapper autour de Presidio : détection + anonymisation des PII dans du texte"""
    def __init__(self):
        print("Initialisation Presidio (CPU)...", flush=True)
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        print("  → Presidio prêt (CPU)", flush=True)

    def process_batch(self, texts, language="en"):
        """Analyse puis anonymise une liste de textes (batch) en une passe"""
        results = []
        for text in texts:
            if not text or not isinstance(text, str):
                results.append("")
                continue
            try:
                # Étape 1 : détection des PII par Presidio
                analyzer_results = self.analyzer.analyze(
                    text=text,
                    language=language,
                    entities=ANALYZER_ENTITIES
                )
                # Étape 2 : remplacement par les balises configurées
                anonymized = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=analyzer_results,
                    operators=ANONYMIZER_CONFIG
                )
                results.append(anonymized.text)
            except Exception:
                results.append(text)  # En cas d'erreur, on garde le texte original
        return results


def count_existing(jsonl_path):
    """Compte le nombre de lignes déjà écrites dans le JSONL (pour la reprise après interruption)"""
    if not jsonl_path.exists():
        return 0
    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def anonymize_dataset_resumable(dataset, name, language="en", max_samples=None, jsonl_path=None):
    """
    Parcourt le dataset par lots de 100, anonymise chaque champ via Presidio,
    sauvegarde au fur et à mesure dans un JSONL (permet la reprise après interruption)
    """
    anonymizer = Anonymizer()

    if max_samples is None:
        max_samples = len(dataset)
    total = min(len(dataset), max_samples)

    # Reprise : on compte les lignes déjà écrites dans le JSONL
    start_idx = 0
    if jsonl_path and jsonl_path.exists():
        start_idx = count_existing(jsonl_path)
        if start_idx > 0:
            print(f"  → Reprise détectée: {start_idx} exemples déjà traités", flush=True)

    remaining = total - start_idx
    if remaining <= 0:
        print(f"  → Dataset déjà entièrement traité ({total} exemples)", flush=True)
        return start_idx

    print(f"\n{name}: anonymisation de {remaining} exemples restants (sur {total}, déjà fait: {start_idx})", flush=True)

    batch_size = 100  # Taille de lot : on traite 100 exemples à la fois

    for i in tqdm(range(start_idx, total, batch_size),
                  desc=f"Anonymisation {name}",
                  initial=start_idx // batch_size,
                  total=total // batch_size + (1 if total % batch_size else 0)):
        end_idx = min(i + batch_size, total)

        # Extraction des champs pour le lot courant
        batch_extracted = []
        for j in range(i, end_idx):
            batch_extracted.append(extract_fields(dataset[j], name))

        # Organisation des données par champ pour l'anonymisation en batch
        field_names = list(batch_extracted[0].keys())
        per_field_values = {fn: [] for fn in field_names}
        for ex_fields in batch_extracted:
            for fn in field_names:
                per_field_values[fn].append(ex_fields[fn])

        # Anonymisation de chaque champ (appels Presidio en série)
        batch_anon = {}
        for fn in field_names:
            batch_anon[fn] = anonymizer.process_batch(per_field_values[fn], language=language)

        # Sauvegarde incrémentale dans le JSONL
        if jsonl_path:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                for j in range(len(batch_extracted)):
                    record = {fn: batch_anon[fn][j] for fn in field_names}
                    parts = [batch_anon[fn][j] for fn in field_names if batch_anon[fn][j].strip()]
                    record["text"] = " ".join(parts)  # Texte complet concaténé (tous champs)
                    record["original_idx"] = i + j
                    record["language"] = language
                    record["source"] = name
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

        # Rapport de progression toutes les 1000 itérations
        if (i - start_idx) % 1000 == 0 and i > start_idx:
            done_so_far = i + batch_size
            print(f"  → {name}: {done_so_far}/{total} ({100*done_so_far//total}%)", flush=True)

    final_count = count_existing(jsonl_path) if jsonl_path else total
    print(f"  → {name}: terminé ({final_count} exemples)", flush=True)
    return final_count


def jsonl_to_hf_dataset(jsonl_path, output_path):
    """Convertit le fichier JSONL en dataset HuggingFace (format .arrow)"""
    print(f"  → Conversion JSONL → HF Dataset...", flush=True)
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        print("  → Aucun enregistrement à convertir", flush=True)
        return

    # Transformation : liste de dicts → dict de listes (format Dataset.from_dict)
    output_dict = {k: [r[k] for r in records] for k in records[0].keys()}
    ds = Dataset.from_dict(output_dict)
    ds.save_to_disk(str(output_path))
    print(f"  → Sauvegardé: {output_path} ({len(ds)} exemples, champs: {list(records[0].keys())})", flush=True)


def process_all_datasets():
    """Orchestrateur : traite chaque dataset brut → anonymisé → HF Dataset"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Configuration : (nom_dataset, langue, nombre_max_échantillons)
    datasets_config = [
        ("medquad", "en", MAX_SAMPLES_PER_DATASET_EN),
        ("ultramedical", "en", MAX_SAMPLES_PER_DATASET_EN),
        ("ultramedical_preference", "en", MAX_SAMPLES_PER_DATASET_EN),
        ("frbmedqa", "fr", MAX_SAMPLES_PER_DATASET_FR),
        ("mediqal_mcqu", "fr", MAX_SAMPLES_PER_DATASET_FR),
        ("mediqal_mcqm", "fr", MAX_SAMPLES_PER_DATASET_FR),
        ("mediqal_oeq", "fr", MAX_SAMPLES_PER_DATASET_FR),
    ]

    for name, lang, max_samples in datasets_config:
        input_path = RAW_DIR / name

        if not input_path.exists():
            print(f"\n  → {name}: non trouvé, ignoré", flush=True)
            continue

        output_path = PROCESSED_DIR / f"{name}_anon"

        # Vérification : déjà traité (présence de fichiers .arrow = HF Dataset complet)
        if output_path.exists() and list(output_path.glob("*.arrow")):
            print(f"\n  → {name}: déjà traité, ignoré", flush=True)
            continue

        output_path.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_path / "data.jsonl"

        print(f"\nTraitement {name} ({lang})...", flush=True)
        ds = load_from_disk(str(input_path))

        split_name = "train" if "train" in ds else list(ds.keys())[0]
        split_data = ds[split_name]

        count = anonymize_dataset_resumable(split_data, name, lang, max_samples, jsonl_path)

        if count > 0:
            jsonl_to_hf_dataset(jsonl_path, output_path)
            jsonl_path.unlink()  # Nettoyage du JSONL temporaire après conversion
        else:
            print(f"  → {name}: rien à convertir", flush=True)

    print("\n✅ Tous les datasets ont été traités avec succès !", flush=True)


if __name__ == "__main__":
    process_all_datasets()
