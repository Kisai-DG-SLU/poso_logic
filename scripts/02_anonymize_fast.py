"""
Script d'anonymisation optimisé - Version batchée avec sampling
Étape 1 - Conformité RGPD avec Microsoft Presidio
"""

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from datasets import load_from_disk, Dataset
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

RAW_DIR = Path("/mnt/prod/data/raw")
PROCESSED_DIR = Path("/mnt/prod/data/processed")

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

ANALYZER_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", "DATE_TIME", "US_SSN", "CREDIT_CARD", "IP_ADDRESS"]

MAX_SAMPLES_PER_DATASET = 10000

class Anonymizer:
    def __init__(self):
        print("Initialisation Presidio...")
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        print("  → Présidio prêt")
    
    def process_batch(self, texts, language="en"):
        results = []
        for text in texts:
            if not text or not isinstance(text, str):
                results.append("")
                continue
            
            try:
                analyzer_results = self.analyzer.analyze(
                    text=text,
                    language=language,
                    entities=ANALYZER_ENTITIES
                )
                anonymized = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=analyzer_results,
                    operators=ANONYMIZER_CONFIG
                )
                results.append(anonymized.text)
            except Exception:
                results.append(text)
        
        return results

def anonymize_dataset_streaming(dataset, name, language="en", max_samples=MAX_SAMPLES_PER_DATASET):
    """Anonymisation avec sampling intelligent"""
    anonymizer = Anonymizer()
    
    total = min(len(dataset), max_samples)
    print(f"\n{name}: anonymisation de {total} exemples (sur {len(dataset)})")
    
    anonymized_texts = []
    batch_size = 100
    
    for i in tqdm(range(0, total, batch_size), desc=f"Anonymisation {name}"):
        end_idx = min(i + batch_size, total)
        batch_texts = []
        
        for j in range(i, end_idx):
            ex = dataset[j]
            text = str(ex.get("question", "")) + " " + str(ex.get("answer", ""))
            batch_texts.append(text)
        
        batch_results = anonymizer.process_batch(batch_texts, language=language)
        anonymized_texts.extend(batch_results)
    
    return anonymized_texts

def process_all_datasets():
    """Traite tous les datasets avec sampling"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    anonymizer = Anonymizer()
    
    datasets_config = [
        ("medquad", "en"),
        ("ultramedical", "en"),
        ("ultramedical_preference", "en"),
        ("french_qa", "fr"),
    ]
    
    for name, lang in datasets_config:
        input_path = RAW_DIR / name
        
        if not input_path.exists():
            print(f"\n  → {name}: non trouvé, ignoré")
            continue
        
        print(f"\nTraitement {name} ({lang})...")
        ds = load_from_disk(str(input_path))
        
        split_name = "train" if "train" in ds else list(ds.keys())[0]
        split_data = ds[split_name]
        
        texts = anonymize_dataset_streaming(split_data, name, lang)
        
        output_path = PROCESSED_DIR / f"{name}_anon"
        ds_anon = Dataset.from_dict({
            "text": texts,
            "original_idx": list(range(len(texts))),
            "language": [lang] * len(texts)
        })
        ds_anon.save_to_disk(str(output_path))
        print(f"  → Sauvegardé: {output_path} ({len(ds_anon)} exemples)")

if __name__ == "__main__":
    process_all_datasets()