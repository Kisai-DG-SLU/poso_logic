"""
Script d'anonymisation optimisé - Traitement par lot
"""

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from datasets import load_from_disk, DatasetDict
from pathlib import Path
import json

RAW_DIR = Path("/mnt/prod/data/raw")
PROCESSED_DIR = Path("/mnt/prod/data/processed")

def anonymize_batch(texts, language="en"):
    """Anonymise un batch de textes"""
    if not texts:
        return []
    
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    
    results = []
    for text in texts:
        if not text:
            results.append(text)
            continue
            
        analyzer_results = analyzer.analyze(
            text=text,
            language=language,
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", "DATE_TIME"]
        )
        
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"})}
        )
        results.append(anonymized.text)
    
    return results

def process_sampled(input_path, output_path, sample_size=5000, language="en"):
    """Traite un échantillon du dataset"""
    print(f"Chargement {input_path.name}...")
    ds = load_from_disk(str(input_path))
    
    train = ds.get("train", ds.get("train", ds["train"]))
    
    if len(train) > sample_size:
        print(f"Échantillonnage de {sample_size} sur {len(train)}...")
        train = train.select(range(sample_size))
    
    print("Anonymisation...")
    questions = [ex.get("question", "") for ex in train]
    anonymized_questions = anonymize_batch(questions, language)
    
    answers = [ex.get("answer", "") for ex in train]
    anonymized_answers = anonymize_batch(answers, language)
    
    new_features = train.features.copy()
    new_features["text_anonymized"] = {"dtype": "string", "_type": "Value"}
    
    anonymized_data = []
    for i, ex in enumerate(train):
        new_ex = dict(ex)
        new_ex["text_anonymized"] = anonymized_questions[i] + " " + anonymized_answers[i]
        anonymized_data.append(new_ex)
    
    from datasets import Dataset
    anonymized_ds = Dataset.from_list(anonymized_data)
    
    print(f"Sauvegarde vers {output_path}...")
    anonymized_ds.save_to_disk(str(output_path))
    
    return anonymized_ds

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = [
        ("medquad", "en"),
        ("ultramedical", "en"),
        ("ultramedical_preference", "en"),
    ]
    
    for name, lang in datasets:
        input_path = RAW_DIR / name
        output_path = PROCESSED_DIR / f"{name}_anon"
        
        if input_path.exists():
            try:
                process_sampled(input_path, output_path, sample_size=5000, language=lang)
                print(f"✓ {name}")
            except Exception as e:
                print(f"✗ {name}: {e}")

if __name__ == "__main__":
    main()