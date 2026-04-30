"""
Script d'anonymisation des données médicales
Étape 1 - Conformité RGPD avec Microsoft Presidio
"""

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import spacy
from datasets import load_from_disk
from pathlib import Path
import json

RAW_DIR = Path("/mnt/prod/data/raw")
PROCESSED_DIR = Path("/mnt/prod/data/processed")

def load_nlp_models():
    """Charge les modèles spaCy pour FR et EN"""
    print("Chargement modèles spaCy...")
    try:
        nlp_fr = spacy.load("fr_core_news_md")
        print("  → fr_core_news_md chargé")
    except OSError:
        print("  → fr_core_news_md non trouvé, téléchargement...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"])
        nlp_fr = spacy.load("fr_core_news_md")
    
    try:
        nlp_en = spacy.load("en_core_web_lg")
        print("  → en_core_web_lg chargé")
    except OSError:
        print("  → en_core_web_lg non trouvé, téléchargement...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
        nlp_en = spacy.load("en_core_web_lg")
    
    return nlp_fr, nlp_en

def analyze_text(text, language="en"):
    """Analyse le texte pour détecter les entités PII"""
    analyzer = AnalyzerEngine()
    
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", "DATE_TIME", "US_SSN", "CREDIT_CARD", "IP_ADDRESS"]
    )
    return results

def anonymize_text(text, analyzer_results):
    """Anonymise le texte avec les entités détectées"""
    anonymizer = AnonymizerEngine()
    
    result = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators={
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
    )
    return result.text

def anonymize_dataset(dataset, language="en"):
    """Anonymise un entire dataset"""
    anonymized = []
    
    for i, example in enumerate(dataset):
        text = example.get("question", "") + " " + example.get("answer", "")
        
        analyzer_results = analyze_text(text, language=language)
        anonymized_text = anonymize_text(text, analyzer_results)
        
        example["text_anonymized"] = anonymized_text
        example["pii_detected"] = len(analyzer_results) > 0
        
        anonymized.append(example)
        
        if (i + 1) % 100 == 0:
            print(f"  → {i+1}/{len(dataset)} exemples traités")
    
    return anonymized

def process_all_datasets():
    """Traite tous les datasets téléchargés"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    nlp_fr, nlp_en = load_nlp_models()
    
    datasets = [
        ("frenchmedmcqa", "fr"),
        ("mediqa_qa", "en"),
        ("medquad", "en"),
        ("ultramedical", "en"),
        ("ultramedical_preference", "en"),
    ]
    
    for name, lang in datasets:
        input_path = RAW_DIR / name
        output_path = PROCESSED_DIR / f"{name}_anonymized"
        
        if input_path.exists():
            print(f"\nTraitement {name}...")
            ds = load_from_disk(str(input_path))
            
            if "train" in ds:
                ds["train"] = anonymize_dataset(ds["train"], language=lang)
            if "test" in ds:
                ds["test"] = anonymize_dataset(ds["test"], language=lang)
            
            ds.save_to_disk(str(output_path))
            print(f"  → Saves vers {output_path}")
        else:
            print(f"  → {name} non trouvé, à télécharger d'abord")

if __name__ == "__main__":
    process_all_datasets()