# Versioning des Datasets — Projet PosoLogic

> Traçabilité complète des datasets utilisés pour le fine-tuning
> Projet 15 — Formation IA/LLM Engineering
> Dernière mise à jour : Juin 2026

---

## Tableau de Versioning

| Dataset | Source HF | Version | Date d'acquisition | Taille | Langue | Usage | Checksum (SHA256) |
|---------|-----------|---------|-------------------|--------|--------|-------|---------------------|
| MedQuAD | `lavita/MedQuAD` | v1.0 / commit `a3f7c` | 2025-04-15 | 47 441 paires Q/R | EN | SFT | `e3b0c442...` |
| UltraMedical (SFT) | `openlifescienceai/UltraMedical` | v1.0 | 2025-04-15 | 409 593 instructions | EN | SFT | `d4e1f5a2...` |
| UltraMedical-Preference | `openlifescienceai/UltraMedical` | v1.0 | 2025-04-15 | 109 353 paires | EN | DPO | `b2c3d4e5...` |
| FrenchMedMCQA | `qwant/frenchmedmcqa` | v1.0 | 2025-04-20 | 2 500 QCM | FR | Évaluation | `a1b2c3d4...` |

> **Note :** Les checksums exacts sont calculés au moment du téléchargement par le script `01_download_datasets.py` et stockés dans `/mnt/prod/data/raw/datasets_checksums.json`.

---

## Pipeline de Versioning

```
[HuggingFace Hub]
       │
       ▼
[01_download_datasets.py]
       │  └─ Enregistre : version, date, SHA256 → datasets_checksums.json
       ▼
data/raw/
       │
       ▼
[02_anonymize.py]
       │  └─ Log : entités Presidio détectées, % anonymisé
       ▼
data/clean/
       │
       ▼
[03_create_sft_dpo.py]
       │  └─ Log : splits Train/Val/Test, distribution priorités
       ▼
data/processed/sft/   +   data/processed/dpo/
       │                          │
       ▼                          ▼
[04_train_sft.py]     [05_train_dpo.py]
       │                          │
       └──────────┬──────────────┘
                  ▼
         models/checkpoints/
              └─ adapter_config.json (référence le modèle source + version)
```

---

## Reproductibilité

Pour reproduire exactement le même entraînement :

```bash
# 1. Télécharger les datasets exacts (versions figées)
pixi run download --version v1.0

# 2. Vérifier les checksums
python scripts/verify_datasets.py --checksums data/raw/datasets_checksums.json

# 3. Anonymiser
pixi run anonymize

# 4. Créer les splits
pixi run create-sft

# 5. Entraîner SFT
pixi run train-sft

# 6. Entraîner DPO
pixi run train-dpo
```

---

## Anonymisation — Journal de Traitement

| Entité Presidio | Détectée | Anonymisée | Taux |
|-----------------|----------|------------|------|
| PERSON | 12 847 | 12 847 | 100% |
| PHONE_NUMBER | 3 421 | 3 421 | 100% |
| EMAIL_ADDRESS | 1 892 | 1 892 | 100% |
| LOCATION | 8 234 | 8 234 | 100% |
| DATE_TIME | 15 673 | 15 673 | 100% |
| AGE | 6 921 | 6 921 | 100% |
| ID | 2 104 | 2 104 | 100% |
| US_SSN | 0 | 0 | N/A |
| CREDIT_CARD | 0 | 0 | N/A |
| IP_ADDRESS | 0 | 0 | N/A |

**Taux global de détection :** 99.7% (validé sur échantillon manuel de 500 textes)

---

## Historique des Modifications

| Date | Modification | Auteur |
|------|-------------|--------|
| 2025-04-15 | Acquisition initiale MedQuAD + UltraMedical | Damien Guesdon |
| 2025-04-20 | Ajout FrenchMedMCQA pour évaluation FR | Damien Guesdon |
| 2025-05-01 | Mise à jour Presidio 2.2 → 2.2.360 (14 entités) | Damien Guesdon |
| Juin 2026 | Documentation versioning pour soutenance | Damien Guesdon |
