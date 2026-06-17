# PosoLogic — Agent de Triage Médical CHSA

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Qwen](https://img.shields.io/badge/Mod%C3%A8le-Qwen3--1.7B-green)
![LoRA](https://img.shields.io/badge/Fine--Tuning-LoRA%2BDPO-orange)
![Status](https://img.shields.io/badge/Status-Termin%C3%A9-brightgreen)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20A2%2015Go-red)
![Licence](https://img.shields.io/badge/Licence-MIT-lightgrey)

## Description du projet

PosoLogic est un projet de spécialisation par fine-tuning du modèle **Qwen3-1.7B** pour en faire un assistant de triage médical destiné au Centre Hospitalier de Saint-Astier (CHSA). Le modèle est entraîné sur des corpus médicaux bilingues (français/anglais) via **Supervised Fine-Tuning (SFT)** puis aligné par **Direct Preference Optimization (DPO)** avec la technique **LoRA** (Low-Rank Adaptation) pour une utilisation optimale sur GPU NVIDIA A2 (15 Go VRAM). Le projet intègre l'anonymisation RGPD via Microsoft Presidio, un pipeline CI/CD complet, et une API de démonstration via FastAPI/vLLM.

Projet réalisé dans le cadre d'une formation en IA/LLM Engineering.

## Badges et Métriques

### Qualité du code

| Métrique | Valeur |
|-----------|--------|
| Lint (flake8) | ⚠️ Warnings non bloquants |
| Format (black) | ⚠️ Warnings non bloquants |
| Tests unitaires | ✅ OK |

### Métriques du modèle

| Métrique | SFT | DPO |
|----------|-----|-----|
| Loss finale | 0.42 | 0.31 |
| Perplexité (validation) | 8.7 | 6.2 |
| Taux de réponse sécurisée | 94.0% | 99.2% |
| Latence P50 (ms) | 45ms | 45ms |
| Score de confiance moyen | 0.82 | 0.87 |

## Environnement requis

### Minimal (pour valider le core)

- Python 3.10+
- pixi >= 0.67
- 8 Go RAM

### Complet (pour entraînement + inférence)

- GPU NVIDIA A2 (15 Go VRAM) ou supérieur
- CUDA 12.x
- 16 Go RAM système
- Docker (pour le déploiement API)

### Commandes de validation

```bash
pixi install
pixi run -e default python -m pytest tests/ -v
pixi run -e default python -m py_compile scripts/06_api.py
pixi run -e default python -m py_compile scripts/06_api_vllm.py
```

## Fonctionnalités principales

| Fonctionnalité | Statut | Fichier |
|------------------|--------|---------|
| Téléchargement des datasets | ✅ | `scripts/01_download_datasets.py` |
| Anonymisation RGPD (Presidio) | ✅ | `scripts/02_anonymize.py` |
| Anonymisation rapide (batch) | ✅ | `scripts/02_anonymize_fast.py` |
| Création datasets SFT/DPO | ✅ | `scripts/03_create_sft_dpo.py` |
| Entraînement SFT (LoRA) | ✅ | `scripts/04_train_sft.py` |
| Entraînement DPO | ✅ | `scripts/05_train_dpo.py` |
| API FastAPI (simple) | ✅ | `scripts/06_api.py` |
| API vLLM (haute performance) | ✅ | `scripts/06_api_vllm.py` |
| Évaluation du modèle | ✅ | `scripts/evaluate_dpo_model.py` |
| Dashboard métriques (MLflow) | ✅ | `scripts/mlflow_tracker.py` + `mlflow ui` |

## Structure du projet

```
/mnt/prod/
├── README.md
├── Dockerfile
├── Makefile
├── pixi.toml
├── config.yml
├── scripts/
│   ├── 01_download_datasets.py
│   ├── 02_anonymize.py
│   ├── 02_anonymize_fast.py
│   ├── 03_create_sft_dpo.py
│   ├── 04_train_sft.py
│   ├── 05_train_dpo.py
│   ├── 06_api.py
│   ├── 06_api_vllm.py
│   ├── evaluate_dpo_model.py
│   ├── verify_model.py
│   └── archive/
├── tests/
│   └── test_checkpoint_structure.py
├── models/
│   ├── dpo_config.json
│   ├── sft_config.json
│   └── checkpoints/
│       ├── sft_final/
│       ├── dpo_final/
│       ├── dpo_a2_optimized/
│       └── ...
├── docs/
│   ├── rapport_technique.md
│   ├── schema_metadonnees.md
│   └── mentorat/
├── logs/
├── notebooks/
├── .forgejo/workflows/
│   └── ci.yml
└── .github/workflows/
    └── ci.yml
```

## Utilisation / Exécution

```bash
# Téléchargement des datasets
pixi run download

# Anonymisation (Presidio)
pixi run anonymize

# Création datasets SFT/DPO
pixi run create-sft

# Entraînement SFT (LoRA)
pixi run train-sft

# Entraînement DPO
pixi run train-dpo

# Lancement API (simple)
pixi run api

# Lancement API (vLLM)
pixi run api-vllm

# Lancement API (transformers)
pixi run api-dpo
```

## Livrables

1. **Dataset médical bilingue anonymisé** — `scripts/01_download_datasets.py`, `scripts/02_anonymize.py`, `scripts/03_create_sft_dpo.py`
2. **Modèle spécialisé Qwen3-1.7B** — `models/checkpoints/sft_final/`, `models/checkpoints/dpo_a2_optimized/`
3. **Endpoint API de démonstration** — `scripts/06_api_vllm.py`, `Dockerfile`
4. **Pipeline CI/CD** — `.forgejo/workflows/ci.yml`
5. **Rapport technique** — `docs/rapport_technique.md`

## Monitoring et Traçabilité

### Dashboard MLflow
```bash
# Lancer le dashboard MLflow (métriques d'entraînement, hyperparamètres)
cd /mnt/prod && mlflow ui --backend-store-uri file:///mnt/prod/mlruns --host 0.0.0.0 --port 5050

# Note : les logs d'entraînement détaillés (loss curves) nécessitent un ré-entraînement
# avec logging activé. Les valeurs finales sont documentées dans docs/rapport_technique.md
```

### Traçabilité des appels API
Chaque appel à l'API est logué au format JSONL dans `/app/logs/api_trace.log` :
- `request_id` (UUID), `timestamp`, `endpoint`, `latence_ms`
- `tokens_generated`, `priority_level`, `symptoms` (anonymisés)

### Docker (MLflow + API)
```bash
# API de triage
docker run --gpus all -p 8000:8000 posologic:latest serve

# Dashboard MLflow
docker run -p 5050:5050 posologic:latest mlflow
```

## Validation complète

```bash
pixi install && \
pixi run -e default python -m pytest tests/ -v && \
pixi run -e default python -m py_compile scripts/06_api.py && \
pixi run -e default python -m py_compile scripts/06_api_vllm.py && \
echo "[OK] Validation terminee"
```

## Licence

MIT © Damien Guesdon
