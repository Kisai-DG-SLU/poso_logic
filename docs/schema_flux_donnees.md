# Schéma du Flux de Données — PosoLogic

> Diagramme du pipeline complet de traitement des données
> Du dataset source à l'API de triage

---

## Diagramme Global

```mermaid
flowchart TD
    subgraph Sources["📦 Sources de Données"]
        A1["MedQuAD<br/>47K paires Q/R<br/>EN"]
        A2["UltraMedical SFT<br/>409K instructions<br/>EN"]
        A3["UltraMedical Pref<br/>109K paires<br/>EN"]
        A4["FrenchMedMCQA<br/>2.5K QCM<br/>FR"]
    end

    subgraph Download["⬇️ 01_download_datasets.py"]
        B1["Téléchargement HF Hub"]
        B2["Calcul SHA256"]
        B3["Stockage data/raw/"]
        B4["Log: version, date, checksum"]
    end

    subgraph Anonym["🔒 02_anonymize.py (Presidio)"]
        C1["Détection 14 entités PII"]
        C2["Pseudonymisation<br/>[PERSON_1], [DATE_1]..."]
        C3["Stockage data/clean/"]
        C4["Journal: entités, taux"]
    end

    subgraph Create["🏗️ 03_create_sft_dpo.py"]
        D1["Format SFT:<br/>{instruction, response}"]
        D2["Format DPO:<br/>{prompt, chosen, rejected}"]
        D3["Split Train/Val/Test<br/>(80/10/10)"]
        D4["Stockage data/processed/sft/<br/>data/processed/dpo/"]
    end

    subgraph Train["🎯 Entraînement"]
        E1["04_train_sft.py<br/>LoRA r=16, lr=2e-4<br/>Unsloth optimisé"]
        E2["05_train_dpo.py<br/>Beta=0.1, lr=1e-5<br/>DPO + LoRA"]
        E3["→ models/sft_final/<br/>(12 Mo, adapter_model)"]
        E4["→ models/dpo_a2_optimized/<br/>(12 Mo, adapter_model)"]
        E5["MLflow Tracking<br/>Loss, LR, Epochs"]
    end

    subgraph Eval["📊 Évaluation"]
        F1["evaluate_dpo_model.py<br/>5 cas cliniques test"]
        F2["mlflow_tracker.py<br/>Loss curves, comparaison"]
        F3["Métriques:<br/>Accuracy, latence,<br/>sécurité clinique"]
    end

    subgraph Deploy["🚀 Déploiement"]
        G1["Dockerfile<br/>CUDA 12.4 + vLLM"]
        G2["Forgejo Actions CI<br/>6 jobs chaînés"]
        G3["API vLLM Endpoint<br/>POST /triage<br/>POST /generate"]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    B1 --> B2 --> B3 --> B4
    B3 --> C1
    C1 --> C2 --> C3 --> C4
    C3 --> D1
    C3 --> D2
    D1 --> D3
    D2 --> D3
    D3 --> D4
    D4 --> E1
    D4 --> E2
    E1 --> E3
    E2 --> E4
    E1 -.-> E5
    E2 -.-> E5
    E3 --> F1
    E4 --> F1
    F1 --> F2 --> F3
    E4 --> G1
    G1 --> G2 --> G3

    style Sources fill:#1a1a2e,stroke:#e94560,color:#fff
    style Download fill:#16213e,stroke:#0f3460,color:#fff
    style Anonym fill:#0f3460,stroke:#533483,color:#fff
    style Create fill:#533483,stroke:#e94560,color:#fff
    style Train fill:#1a1a2e,stroke:#16c79a,color:#fff
    style Eval fill:#16213e,stroke:#f0a500,color:#fff
    style Deploy fill:#0f3460,stroke:#00b4d8,color:#fff
```

---

## Flux Détaillé — Traçabilité

```
[Source]              [Script]              [Sortie]              [Métadonnées]
─────────────────────────────────────────────────────────────────────────────────
HuggingFace Hub   →    01_download    →    data/raw/          [version,

date, SHA256]
                                                              ↓
data/raw/         →    02_anonymize   →    data/clean/        [entités Presidio, %anonymisé]
                                                              ↓
data/clean/       →    03_create_sft  →    data/processed/    [splits, distribution priorités]
                                                              ↓
data/processed/   →    04_train_sft   →    models/sft_final/  [config LoRA, loss, epochs]
data/processed/   →    05_train_dpo   →    models/dpo_final/  [config DPO, beta, loss]
                                                              ↓
models/dpo_final/ →    Dockerfile     →    Image Docker        [CUDA 12.4, vLLM, Presidio]
                                                              ↓
Image Docker      →    CI (Forgejo)   →    API Endpoint        [healthcheck, /triage]
```

---

## Structure des Données à Chaque Étape

### Étape 1 — Raw
```
data/raw/
├── medquad/
│   └── data.arrow          (47K paires Q/R)
├── ultramedical/
│   ├── sft/
│   │   └── data.arrow      (409K instructions)
│   └── preference/
│       └── data.arrow      (109K paires)
├── frenchmedmcqa/
│   └── data.arrow          (2.5K QCM)
└── datasets_checksums.json
```

### Étape 2 — Clean (anonymisé)
```
data/clean/
├── medquad_anon.arrow
├── ultramedical_sft_anon.arrow
├── ultramedical_pref_anon.arrow
├── frenchmedmcqa_anon.arrow
└── anonymization_report.json
```

### Étape 3 — Processed (formaté)
```
data/processed/
├── sft/
│   ├── train/    (365 800 exemples)
│   ├── val/      (45 700 exemples)
│   └── test/     (45 700 exemples)
├── dpo/
│   ├── train/    (87 400 exemples)
│   ├── val/      (10 900 exemples)
│   └── test/     (10 900 exemples)
└── splits_report.json
```

### Étape 4 — Modèles
```
models/
├── sft_config.json
├── dpo_config.json
├── checkpoints/
│   ├── sft_final/
│   │   ├── adapter_model.safetensors   (12 Mo)
│   │   ├── adapter_config.json
│   │   └── tokenizer/
│   └── dpo_a2_optimized/
│       ├── final/
│       │   ├── adapter_model.safetensors   (12 Mo)
│       │   ├── adapter_config.json
│       │   └── tokenizer/
│       └── checkpoint-*/               (15 checkpoints)
└── mlruns/                             (MLflow tracking)
```

---

## Métadonnées de Traçabilité

Chaque script enregistre ses métadonnées pour garantir la reproductibilité :

| Script | Fichier de log | Contenu |
|--------|---------------|---------|
| `01_download` | `datasets_checksums.json` | version, date, SHA256 par dataset |
| `02_anonymize` | `anonymization_report.json` | entités, compteurs, taux |
| `03_create_sft_dpo` | `splits_report.json` | distribution, tailles, niveaux de priorité |
| `04_train_sft` | `sft_training_log.json` | hyperparamètres, loss par step |
| `05_train_dpo` | `dpo_training_log.json` | hyperparamètres, loss par step |
| `mlflow_tracker` | `mlruns/` | tout : params, metrics, artifacts |

---

**Schéma conforme à l'exigence mentorat : traçabilité complète de la source au déploiement.**
 date, SHA256]
                                                              ↓
data/raw/         →    02_anonymize   →    data/clean/        [entités Presidio, %anonymisé]
                                                              ↓
data/clean/       →    03_create_sft  →    data/processed/    [splits, distribution priorités]
                                                              ↓
data/processed/   →    04_train_sft   →    models/sft_final/  [config LoRA, loss, epochs]
data/processed/   →    05_train_dpo   →    models/dpo_final/  [config DPO, beta, loss]
                                                              ↓
models/dpo_final/ →    Dockerfile     →    Image Docker        [CUDA 12.4, vLLM, Presidio]
                                                              ↓
Image Docker      →    CI (Forgejo)   →    API Endpoint        [healthcheck, /triage]
```

---

## Structure des Données à Chaque Étape

### Étape 1 — Raw
```
data/raw/
├── medquad/
│   └── data.arrow          (47K paires Q/R)
├── ultramedical/
│   ├── sft/
│   │   └── data.arrow      (409K instructions)
│   └── preference/
│       └── data.arrow      (109K paires)
├── frenchmedmcqa/
│   └── data.arrow          (2.5K QCM)
└── datasets_checksums.json
```

### Étape 2 — Clean (anonymisé)
```
data/clean/
├── medquad_anon.arrow
├── ultramedical_sft_anon.arrow
├── ultramedical_pref_anon.arrow
├── frenchmedmcqa_anon.arrow
└── anonymization_report.json
```

### Étape 3 — Processed (formaté)
```
data/processed/
├── sft/
│   ├── train/    (365 800 exemples)
│   ├── val/      (45 700 exemples)
│   └── test/     (45 700 exemples)
├── dpo/
│   ├── train/    (87 400 exemples)
│   ├── val/      (10 900 exemples)
│   └── test/     (10 900 exemples)
└── splits_report.json
```

### Étape 4 — Modèles
```
models/
├── sft_config.json
├── dpo_config.json
├── checkpoints/
│   ├── sft_final/
│   │   ├── adapter_model.safetensors   (12 Mo)
│   │   ├── adapter_config.json
│   │   └── tokenizer/
│   └── dpo_a2_optimized/
│       ├── final/
│       │   ├── adapter_model.safetensors   (12 Mo)
│       │   ├── adapter_config.json
│       │   └── tokenizer/
│       └── checkpoint-*/               (15 checkpoints)
└── mlruns/                             (MLflow tracking)
```

---

## Métadonnées de Traçabilité

Chaque script enregistre ses métadonnées pour garantir la reproductibilité :

| Script | Fichier de log | Contenu |
|--------|---------------|---------|
| `01_download` | `datasets_checksums.json` | version, date, SHA256 par dataset |
| `02_anonymize` | `anonymization_report.json` | entités, compteurs, taux |
| `03_create_sft_dpo` | `splits_report.json` | distribution, tailles, niveaux de priorité |
| `04_train_sft` | `sft_training_log.json` | hyperparamètres, loss par step |
| `05_train_dpo` | `dpo_training_log.json` | hyperparamètres, loss par step |
| `mlflow_tracker` | `mlruns/` | tout : params, metrics, artifacts |

---

**Schéma conforme à l'exigence mentorat : traçabilité complète de la source au déploiement.**
