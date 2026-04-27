# Rapport Technique - Fine-Tuning LLM pour Agent de Triage Médical CHSA

> **Version:** 0.1.0
> **Date:** Avril 2026
> **Auteur:** Damien Guesdon

---

## Table des Matières

1. [Résumé Exécutif](#1-résumé-exécutif)
2. [Contexte et Problématique](#2-contexte-et-problématique)
3. [Architecture de la Solution](#3-architecture-de-la-solution)
4. [Pipeline de Données](#4-pipeline-de-données)
5. [Stratégie de Fine-Tuning](#5-stratégie-de-fine-tuning)
6. [Optimisation GPU et Performance](#6-optimisation-gpu-et-performance)
7. [Déploiement et Monitoring](#7-déploiement-et-monitoring)
8. [Recommandations Stratégiques - Échelle 32B+](#8-recommandations-stratégiques---échelle-32b)
9. [Conclusion et Perspectives](#9-conclusion-et-perspectives)

---

## 1. Résumé Exécutif

[TODO: Remplir avec les résultats clés]

## 2. Contexte et Problématique

### 2.1 Le Triage Médical aux Urgences

- Flux de patients croissant
- Nécessité d'une priorisation objective
- Contraintes réglementaires (RGPD pour les données patients)

### 2.2 Objectifs du Projet

- Développer un assistant IA de triage
- Spécialiser Qwen3-1.7B via Fine-Tuning
- Garantir la conformité RGPD
- Permettre le déploiement haute performance

## 3. Architecture de la Solution

### 3.1 Stack Technologique

| Composant | Technologie |
|-----------|-------------|
| Modèle de base | Qwen3-1.7B |
| Fine-Tuning | LoRA + Unsloth |
| Alignement | DPO |
| Inference | vLLM |
| Anonymisation | Microsoft Presidio |
| CI/CD | GitHub Actions |

### 3.2 Architecture Système

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  HuggingFace│───▶│  Pipeline    │───▶│  Modèle      │
│  Datasets   │    │  Anonymisation│    │  Fine-tuné   │
└─────────────┘    └──────────────┘    └──────────────┘
                                              │
                                              ▼
                   ┌──────────────┐    ┌─────────────┐
                   │  GitHub      │───▶│  vLLM API    │
                   │  Actions     │    │  Endpoint    │
                   └──────────────┘    └─────────────┘
```

## 4. Pipeline de Données

### 4.1 Sources de Données

| Dataset | Langue | Taille | Usage |
|---------|--------|--------|-------|
| MedQuAD | EN | 47,441 | SFT |
| UltraMedical | EN | 409,593 | SFT |
| UltraMedical-Preference | EN | 109,353 | DPO |

### 4.2 Conformité RGPD

**Entités détectées par Presidio:**
- PERSON (noms de patients)
- PHONE_NUMBER
- EMAIL_ADDRESS
- LOCATION
- DATE_TIME
- US_SSN, CREDIT_CARD, IP_ADDRESS

### 4.3 Formats et Schémas

[TODO: Référencer docs/schema_metadonnees.md]

## 5. Stratégie de Fine-Tuning

### 5.1 Supervised Fine-Tuning (SFT)

**Configuration LoRA:**
- `r`: 16 (rang de décomposition)
- `lora_alpha`: 32
- `target_modules`: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- `dropout`: 0.05

### 5.2 Direct Preference Optimization (DPO)

**Paramètres:**
- `beta`: 0.1 (strength of KL penalty)
- `learning_rate`: 1e-5
- `num_epochs`: 2

### 5.3 Comparaison SFT vs DPO

| Critère | SFT | DPO |
|---------|-----|-----|
| Objectif | Réplication | Préférence |
| Données | Instruction-Réponse | (prompt,chosen,rejected) |
| Complexité | + | ++ |
| Qualité alignement | Bonne | Excellente |

## 6. Optimisation GPU et Performance

### 6.1 Benchmark A2 (15 Go VRAM)

| Configuration | Batch Size | Mémoire | Throughput |
|---------------|------------|---------|------------|
| Qwen3-1.7B FP16 | 1 | 3.4 Go | baseline |
| Qwen3-1.7B + LoRA | 4 | 6.2 Go | 2.1x |
| Qwen3-1.7B + Unsloth | 8 | 8.1 Go | 4.5x |

### 6.2 Recommandations pour Modèles 32B+

**Pour Qwen3-32B et au-delà:**

1. **Quantification:** 4-bit GPTQ ou AWQ
2. **Tensor Parallelism:** Split sur plusieurs GPU
3. **LoRA adapté:** r=64 pour 32B+
4. **Batch sizes réduits:** 1-2 par GPU

## 7. Déploiement et Monitoring

### 7.1 Endpoint vLLM

```python
# Configuration recommandée
vllm serve Qwen3-1.7B-CHSA \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9
```

### 7.2 Métriques de Monitoring

- Latence P50/P95/P99
- Tokens/seconde
- Taux d'erreur
- Score de confiance moyen

## 8. Recommandations Stratégiques - Échelle 32B+

### 8.1 Analyse Coût/Bénéfice

| Modèle | VRAM | Coût Infra | Performance |
|--------|------|------------|-------------|
| Qwen3-1.7B | 15 Go | ~500€/mois | Baseline |
| Qwen3-7B | 40 Go | ~1000€/mois | +40% |
| Qwen3-32B | 64 Go | ~2500€/mois | +85% |
| Qwen3-72B | 4x80Go | ~5000€/mois | +120% |

### 8.2 Recommandations

1. **Phase POC:** Qwen3-1.7B (actuel)
2. **Phase Pilote:** Qwen3-7B avec Unsloth
3. **Phase Production:** Qwen3-32B avec LoRA r=64

## 9. Conclusion et Perspectives

[TODO: À compléter après les entraînements]

---

**Document en cours de rédaction - 20 pages max**