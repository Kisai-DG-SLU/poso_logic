# Mentorat Projet P14 - Découverte de l'entraînement DPO

## Introduction - Mon parcours sur ce projet

Pour ce projet P14 (Agent de Triage Médical CHSA), je découvre l'entraînement DPO (Direct Preference Optimization). C'est ma première expérience avec cette technique d'alignement de modèles de langage.

### L'environnement que je me suis imposé
Avant de commencer le projet, j'ai choisi de travailler sur un **Pod OKD avec GPU A2 (16GB de VRAM)**. Ce n'est pas une exigence du projet, c'est un choix personnel pour me challenger. Je voulais comprendre les contraintes de mémoire et d'optimisation.

## Ce que j'ai compris qu'il fallait faire dans ce projet

### 1. **Compréhension du flux DPO**
Le projet demande de spécialiser Qwen3-1.7B pour le triage médical. J'ai compris que DPO est une méthode d'alignement qui :
- Compare deux réponses (chosen vs rejected)
- Apprend au modèle à préférer la "bonne" réponse
- Ne nécessite pas de modèle de récompense séparé (contrairement à RLHF)

### 2. **Les étapes clés (ce que j'ai appris)**

#### Étape 1 : Préparation des données (déjà acquise)
- Création d'un dataset avec `instruction`, `chosen`, `rejected`
- Formatage pour Qwen3 (respect de son tokenizer spécifique)
- Anonymisation RGPD avec Presidio

#### Étape 2 : Configuration de l'entraînement DPO
Voici ce que j'ai compris qu'il fallait configurer :

**Le Tokenizer :**
- Utiliser le tokenizer du modèle de base, pas celui du SFT
- J'ai appris que les tokenizers ont des spécificités (Qwen3 utilise des tokens spéciaux comme `<|im_start|>`)

**La configuration LoRA :**
- `target_modules` : Modules cibles dans le modèle
- Pour Qwen3, j'ai identifié : `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- `r` (rang) : Plus c'est élevé, plus le modèle apprend, mais plus c'est lourd
- `lora_alpha` : Généralement 2x `r`
- `lora_dropout` : Pour régulariser

**Gestion de la mémoire GPU :**
C'est là que ça devient technique ! Avec mon GPU A2 (16GB) :
- `max_seq_length` : Longueur maximale des séquences (j'ai dû réduire à 128 tokens)
- `batch_size` : 1 (pas le choix avec 16GB)
- `gradient_accumulation_steps` : Compenser le petit batch (j'ai dû monter à 64)
- `mixed_precision` : Utiliser FP16 pour économiser la mémoire
- `gradient_checkpointing` : Sacrifier un peu de vitesse pour économiser la mémoire

### 3. **Les problèmes que j'ai rencontrés (et que j'apprends à résoudre)**

#### Problème 1 : Out of Memory (OOM)
- **Ce que j'ai compris** : Le GPU A2 16GB est vite saturé
- **Solution que j'applique** :
  - Réduire `lora_r` à 2 (très petit)
  - Ne cibler que 2 modules : `q_proj` et `v_proj`
  - Utiliser `load_in_8bit=True` pour quantifier le modèle de référence
  - Désactiver le multiprocessing dans les DataLoaders

#### Problème 2 : Temps d'entraînement
- **Ce que j'ai calculé** : 
  - 24604 steps pour l'ensemble du dataset (196k exemples)
  - ~7 secondes par step avec ma config
  - **Total : ~48 heures !**
- **Ce que je réalise** : Un GPU A2 n'est pas idéal pour ce projet. Le projet spécifie "Efficient GPU usage" et je comprends pourquoi maintenant !

### 4. **Mon plan d'action actuel**

Je suis en train de régler ces problèmes. Voici mon approche :

1. **Tests en cours** : Je lance des batchs de tests (générés par l'IA car je n'maîtrise pas encore tout ça) pour :
   - Vérifier que le tokenizer fonctionne
   - Tester différentes configurations LoRA
   - Mesurer la consommation mémoire

2. **Script en cours de développement** :
```python
# Ce que j'apprends à configurer
config = {
    "model_name": "Qwen/Qwen3-1.7B",
    "max_seq_length": 128,      # Drastiquement réduit
    "batch_size": 1,
    "gradient_accumulation_steps": 8,  # Réduit de 64 à 8
    "lora_r": 2,               # Très réduit
    "target_modules": ["q_proj", "v_proj"],  # Seulement 2 modules
    "fp16": True,              # Mixed precision
    "gradient_checkpointing": True,
}
```

3. **Ce que j'observe** :
   - Loss qui descend doucement (6.93 → 3.59 actuellement)
   - Accuracy qui devient stable (100% sur mes tests)
   - GPU utilisé à ~60% (6.4GB/8.8GB)

### 5. **Ce que je n'ai pas encore compris (et que je vais apprendre)**

- **DPO vs SFT** : La différence théorique profonde
- **Pourquoi 48h d'entraînement** : Est-ce normal ? Est-ce que j'ai mal configuré ?
- **Métriques d'évaluation clinique** : Comment valider que mon modèle trie bien les urgences ?
- **Passage à l'échelle** : Le projet mentionne des modèles 32B+, je ne vois pas encore comment faire

### 6. **Ma stratégie pour la suite**

1. **Court terme** (ce weekend) :
   - Laisser l'entraînement tourner (48h c'est long !)
   - Commencer à analyser les résultats sur les premiers checkpoints
   - Documenter ce que je comprends

2. **Moyen terme** :
   - Comprendre pourquoi mon modèle met autant de temps
   - Étudier les techniques d'optimisation (DeepSpeed, FSDP)
   - Tester sur un GPU plus puissant (A100 si possible)

3. **Pour le rapport technique** :
   - Expliquer mes choix de config (pourquoi j'ai opté pour ce setup)
   - Analyser les limites de mon GPU A2
   - Proposer une architecture pour scaler (comme demandé dans le projet)

## Ce que je voudrais apprendre du mentorat

1. **Sur la technique** :
   - Est-ce que ma config LoRA est correcte pour apprendre ?
   - Comment je peux accélérer l'entraînement sans GPU plus puissant ?
   - Quelles métriques cliniques sont les plus pertinentes ?

2. **Sur le projet** :
   - Comment je devrais structurer mon rapport de 20 pages ?
   - Qu'est-ce que je dois absolument montrer lors de la démonstration ?
   - Le GPU A2 est-il vraiment trop juste pour ce projet ?

3. **Sur l'alignement** :
   - DPO est-il la meilleure méthode pour le triage médical ?
   - Comment je peux quantifier l'amélioration par rapport au modèle de base ?

## Conclusion - Mon état d'esprit

Je suis en pleine phase d'apprentissage. Je ne maîtrise pas encore les notions techniques (LoRA, DPO, gradient checkpointing...), mais je commence à comprendre le "pourquoi" des choix techniques. 

Mon setup GPU A2 16GB n'est probablement pas optimal, mais il me force à comprendre la gestion mémoire et l'optimisation. Je suis en train de faire tourner l'entraînement (48h !) tout en documentant ce que j'apprends.

**Ma question au mentor** : Est-ce que ma compréhension de DPO est correcte ? Et surtout, comment je peux mieux appréhender ces concepts techniques qui me semblent encore abstraits ?

---
*Document rédigé pendant l'entraînement DPO (step ~6000/24604, Loss: 3.59, ETA: 34h restantes)*