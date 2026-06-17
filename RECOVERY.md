# Récupération après reboot du pod

```bash
# 1. Cloner le dépôt
git clone ssh://git@forgejo-external.sophia-sandbox.svc.cluster.local:2222/projets-formation/poso_logic.git
cd poso_logic

# 2. Installer l'environnement pixi
pixi install

# 3. Télécharger le dataset de test UltraMedical-Preference
pixi run python scripts/01_download_datasets.py
#   → data/raw/ultramedical_preference/

# 4. (Optionnel) Ré-entraîner le DPO
pixi run python scripts/05_train_dpo_final_prod_reduced.py
#   → models/checkpoints/dpo_a2_optimized/final/

# 5. Fusionner les poids LoRA dans le modèle de base
pixi run python scripts/merge_lora_to_vllm.py
#   → models/merged_dpo_vllm/  (3.4 Go)

# 6. Lancer l'API vLLM
pixi run python scripts/06_api_vllm.py
#   → http://localhost:8000

# 7. (Optionnel) Ré-évaluer
pixi run python scripts/evaluate_ultramedical_preference.py
#   → eval_reports/eval_ultramedical_preference.json
```

Le merge (étape 5) prend ~2 minutes et ne nécessite que le modèle de base + l'adaptateur LoRA.
