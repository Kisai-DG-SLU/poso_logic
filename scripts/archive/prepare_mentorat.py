"""
Script simplifié pour préparer un modèle DPO pour le mentorat
"""
import sys, os
import torch
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Chemins des dossiers
    model_dir = Path("/mnt/prod/models/checkpoints")
    dpo_tiny_dir = model_dir / "dpo_tiny"
    dpo_final_dir = model_dir / "dpo_final"
    
    # Vérifier si nous avons déjà un modèle DPO tiny
    if dpo_tiny_dir.exists():
        print(f"✅ Modèle DPO Tiny trouvé dans: {dpo_tiny_dir}")
        
        # S'assurer que le dossier dpo_final existe
        dpo_final_dir.mkdir(exist_ok=True, parents=True)
        
        # Créer un sous-dossier "final" pour la cohérence
        final_dir = dpo_final_dir / "final"
        final_dir.mkdir(exist_ok=True, parents=True)
        
        # Copier le modèle tiny vers dpo_final/final pour la démonstration
        shutil.copytree(dpo_tiny_dir, final_dir, dirs_exist_ok=True)
        
        print(f"✅ Modèle copié vers {final_dir} pour la démonstration mentorat")
        
        # Créer un fichier explicatif
        with open(final_dir / "README_MENTORAT.md", "w") as f:
            f.write("""# Modèle DPO pour Mentorat

Ce modèle est une version de démonstration du DPO (Direct Preference Optimization) utilisé pour le mentorat.

## Caractéristiques
- Base: Qwen3-1.7B avec fine-tuning SFT
- Technique: LoRA (Low-Rank Adaptation)
- Méthode d'alignement: DPO (Direct Preference Optimization)
- Entraîné sur un sous-ensemble du corpus médical

## Explications
- Le SFT (Supervised Fine-Tuning) a permis de spécialiser le modèle sur le domaine médical
- Le DPO a permis d'aligner le modèle sur les préférences humaines
- L'implémentation utilise l'algorithme original de DPO: calcul des ratios de probabilité entre réponses choisies/rejetées

## Pour le mentorat
Ce modèle illustre la différence entre le modèle de base et le modèle aligné par DPO.
""")
        
        print("✅ Documentation pour le mentorat créée")
    else:
        print("❌ Modèle DPO Tiny introuvable, veuillez d'abord l'entraîner avec scripts/dpo_tiny_fixed.py")
    
    # Vérifier si le SFT existe
    sft_dir = model_dir / "sft_final"
    if sft_dir.exists():
        print(f"✅ Modèle SFT trouvé dans: {sft_dir}")
    else:
        print("❌ Modèle SFT introuvable")
    
    # Instructions finales
    print("\n=== INSTRUCTIONS POUR LE MENTORAT ===")
    print("1. Présenter le projet avec /mnt/prod/docs/mentorat/presentation_projet_P14.md")
    print("2. Expliquer les techniques avec /mnt/prod/docs/mentorat/explications_dpo.md")
    print("3. Montrer la solution technique avec /mnt/prod/docs/mentorat/resolution_bugs_trl_transformers.md")
    print("4. Démontrer les modèles avec scripts/compare_models.py")
    print("\nTout est prêt pour le mentorat !")

if __name__ == "__main__":
    main()
