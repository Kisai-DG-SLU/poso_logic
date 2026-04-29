"""
Script de secours pour résoudre le problème DPO en utilisant des versions spécifiques de TRL et transformers
connues pour être compatibles entre elles
"""
import sys
import subprocess
import argparse

def install_compatible_versions():
    """
    Installe des versions de TRL et transformers connues pour être compatibles
    pour DPO
    """
    print("Installation des versions compatibles TRL/transformers...")
    
    # Versions connues pour être compatibles
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "trl==0.7.1", 
        "transformers==4.34.0",
        "--no-deps"  # Important pour ne pas casser d'autres dépendances
    ])
    
    print("Installation terminée!")

def run_dpo_training():
    """
    Lance le script d'entraînement DPO avec les versions installées
    """
    print("\nLancement de l'entraînement DPO...")
    
    # Exécute le script d'origine
    subprocess.check_call([
        sys.executable, "/mnt/prod/scripts/05_train_dpo.py"
    ])

def restore_original_versions():
    """
    Restaure les versions originales de TRL et transformers depuis pixi.toml
    """
    print("\nRestauration des versions originales...")
    
    # Réinstalle les packages depuis pixi
    subprocess.check_call([
        "pixi", "install"
    ])
    
    print("Versions originales restaurées!")

def main():
    parser = argparse.ArgumentParser(description="Solution de secours pour l'entraînement DPO")
    parser.add_argument("--install-only", action="store_true", 
                        help="Installer uniquement les versions compatibles sans lancer l'entraînement")
    parser.add_argument("--train-only", action="store_true", 
                        help="Lancer uniquement l'entraînement sans modifier les versions")
    parser.add_argument("--restore-only", action="store_true",
                        help="Restaurer uniquement les versions originales")
    
    args = parser.parse_args()
    
    if args.install_only:
        install_compatible_versions()
    elif args.train_only:
        run_dpo_training()
    elif args.restore_only:
        restore_original_versions()
    else:
        # Flux complet par défaut
        install_compatible_versions()
        run_dpo_training()
        restore_original_versions()

if __name__ == "__main__":
    main()
