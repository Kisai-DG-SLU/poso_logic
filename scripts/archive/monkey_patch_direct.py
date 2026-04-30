"""
Script de monkey-patching direct pour corriger l'incompatibilité entre
TRL 0.11.4 et transformers 4.51.0 pour DPO
"""
import sys
from transformers import Trainer

def apply_trl_patches():
    """
    Applique directement les patches nécessaires pour rendre
    TRL 0.11.4 compatible avec transformers 4.51.0
    """
    # Sauvegarde de la méthode originale
    original_inner_loop = Trainer._inner_training_loop
    
    # Définition de la méthode de remplacement
    def patched_inner_loop(self, *args, **kwargs):
        # Vérifie si c'est une instance de DPOTrainer en cherchant des attributs caractéristiques
        is_dpo_trainer = hasattr(self, "beta") and hasattr(self, "ref_model")
        
        if is_dpo_trainer:
            print("✅ Patch TRL appliqué pour DPOTrainer")
            
            # Modification de get_batch_samples pour qu'il fonctionne avec différents nombres d'arguments
            if hasattr(self, "get_batch_samples"):
                original_get_batch_samples = self.get_batch_samples
                
                def fixed_get_batch_samples(*batch_args, **batch_kwargs):
                    """Wrapper qui adapte les arguments selon les besoins."""
                    # Si le premier argument n'est pas self, ajouter self
                    if batch_args and batch_args[0] is not self:
                        batch_args = (self,) + batch_args
                    
                    # Si nous avons 4 arguments (self, batch, num_batches, device)
                    if len(batch_args) >= 4:
                        # Utiliser seulement self, batch et device
                        return original_get_batch_samples(batch_args[0], batch_args[1], batch_args[3])
                    
                    # Si nous avons 3 arguments ou moins, passer directement
                    return original_get_batch_samples(*batch_args, **batch_kwargs)
                
                # Remplacer la méthode originale
                self.get_batch_samples = fixed_get_batch_samples
        
        # Appeler la méthode originale avec les arguments passés
        return original_inner_loop(self, *args, **kwargs)
    
    # Application du monkey patch
    Trainer._inner_training_loop = patched_inner_loop
    print("Monkey patch appliqué avec succès à Trainer._inner_training_loop")

# Si ce script est exécuté directement, appliquer les patches
if __name__ == "__main__":
    apply_trl_patches()
    print("Patches appliqués. Prêt pour l'entraînement DPO.")
