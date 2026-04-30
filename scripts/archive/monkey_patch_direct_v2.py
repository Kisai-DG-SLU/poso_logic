"""
Script de monkey-patching direct amélioré pour corriger l'incompatibilité entre
TRL 0.11.4 et transformers 4.51.0 pour DPO
"""
import sys
from transformers import Trainer

def apply_trl_patches():
    """
    Applique directement les patches nécessaires pour rendre
    TRL 0.11.4 compatible avec transformers 4.51.0
    """
    # Sauvegarde de la méthode originale du Trainer
    original_inner_loop = Trainer._inner_training_loop
    
    # Définition de la méthode de remplacement
    def patched_inner_loop(self, *args, **kwargs):
        # Vérifie si c'est une instance de DPOTrainer en cherchant des attributs caractéristiques
        is_dpo_trainer = hasattr(self, "beta") and hasattr(self, "ref_model")
        
        if is_dpo_trainer:
            print("✅ Patch TRL appliqué pour DPOTrainer")
            
            # Définir manuellement les attributs manquants par précaution
            required_attrs = {
                "model_init_kwargs": None,
                "ref_model_init_kwargs": None,
                "generate_during_eval": False,
                "precompute_ref_log_probs": False,
                "remove_unused_columns": False,
                "include_tokens_per_second": False,
                "model_adapter_name": None,
                "ref_adapter_name": None,
                "reference_free": False,
                "truncation_mode": "keep_end",
                "optimization_method": "offline",
                "label_pad_token_id": -100,
                "disable_dropout": True,
                "dataset_num_proc": 1,
                "sync_ref_model": False,
                "f_divergence_type": "kl",
                "f_alpha_divergence_coef": 1.0,
                "include_num_input_tokens_seen": False,
                "lm_head_name": "lm_head",
                "force_use_ref_model": False,
            }
            
            # Ajouter les attributs manquants à self
            for attr, value in required_attrs.items():
                if not hasattr(self, attr):
                    setattr(self, attr, value)
            
            # Patch pour get_batch_samples (problème d'arguments)
            if hasattr(self, "get_batch_samples"):
                original_get_batch_samples = self.get_batch_samples
                
                def fixed_get_batch_samples(*batch_args, **batch_kwargs):
                    """Wrapper qui adapte les arguments selon les besoins."""
                    # Si le premier argument n'est pas self, ajouter self
                    if batch_args and batch_args[0] is not self:
                        batch_args = (self,) + batch_args
                    
                    # Cas dataloader manquant
                    if len(batch_args) == 2 and 'dataloader' in kwargs:
                        # Ignorer le dataloader
                        return original_get_batch_samples(*batch_args)
                    
                    # Si nous avons 4 arguments (self, batch, num_batches, device)
                    elif len(batch_args) >= 4:
                        # Utiliser seulement self, batch et device
                        return original_get_batch_samples(batch_args[0], batch_args[1], batch_args[3])
                    
                    # Si nous avons 3 arguments ou moins, passer directement
                    return original_get_batch_samples(*batch_args, **batch_kwargs)
                
                # Remplacer la méthode originale
                self.get_batch_samples = fixed_get_batch_samples
            
            # Patch pour compute_loss si nécessaire
            if hasattr(self, "compute_loss"):
                original_compute_loss = self.compute_loss
                
                def fixed_compute_loss(*loss_args, **loss_kwargs):
                    """Wrapper pour adapter compute_loss si besoin."""
                    try:
                        return original_compute_loss(*loss_args, **loss_kwargs)
                    except TypeError as e:
                        if "missing 1 required positional argument: 'dataloader'" in str(e):
                            # Si l'erreur est liée à dataloader manquant, l'ignorer
                            return original_compute_loss(loss_args[0], loss_args[1])
                        else:
                            raise
                
                # Remplacer la méthode
                self.compute_loss = fixed_compute_loss
                
            # Patch pour _get_batch_samples si existant
            if hasattr(self, "_get_batch_samples"):
                original_get_batch = self._get_batch_samples
                
                def fixed_get_batch_samples(*batch_args, **batch_kwargs):
                    """Wrapper qui adapte _get_batch_samples."""
                    # Adaptations similaires à get_batch_samples
                    if batch_args and batch_args[0] is not self:
                        batch_args = (self,) + batch_args
                    
                    # Si dataloader est dans les kwargs mais pas utilisé
                    if 'dataloader' in batch_kwargs and len(batch_args) <= 3:
                        # Supprimer dataloader des kwargs
                        batch_kwargs.pop('dataloader')
                    
                    return original_get_batch(*batch_args, **batch_kwargs)
                
                # Remplacer la méthode
                self._get_batch_samples = fixed_get_batch_samples
                
            # Patch pour les fonctions auxiliaires
            for attr_name in dir(self):
                if attr_name.startswith("_") and callable(getattr(self, attr_name)):
                    try:
                        orig_func = getattr(self, attr_name)
                        
                        def make_wrapper(func):
                            def wrapper(*func_args, **func_kwargs):
                                try:
                                    return func(*func_args, **func_kwargs)
                                except TypeError as e:
                                    if "missing 1 required positional argument: 'dataloader'" in str(e):
                                        # Pour les erreurs de dataloader manquant
                                        if len(func_args) >= 2:
                                            return func(func_args[0], func_args[1])
                                    raise
                            return wrapper
                        
                        # Ne pas remplacer les méthodes déjà patchées
                        if attr_name not in ["get_batch_samples", "compute_loss", "_get_batch_samples"]:
                            setattr(self, attr_name, make_wrapper(orig_func))
                    except (AttributeError, TypeError):
                        pass
        
        # Appeler la méthode originale avec les arguments passés
        try:
            return original_inner_loop(self, *args, **kwargs)
        except TypeError as e:
            if "missing 1 required positional argument: 'dataloader'" in str(e) and is_dpo_trainer:
                # Si l'erreur est liée à dataloader manquant
                if len(args) > 0 and 'dataloader' in kwargs:
                    # Ignorer dataloader et appeler avec args uniquement
                    return original_inner_loop(self, args[0])
                elif len(args) > 1:
                    # Utiliser seulement les deux premiers arguments
                    return original_inner_loop(self, args[0])
            raise
    
    # Application du monkey patch
    Trainer._inner_training_loop = patched_inner_loop
    print("Monkey patch appliqué avec succès à Trainer._inner_training_loop")

# Si ce script est exécuté directement, appliquer les patches
if __name__ == "__main__":
    apply_trl_patches()
    print("Patches appliqués. Prêt pour l'entraînement DPO.")
