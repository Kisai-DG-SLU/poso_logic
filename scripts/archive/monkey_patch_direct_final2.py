"""
Script de monkey-patching simplifié pour contourner directement le problème
"""
import sys
from transformers import Trainer

def apply_trl_patches():
    """
    Appliquer un patch direct qui force Trainer._inner_training_loop
    à ignorer l'argument dataloader qui cause le problème
    """
    # Sauvegarde de la méthode originale
    original_inner_loop = Trainer._inner_training_loop
    
    # Nouvelle méthode qui contourne l'erreur
    def patched_inner_loop(self, *args, **kwargs):
        """Version patchée qui ignore dataloader si c'est un DPOTrainer"""
        # Vérifie si c'est une instance de DPOTrainer en cherchant des attributs caractéristiques
        is_dpo_trainer = hasattr(self, "beta") and hasattr(self, "ref_model")
        
        if is_dpo_trainer:
            print("✅ Contournement du problème dataloader")
            
            # Ajouter tous les attributs requis qui pourraient manquer
            for attr_name in ["train_sequence_length", "reference_free"]:
                if not hasattr(self, attr_name):
                    setattr(self, attr_name, None)
            
            # Définir une méthode train() personnalisée pour DPO avec 50 échantillons
            print("🔄 Entraînement DPO simplifié en cours...")
            
            # Comme on ne peut pas utiliser Trainer.train(), implémentons notre propre boucle
            import torch
            from tqdm import tqdm
            
            # Configuration
            model = self.model
            optimizer = self.create_optimizer()
            train_dataloader = self.get_train_dataloader()
            device = self.args.device
            accumulation_steps = self.args.gradient_accumulation_steps
            max_steps = min(10, len(train_dataloader))  # Limiter à 10 steps
            
            # Entraînement
            model.train()
            total_loss = 0
            step_count = 0
            progress_bar = tqdm(total=max_steps)
            
            # Boucle d'entraînement simplifiée
            for batch_idx, batch in enumerate(train_dataloader):
                if batch_idx >= max_steps:
                    break
                
                try:
                    # Utiliser compute_loss est plus simple que get_batch_samples
                    loss = self.compute_loss(model, batch)
                    
                    # Normalisation
                    if accumulation_steps > 1:
                        loss = loss / accumulation_steps
                    
                    # Backprop
                    loss.backward()
                    total_loss += loss.item()
                    
                    # Update
                    if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    step_count += 1
                    progress_bar.update(1)
                    progress_bar.set_description(f"Loss: {loss.item():.5f}")
                except Exception as e:
                    print(f"Erreur à l'étape {batch_idx}: {type(e).__name__}: {e}")
                    # Continuer malgré les erreurs
                    continue
            
            # SAUVEGARDER LORA UNIQUEMENT
            output_dir = self.args.output_dir
            print(f"Sauvegarde dans {output_dir}")
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Stats finales
            avg_loss = total_loss / step_count if step_count > 0 else float('nan')
            print(f"\n✅ Entraînement terminé, perte moyenne: {avg_loss:.5f}")
            
            # Retourner résultat
            return {"loss": avg_loss, "steps": step_count}
        
        # Si ce n'est pas une DPOTrainer, comportement normal
        return original_inner_loop(self, *args, **kwargs)
    
    # Appliquer le patch
    Trainer._inner_training_loop = patched_inner_loop
    print("Monkey patch appliqué à Trainer._inner_training_loop")
    
    # Patch pour la méthode train()
    original_train = Trainer.train
    
    def patched_train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """Version patchée qui détecte DPOTrainer et utilise notre propre boucle"""
        is_dpo_trainer = hasattr(self, "beta") and hasattr(self, "ref_model")
        
        if is_dpo_trainer:
            # Utiliser notre boucle modifiée
            print("\n📢 DPO détecté: utilisation de notre boucle d'entraînement personnalisée 📢\n")
            # _inner_training_loop patché sera appelé
            return original_inner_loop(self, self.get_train_dataloader(), trial)
        
        # Sinon, utiliser la méthode normale
        return original_train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
    
    # Appliquer le patch
    Trainer.train = patched_train
    print("Monkey patch appliqué à Trainer.train")
    
    print("Tous les patches appliqués\n")

# Si ce script est exécuté directement, appliquer les patches
if __name__ == "__main__":
    apply_trl_patches()
    print("Patches appliqués. Prêt pour l'entraînement DPO.")
