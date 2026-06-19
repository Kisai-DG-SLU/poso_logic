# Script de publication des datasets anonymisés sur HuggingFace Hub
from datasets import load_from_disk, DatasetDict
from huggingface_hub import login, HfApi
import os

# Token HuggingFace (variable d'environnement) et nom du dépôt cible
HF_TOKEN = os.environ["HF_TOKEN"]
REPO = "damienguesdon/poso-logic2"

# Connexion à HuggingFace Hub
login(HF_TOKEN, add_to_git_credential=False)
api = HfApi()

# Création du dépôt s'il n'existe pas (exist_ok=True)
api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)

# Upload du dataset SFT (train/validation/test)
print("Upload SFT dataset...")
sft = load_from_disk("/mnt/prod/data/processed/sft_dataset")
sft.push_to_hub(REPO, config_name="sft", private=False)
print("  SFT OK")

# Upload du dataset DPO (train/validation)
print("Upload DPO dataset...")
dpo = load_from_disk("/mnt/prod/data/processed/dpo_dataset")
dpo.push_to_hub(REPO, config_name="dpo", private=False)
print("  DPO OK")

print("Fini !")
