"""
Mini SFT de validation : 1000 échantillons, 1 epoch, max_seq_length=512
Utilisé pour tester le pipeline SFT avant le run complet

Points clés :
- Crée un mini dataset (1000 train / 100 val / 100 test) depuis le dataset complet
- Modifie temporairement sft_config.json : 1 epoch, 512 tokens max
- Modifie 04_train_sft.py pour pointer sur le mini dataset, puis restaure
- Permet de valider le pipeline en ~5 min avant un run complet
"""
import os, sys, json, shutil, subprocess
from pathlib import Path
from datasets import load_from_disk

N_TRAIN = 1000
N_VAL = 100
N_TEST = 100

SCRIPTS = Path("/mnt/prod/scripts")
PROCESSED = Path("/mnt/prod/data/processed")
MODELS = Path("/mnt/prod/models")
MINI_DS = PROCESSED / "sft_dataset_mini"
CONFIG_PATH = MODELS / "sft_config.json"
TRAIN_SCRIPT = SCRIPTS / "04_train_sft.py"

if MINI_DS.exists():
    shutil.rmtree(str(MINI_DS))

print("Préparation du dataset réduit...")
ds = load_from_disk(str(PROCESSED / "sft_dataset"))
ds["train"] = ds["train"].select(range(min(N_TRAIN, len(ds["train"]))))
ds["validation"] = ds["validation"].select(range(min(N_VAL, len(ds["validation"]))))
ds["test"] = ds["test"].select(range(min(N_TEST, len(ds["test"]))))
ds.save_to_disk(str(MINI_DS))
print(f"  Dataset réduit: train={len(ds['train'])}, val={len(ds['validation'])}, test={len(ds['test'])}")

print("Modification de la config SFT...")
config = json.loads(CONFIG_PATH.read_text())
config["num_train_epochs"] = 1
config["max_seq_length"] = 512
config["logging_steps"] = 10
config["save_steps"] = 50
CONFIG_PATH.write_text(json.dumps(config, indent=2))
print("  Config mise à jour: 1 epoch, max_seq_length=512")

print("Modification temporaire du script pour pointer sur le mini dataset...")
code = TRAIN_SCRIPT.read_text()
code = code.replace('"sft_dataset"', '"sft_dataset_mini"')
code = code.replace('"sft_dataset_mini"', '"sft_dataset_mini"')
TRAIN_SCRIPT.write_text(code)

try:
    print("Lancement du mini SFT...")
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--mode", "sft"],
        cwd="/mnt/prod",
        capture_output=False
    )
finally:
    print("Restauration du script original...")
    code = TRAIN_SCRIPT.read_text()
    code = code.replace('"sft_dataset_mini"', '"sft_dataset"')
    TRAIN_SCRIPT.write_text(code)

print("Mini-SFT terminé !")
