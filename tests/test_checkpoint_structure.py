"""Tests unitaires de base pour la validation CI du projet poso_logic."""
import os
import json


def test_sft_checkpoint_exists():
    assert os.path.isdir("models/checkpoints/sft_final"), "Dossier checkpoint SFT introuvable"


def test_adapter_model_exists():
    path = "models/checkpoints/sft_final/adapter_model.safetensors"
    assert os.path.isfile(path), f"Fichier adapter introuvable: {path}"
    size_mb = os.path.getsize(path) / (1024 * 1024)
    assert size_mb > 1, f"Fichier adapter trop petit: {size_mb:.1f} MB"


def test_adapter_config_valid():
    path = "models/checkpoints/sft_final/adapter_config.json"
    assert os.path.isfile(path), f"Fichier config introuvable: {path}"
    with open(path) as f:
        cfg = json.load(f)
    assert cfg.get("peft_type") == "LORA", f"Type PEFT inattendu: {cfg.get('peft_type')}"
    assert cfg.get("base_model_name_or_path") == "Qwen/Qwen3-1.7B"
    assert cfg.get("task_type") == "CAUSAL_LM"


def test_api_script_syntax():
    import py_compile
    py_compile.compile("scripts/06_api_dpo.py", doraise=True)


def test_api_script_imports():
    import ast
    with open("scripts/06_api_dpo.py") as f:
        tree = ast.parse(f.read())
    imports = [node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)]
    imports += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
    expected = {"vllm", "torch", "json", "os", "logging", "fastapi", "uvicorn", "pydantic"}
    found = set(imports) & expected
    assert len(found) >= 3, f"Imports de l'API insuffisants: {found}"
