#!/usr/bin/env python3
"""
MLflow Tracker — Dashboard de métriques pour le projet PosoLogic
Intègre le tracking MLflow aux entraînements SFT et DPO.
Stocke les métriques, hyperparamètres, artefacts et poids dans /mnt/prod/mlruns/
"""

import mlflow
import mlflow.pytorch
from pathlib import Path
import json
import time
import torch
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

# Configuration
MLFLOW_TRACKING_DIR = Path("/mnt/prod/mlruns")
MLFLOW_TRACKING_DIR.mkdir(exist_ok=True, parents=True)
mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")

EXPERIMENT_NAME = "poso_logic_triage_medical"


def setup_experiment() -> str:
    """Crée ou récupère l'expérience MLflow"""
    try:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=str(MLFLOW_TRACKING_DIR / EXPERIMENT_NAME)
        )
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id
    return experiment_id


def log_sft_hyperparams(config: Dict[str, Any]) -> None:
    """Log les hyperparamètres SFT dans MLflow"""
    params = {
        "phase": "SFT",
        "model_name": config.get("model_name", "Qwen/Qwen3-1.7B"),
        "lora_r": config.get("lora_r", 16),
        "lora_alpha": config.get("lora_alpha", 32),
        "lora_dropout": config.get("lora_dropout", 0.05),
        "learning_rate": config.get("learning_rate", 2e-4),
        "num_epochs": config.get("num_train_epochs", 3),
        "batch_size": config.get("per_device_batch_size", 1),
        "gradient_accumulation": config.get("gradient_accumulation_steps", 16),
        "max_seq_length": config.get("max_seq_length", 1024),
        "warmup_ratio": config.get("warmup_ratio", 0.1),
        "target_modules": ",".join(config.get("target_modules", [])),
    }
    mlflow.log_params(params)


def log_dpo_hyperparams(config: Dict[str, Any]) -> None:
    """Log les hyperparamètres DPO dans MLflow"""
    params = {
        "phase": "DPO",
        "model_name": config.get("model_name", "Qwen/Qwen3-1.7B"),
        "lora_r": config.get("lora_r", 16),
        "lora_alpha": config.get("lora_alpha", 32),
        "learning_rate": config.get("learning_rate", 1e-5),
        "num_epochs": config.get("num_train_epochs", 2),
        "batch_size": config.get("per_device_batch_size", 2),
        "gradient_accumulation": config.get("gradient_accumulation_steps", 4),
        "max_seq_length": config.get("max_seq_length", 2048),
        "beta": config.get("beta", 0.1),
        "loss_type": config.get("loss_type", "sigmoid"),
        "target_modules": ",".join(config.get("target_modules", [])),
    }
    mlflow.log_params(params)


def log_training_metrics(
    step: int,
    loss: float,
    learning_rate: float,
    epoch: float,
    phase: str = "SFT",
    extra_metrics: Optional[Dict[str, float]] = None
) -> None:
    """Log les métriques d'entraînement par step"""
    metrics = {
        "loss": loss,
        "learning_rate": learning_rate,
        "epoch": epoch,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    mlflow.log_metrics(metrics, step=step)


def log_evaluation_metrics(
    eval_results: Dict[str, Any],
    phase: str = "DPO"
) -> None:
    """Log les métriques d'évaluation clinique"""
    metrics = {
        f"{phase}_accuracy": eval_results.get("accuracy", 0),
        f"{phase}_correct_identifications": eval_results.get("correct_identifications", 0),
        f"{phase}_total_cases": eval_results.get("total_cases", 0),
        f"{phase}_avg_latency_ms": eval_results.get("average_latency", 0) * 1000,
    }
    mlflow.log_metrics(metrics)


def log_model_artifacts(
    checkpoint_path: Path,
    phase: str = "SFT"
) -> None:
    """Log les poids LoRA comme artefacts"""
    # Log adapter_config.json
    config_path = checkpoint_path / "adapter_config.json"
    if config_path.exists():
        mlflow.log_artifact(str(config_path), f"model_{phase}")

    # Log les poids LoRA (taille raisonnable, ~12 Mo)
    adapter_path = checkpoint_path / "adapter_model.safetensors"
    if adapter_path.exists():
        mlflow.log_artifact(str(adapter_path), f"model_{phase}")

    # Log tokenizer si présent
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        tok_path = checkpoint_path / tok_file
        if tok_path.exists():
            mlflow.log_artifact(str(tok_path), f"model_{phase}/tokenizer")


def log_benchmark_results(
    benchmark: Dict[str, Any]
) -> None:
    """Log les résultats de benchmark GPU"""
    metrics = {
        "bench_vram_used_mb": benchmark.get("vram_used_mb", 0),
        "bench_throughput_tok_s": benchmark.get("throughput_tok_s", 0),
        "bench_batch_size": benchmark.get("batch_size", 0),
    }
    mlflow.log_metrics(metrics)

    # Log le benchmark complet comme artefact JSON
    bench_path = Path("/tmp/benchmark_results.json")
    with open(bench_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    mlflow.log_artifact(str(bench_path), "benchmarks")


def log_loss_curve_from_logs(log_file: Path, run_name: str) -> None:
    """Extrait et log la courbe de loss depuis un fichier de log"""
    import re

    losses = []
    steps = []

    with open(log_file) as f:
        for line in f:
            # Pattern TRL: "{'loss': 0.42, 'learning_rate': 1e-5, 'epoch': 1.5}"
            match = re.search(r"'loss':\s*([\d.]+)", line)
            step_match = re.search(r"'step':\s*(\d+)", line)
            if match:
                loss = float(match.group(1))
                losses.append(loss)
                if step_match:
                    steps.append(int(step_match.group(1)))
                else:
                    steps.append(len(steps) + 1)

    if losses:
        for step, loss in zip(steps, losses):
            mlflow.log_metrics({"loss_from_log": loss}, step=step)

        # Créer et sauvegarder un graphique matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps, losses, 'b-', linewidth=0.8, alpha=0.7)
            ax.set_xlabel("Steps")
            ax.set_ylabel("Loss")
            ax.set_title(f"Courbe de Loss — {run_name}")
            ax.grid(True, alpha=0.3)

            # Moyenne glissante
            if len(losses) > 10:
                window = min(50, len(losses) // 5)
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                ax.plot(
                    steps[window-1:], smoothed,
                    'r-', linewidth=2, label=f"Moyenne glissante (n={window})"
                )
                ax.legend()

                chart_path = Path("/tmp/loss_curve.png")
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(str(chart_path), "charts")
        except ImportError:
            pass


def compare_runs(run_ids: list, metric_key: str = "loss") -> Dict:
    """Compare plusieurs runs sur une métrique donnée"""
    comparison = {}
    client = mlflow.tracking.MlflowClient()

    for run_id in run_ids:
        run = client.get_run(run_id)
        metrics_history = client.get_metric_history(run_id, metric_key)
        comparison[run_id] = {
            "run_name": run.data.tags.get("mlflow.runName", run_id),
            "phase": run.data.params.get("phase", "N/A"),
            "min_loss": min(m.value for m in metrics_history) if metrics_history else None,
            "final_loss": metrics_history[-1].value if metrics_history else None,
            "num_steps": len(metrics_history),
        }

    return comparison


def create_comparison_chart(run_ids: list, output_path: str = "/tmp/runs_comparison.png"):
    """Crée un graphique comparatif de plusieurs runs"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    client = mlflow.tracking.MlflowClient()
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
    for i, run_id in enumerate(run_ids):
        metrics = client.get_metric_history(run_id, "loss")
        if metrics:
            steps = [m.step for m in metrics]
            values = [m.value for m in metrics]
            run = client.get_run(run_id)
            phase = run.data.params.get("phase", "SFT")
            color = colors[i % len(colors)]
            ax.plot(steps, values, color=color, linewidth=1, alpha=0.7,
                   label=f"{phase} — {run.data.tags.get('mlflow.runName', run_id[:8])}")

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Comparaison SFT vs DPO — Courbes de Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path


# ============================================================
# Exemple d'utilisation dans un script d'entraînement
# ============================================================

def example_sft_training_with_mlflow():
    """Exemple d'intégration MLflow dans le script SFT"""
    experiment_id = setup_experiment()

    sft_config = {
        "model_name": "Qwen/Qwen3-1.7B",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_seq_length": 1024,
        "warmup_ratio": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"SFT_{datetime.now().strftime('%Y%m%d_%H%M')}"
    ) as run:
        log_sft_hyperparams(sft_config)

        # Pendant l'entraînement, appeler à chaque logging step :
        # log_training_metrics(step=step, loss=loss, learning_rate=lr, epoch=epoch, phase="SFT")

        # Après l'entraînement :
        # log_model_artifacts(Path("/mnt/prod/models/checkpoints/sft_final"), "SFT")

        print(f"Run SFT créé : {run.info.run_id}")
        print(f"UI : mlflow ui --backend-store-uri file://{MLFLOW_TRACKING_DIR}")


def example_dpo_training_with_mlflow():
    """Exemple d'intégration MLflow dans le script DPO"""
    experiment_id = setup_experiment()

    dpo_config = {
        "model_name": "Qwen/Qwen3-1.7B",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-5,
        "num_train_epochs": 2,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "beta": 0.1,
        "loss_type": "sigmoid",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"DPO_{datetime.now().strftime('%Y%m%d_%H%M')}"
    ) as run:
        log_dpo_hyperparams(dpo_config)

        # Pendant l'entraînement, appeler à chaque logging step :
        # log_training_metrics(step=step, loss=loss, learning_rate=lr, epoch=epoch, phase="DPO")

        # Après l'entraînement :
        # log_model_artifacts(Path("/mnt/prod/models/checkpoints/dpo_a2_optimized/final"), "DPO")

        print(f"Run DPO créé : {run.info.run_id}")
        print(f"UI : mlflow ui --backend-store-uri file://{MLFLOW_TRACKING_DIR}")


def reconstruct_runs_from_logs():
    """
    Reconstruit les runs MLflow à partir des logs d'entraînement existants.
    Utile pour rétroactivement peupler le dashboard.
    """
    experiment_id = setup_experiment()

    # Log SFT
    sft_logs = sorted(Path("/mnt/prod/logs").glob("sft*"))
    dpo_logs = sorted(Path("/mnt/prod/logs").glob("dpo*"))

    # DPO runs from logs
    for log_file in dpo_logs[-3:]:  # 3 derniers logs DPO
        run_name = f"DPO_{log_file.stem}"
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            # Charger config DPO
            with open("/mnt/prod/models/dpo_config.json") as f:
                dpo_cfg = json.load(f)
            log_dpo_hyperparams(dpo_cfg)
            log_loss_curve_from_logs(log_file, run_name)
        print(f"Run rétroactif créé : {run_name}")

    # SFT runs
    for log_file in sft_logs[-2:]:
        run_name = f"SFT_{log_file.stem}"
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            with open("/mnt/prod/models/sft_config.json") as f:
                sft_cfg = json.load(f)
            log_sft_hyperparams(sft_cfg)
            log_loss_curve_from_logs(log_file, run_name)
        print(f"Run rétroactif créé : {run_name}")

    print("\nTous les runs rétroactifs ont été créés.")
    print(f"Lancez : mlflow ui --backend-store-uri file://{MLFLOW_TRACKING_DIR}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "reconstruct":
        reconstruct_runs_from_logs()
    elif len(sys.argv) > 1 and sys.argv[1] == "sft":
        example_sft_training_with_mlflow()
    elif len(sys.argv) > 1 and sys.argv[1] == "dpo":
        example_dpo_training_with_mlflow()
    else:
        print("Usage:")
        print("  python mlflow_tracker.py sft          # Exemple run SFT")
        print("  python mlflow_tracker.py dpo          # Exemple run DPO")
        print("  python mlflow_tracker.py reconstruct  # Reconstruire depuis les logs")
        print(f"\nMLflow tracking URI: file://{MLFLOW_TRACKING_DIR}")
        print(f"UI command: mlflow ui --backend-store-uri file://{MLFLOW_TRACKING_DIR}")