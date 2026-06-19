#!/usr/bin/env python3
"""Génère les courbes de loss et métriques à partir des checkpoints DPO.

Points clés :
- Parcourt tous les checkpoints DPO triés par step
- Lit training_stats.json dans chaque checkpoint
- Génère 3 graphiques : loss curve, reward accuracy, chosen/rejected ratios
- Sauvegarde en PNG (150dpi) dans /mnt/prod/docs/figures/
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CHECKPOINT_DIR = Path("/mnt/prod/models/checkpoints/dpo_a2_optimized")
OUTPUT_DIR = Path("/mnt/prod/docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_stats():
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    stats = []
    for cp in checkpoints:
        stats_file = cp / "training_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                data = json.load(f)
            data["checkpoint"] = int(cp.name.split("-")[1])
            stats.append(data)
    return stats

def plot_loss_curve(stats):
    steps = [s["checkpoint"] for s in stats]
    losses = [s["last_metrics"]["loss"] for s in stats]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, losses, 'b-', linewidth=1, alpha=0.7, label="Loss")

    if len(losses) > 5:
        window = max(3, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], smoothed, 'r-', linewidth=2, label=f"Moyenne glissante (n={window})")

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Courbe de Loss — Entraînement DPO (Qwen3-1.7B + LoRA)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.65, top=min(0.72, max(losses) + 0.01))

    plt.tight_layout()
    path = OUTPUT_DIR / "dpo_loss_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")
    return path

def plot_reward_accuracy(stats):
    steps = [s["checkpoint"] for s in stats]
    accs = [s["last_metrics"].get("reward_acc", 0) for s in stats]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps, accs, 'g-', linewidth=1.5)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward Accuracy")
    ax.set_title("Précision du Reward — Entraînement DPO")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "dpo_reward_accuracy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")
    return path

def plot_chosen_rejected_ratio(stats):
    steps = [s["checkpoint"] for s in stats]
    chosen = [s["last_metrics"].get("chosen_ratio", 0) for s in stats]
    rejected = [s["last_metrics"].get("rejected_ratio", 0) for s in stats]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, chosen, 'b-', linewidth=1, label="Ratio Chosen")
    ax.plot(steps, rejected, 'r-', linewidth=1, label="Ratio Rejected")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Log Ratio")
    ax.set_title("Rapports Chosen/Rejected — Entraînement DPO")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "dpo_chosen_rejected.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")
    return path

def main():
    print("Lecture des stats checkpoints DPO...")
    stats = load_all_stats()
    print(f"  → {len(stats)} checkpoints trouvés")
    if not stats:
        print("Aucune stat trouvée.")
        sys.exit(1)

    print(f"  → Steps: {stats[0]['checkpoint']} → {stats[-1]['checkpoint']}")
    print(f"  → Loss: {stats[0]['last_metrics']['loss']:.4f} → {stats[-1]['last_metrics']['loss']:.4f}")

    print("\nGénération des courbes...")
    plot_loss_curve(stats)
    plot_reward_accuracy(stats)
    plot_chosen_rejected_ratio(stats)

    print(f"\n✅ Courbes sauvegardées dans {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
