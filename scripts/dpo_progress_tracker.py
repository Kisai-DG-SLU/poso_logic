#!/usr/bin/env python3
"""
Tracker de progression pour l'entraînement DPO
Génère des rapports réguliers et estime le temps restant
"""
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CHECKPOINT_DIR = Path("/mnt/prod/models/checkpoints/dpo_a2_optimized")
OUTPUT_DIR = Path("/mnt/memory/dpo_progress")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_checkpoint_stats():
    """Charge les statistiques de tous les checkpoints"""
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint-*/training_stats.json"))
    stats = []
    
    for cp in checkpoints:
        with open(cp, 'r') as f:
            data = json.load(f)
            stats.append(data)
    
    return stats

def generate_progress_plot(stats):
    """Génère un graphique de progression"""
    if not stats:
        return
    
    steps = [s['global_step'] for s in stats]
    speeds = [s['steps_per_second'] for s in stats]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Progression
    ax1.plot(steps, marker='o')
    ax1.set_xlabel('Checkpoint')
    ax1.set_ylabel('Steps complétés')
    ax1.set_title('Progression de l\'entraînement DPO')
    ax1.grid(True)
    
    # Vitesse
    ax2.plot(steps, speeds, marker='o', color='orange')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Steps/seconde')
    ax2.set_title('Vitesse d\'entraînement')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dpo_progress.png')
    plt.close()

def estimate_completion(stats):
    """Estime le temps de complétion"""
    if not stats:
        return None
    
    latest = stats[-1]
    current_step = latest['global_step']
    speed = latest['steps_per_second']
    
    # Total estimé (basé sur 196k exemples, batch=1, accum=8)
    total_steps = 196835 // 8  # ~24,604 steps
    
    remaining_steps = total_steps - current_step
    remaining_seconds = remaining_steps / speed if speed > 0 else 0
    
    eta = datetime.now() + timedelta(seconds=remaining_seconds)
    
    return {
        'current_step': current_step,
        'total_steps': total_steps,
        'progress_percent': (current_step / total_steps) * 100,
        'remaining_hours': remaining_seconds / 3600,
        'eta': eta.strftime('%Y-%m-%d %H:%M:%S'),
        'average_speed': speed
    }

def generate_report():
    """Génère un rapport complet"""
    stats = load_checkpoint_stats()
    
    if not stats:
        print("Aucun checkpoint trouvé. L'entraînement n'a pas encore commencé.")
        return
    
    # Générer le graphique
    generate_progress_plot(stats)
    
    # Estimer la complétion
    estimation = estimate_completion(stats)
    
    # Créer le rapport
    report = {
        'timestamp': datetime.now().isoformat(),
        'num_checkpoints': len(stats),
        'latest_checkpoint': stats[-1]['global_step'],
        'estimation': estimation,
        'checkpoints': stats
    }
    
    # Sauvegarder le rapport
    with open(OUTPUT_DIR / 'latest_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Afficher le résumé
    print("=" * 60)
    print("RAPPORT DE PROGRESSION DPO - PROJET P14")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoints trouvés: {len(stats)}")
    print(f"Dernier checkpoint: Step {estimation['current_step']}/{estimation['total_steps']}")
    print(f"Progression: {estimation['progress_percent']:.1f}%")
    print(f"Vitesse moyenne: {estimation['average_speed']:.2f} steps/sec")
    print(f"Temps restant estimé: {estimation['remaining_hours']:.1f} heures")
    print(f"ETA: {estimation['eta']}")
    print("=" * 60)
    print(f"Graphique sauvegardé: {OUTPUT_DIR}/dpo_progress.png")
    print(f"Rapport JSON: {OUTPUT_DIR}/latest_report.json")

if __name__ == "__main__":
    while True:
        try:
            generate_report()
            
            # Attendre 30 minutes avant la prochaine mise à jour
            print("\nProchaine mise à jour dans 30 minutes...")
            time.sleep(1800)
            
        except KeyboardInterrupt:
            print("\nArrêt du tracker.")
            break
        except Exception as e:
            print(f"Erreur: {e}")
            time.sleep(60)  # Réessayer dans 1 minute en cas d'erreur