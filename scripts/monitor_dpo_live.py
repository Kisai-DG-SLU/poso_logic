#!/usr/bin/env python3
import subprocess
import time
import sys

def get_gpu_info():
    """Récupérer les infos GPU"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            line = result.stdout.strip()
            util, used, total = line.split(', ')
            return f"GPU: {util}% | Mém: {used}/{total} MB"
    except:
        return "GPU: N/A"
    return "GPU: Erreur"

def check_log_file(log_path="/mnt/prod/logs/dpo_training_20260429_231020.log"):
    """Vérifier le fichier de log"""
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            return len(lines), lines[-1].strip() if lines else "Vide"
    except:
        return 0, "Fichier non trouvé"

print("Monitoring de l'entraînement DPO...")
print("Appuyez sur Ctrl+C pour arrêter\n")

last_line_count = 0
stuck_count = 0

try:
    while True:
        gpu_info = get_gpu_info()
        line_count, last_line = check_log_file()
        
        print(f"\r{time.strftime('%H:%M:%S')} | {gpu_info} | Lignes log: {line_count} | ", end="")
        
        if line_count == last_line_count:
            stuck_count += 1
            print(f"⚠️  Pas de progression ({stuck_count})", end="")
        else:
            stuck_count = 0
            print(f"✅ Progression: +{line_count - last_line_count} lignes", end="")
            last_line_count = line_count
        
        sys.stdout.flush()
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\n\nMonitoring arrêté.")
    print(f"Dernière ligne du log: {last_line}")