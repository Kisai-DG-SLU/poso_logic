#!/bin/bash
LOG=$(cat /mnt/prod/models/checkpoints/dpo_a2_optimized/training.log)
PID=$(cat /mnt/prod/models/checkpoints/dpo_a2_optimized/training.pid)

# Vérifier si le processus tourne
if ps -p $PID > /dev/null; then
    echo "✅ Processus actif (PID: $PID)"
    
    # Dernières lignes du log
    echo -e "\n📊 Progression:"
    tail -3 $LOG | grep -E "(Loss|Step|ETA)"
    
    # État GPU
    echo -e "\n🖥️ GPU:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
    
    # Checkpoints
    echo -e "\n💾 Checkpoints:"
    ls -la /mnt/prod/models/checkpoints/dpo_a2_optimized/checkpoint-* 2>/dev/null | tail -3
else
    echo "❌ Processus terminé ou arrêté"
fi
