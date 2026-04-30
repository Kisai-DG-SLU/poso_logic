#!/bin/bash
# Script de lancement pour l'entraînement DPO COMPLET
# Projet P14 - Respect strict des livrables

echo "=========================================="
echo "PROJET P14 - ENTRAÎNEMENT DPO COMPLET"
echo "Dataset: 196,835 exemples"
echo "Durée estimée: 46-50 heures"
echo "=========================================="

# Configuration
SCRIPT_PATH="/mnt/prod/scripts/05_train_dpo_a2_optimized.py"
LOG_DIR="/mnt/prod/logs"
CHECKPOINT_DIR="/mnt/prod/models/checkpoints/dpo_a2_optimized"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/dpo_full_$TIMESTAMP.log"

# Créer les répertoires
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

# Informations système
echo "Date de début: $(date)" | tee -a $LOG_FILE
echo "GPU disponible:" | tee -a $LOG_FILE
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | tee -a $LOG_FILE

# Vérification espace disque
echo -e "\nEspace disque:" | tee -a $LOG_FILE
df -h /mnt/prod | tee -a $LOG_FILE

# Confirmation
echo -e "\n⚠️  ATTENTION: Cet entraînement prendra ~48 heures"
echo "Les checkpoints seront sauvegardés toutes les 500 étapes"
echo "Le processus continuera même si vous fermez ce terminal"
echo -e "\nAppuyez sur ENTER pour confirmer le lancement..."
read

# Lancement en arrière-plan avec nohup
echo "Lancement de l'entraînement..." | tee -a $LOG_FILE
nohup python -u $SCRIPT_PATH > $LOG_FILE 2>&1 &
PID=$!

echo "✅ Entraînement lancé avec PID: $PID" | tee -a $LOG_FILE
echo "📋 Logs: tail -f $LOG_FILE"
echo "🔍 Monitoring: watch -n 60 nvidia-smi"
echo "⏹️  Pour arrêter: kill $PID"

# Sauvegarder les infos du processus
echo $PID > $CHECKPOINT_DIR/training.pid
echo $LOG_FILE > $CHECKPOINT_DIR/training.log

# Script de monitoring automatique
cat > $LOG_DIR/monitor_dpo.sh << 'EOF'
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
EOF

chmod +x $LOG_DIR/monitor_dpo.sh

echo -e "\n📊 Pour surveiller: $LOG_DIR/monitor_dpo.sh"
echo "💡 Conseil: Lancez 'watch -n 300 $LOG_DIR/monitor_dpo.sh' dans un autre terminal"

# Programmer une sauvegarde mémoire dans 24h
echo "memory-push 'Entraînement DPO P14 - Checkpoint 24h'" | at now + 24 hours 2>/dev/null || true

echo -e "\n🚀 L'entraînement DPO complet est maintenant en cours..."
echo "Prochain checkpoint dans environ 1 heure"