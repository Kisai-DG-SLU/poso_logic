#!/bin/bash
# Script de lancement sécurisé pour l'entraînement DPO

echo "🚀 Préparation du lancement de l'entraînement DPO..."

# Vérification de l'environnement
echo "📋 Vérification de l'environnement..."

# Vérifier que nous sommes dans pixi
if [ -z "$PIXI_PROJECT_ROOT" ]; then
    echo "❌ Erreur : Cet script doit être exécuté dans l'environnement pixi"
    echo "Utilisez : pixi run bash run_dpo_training.sh"
    exit 1
fi

# Vérifier la présence du GPU
if ! nvidia-smi &> /dev/null; then
    echo "❌ Erreur : GPU non détecté"
    exit 1
fi

# Afficher l'état du GPU
echo "🖥️ État du GPU:"
nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv

# Vérifier l'espace disque
echo -e "\n💾 Espace disque disponible:"
df -h | grep -E "/$|/mnt/prod"

# Test rapide du script
echo -e "\n🧪 Test rapide du script d'entraînement..."
cd /mnt/prod/scripts
python 05_train_dpo_robust.py --test_mode

if [ $? -ne 0 ]; then
    echo "❌ Le test a échoué. Corrigez les erreurs avant de lancer l'entraînement."
    exit 1
fi

echo -e "\n✅ Tous les tests sont passés!"

# Création d'un dossier pour les logs timestampés
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/mnt/prod/logs/dpo_training_${TIMESTAMP}.log"

echo -e "\n📝 Les logs seront sauvegardés dans: $LOG_FILE"

# Options de lancement
echo -e "\n⚙️ Configuration de l'entraînement:"
echo "- Batch size: 1"
echo "- Gradient accumulation: 64"
echo "- Learning rate: 2e-6"
echo "- Max sequence length: 256"
echo "- Reprise depuis checkpoint: Oui"

# Demander confirmation
echo -e "\n⏸️ Appuyez sur ENTER pour démarrer l'entraînement DPO complet ou Ctrl+C pour annuler..."
read

# Lancer l'entraînement avec nohup pour pouvoir fermer le terminal
echo "🔥 Lancement de l'entraînement DPO..."
echo "💡 Utilisez 'tail -f $LOG_FILE' pour suivre la progression"
echo "💡 L'entraînement continuera même si vous fermez ce terminal"

nohup python 05_train_dpo_robust.py \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 2e-6 \
    --max_seq_length 256 \
    --save_steps 25 \
    --resume_from_checkpoint True \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo -e "\n📌 PID du processus d'entraînement: $PID"
echo "Pour arrêter l'entraînement : kill $PID"

# Afficher les premières lignes du log
sleep 5
echo -e "\n📜 Premières lignes du log:"
head -n 20 "$LOG_FILE"

echo -e "\n✅ L'entraînement DPO a démarré avec succès!"
echo "Suivez la progression avec: tail -f $LOG_FILE"