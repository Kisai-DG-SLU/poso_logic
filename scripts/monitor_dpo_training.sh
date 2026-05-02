#!/bin/bash
# Script de surveillance de l'entraînement DPO

echo "📊 Surveillance de l'entraînement DPO"

# Trouver le dernier fichier de log
LATEST_LOG=$(ls -t /mnt/prod/logs/dpo_training_*.log 2>/dev/null | head -n1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ Aucun log d'entraînement trouvé"
    exit 1
fi

echo "📝 Fichier de log: $LATEST_LOG"

# Fonction pour extraire les métriques du log
show_metrics() {
    echo -e "\n📈 Dernières métriques:"
    grep -E "(Step [0-9]+|Eval Loss|Eval Reward Acc)" "$LATEST_LOG" | tail -n 10
}

# Fonction pour vérifier l'état du GPU
show_gpu_status() {
    echo -e "\n🖥️ État du GPU:"
    nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv
}

# Fonction pour vérifier l'état des checkpoints
show_checkpoints() {
    echo -e "\n💾 Checkpoints sauvegardés:"
    ls -lh /mnt/prod/models/checkpoints/dpo_robust/checkpoint-* 2>/dev/null | tail -n 5
    
    if [ -d "/mnt/prod/models/checkpoints/dpo_robust/best_model" ]; then
        echo -e "\n🏆 Meilleur modèle:"
        ls -lh /mnt/prod/models/checkpoints/dpo_robust/best_model/
    fi
}

# Fonction pour vérifier si l'entraînement est toujours en cours
check_training_status() {
    # Chercher le processus python exécutant le script DPO
    if pgrep -f "05_train_dpo_robust.py" > /dev/null; then
        echo -e "\n✅ L'entraînement est en cours (PID: $(pgrep -f '05_train_dpo_robust.py'))"
    else
        echo -e "\n⚠️ Aucun processus d'entraînement actif détecté"
    fi
}

# Menu interactif
while true; do
    echo -e "\n=========================================="
    check_training_status
    echo -e "\nOptions:"
    echo "1) Afficher les dernières métriques"
    echo "2) Afficher l'état du GPU"
    echo "3) Afficher les checkpoints"
    echo "4) Suivre le log en temps réel (tail -f)"
    echo "5) Afficher les erreurs récentes"
    echo "6) Rafraîchir"
    echo "q) Quitter"
    
    read -p "Choix: " choice
    
    case $choice in
        1)
            show_metrics
            ;;
        2)
            show_gpu_status
            ;;
        3)
            show_checkpoints
            ;;
        4)
            echo "Appuyez sur Ctrl+C pour revenir au menu"
            sleep 2
            tail -f "$LATEST_LOG"
            ;;
        5)
            echo -e "\n❌ Dernières erreurs:"
            grep -E "(Error|ERROR|Exception|Traceback)" "$LATEST_LOG" | tail -n 20
            ;;
        6)
            clear
            ;;
        q)
            exit 0
            ;;
        *)
            echo "Option invalide"
            ;;
    esac
    
    if [ "$choice" != "6" ]; then
        echo -e "\nAppuyez sur ENTER pour continuer..."
        read
    fi
done