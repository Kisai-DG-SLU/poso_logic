#!/bin/bash
# Script pour démarrer Jupyter Lab optimisé pour la présentation du mentorat

echo "Démarrage de Jupyter Lab pour la présentation du mentorat..."

cd /mnt/prod

# Trouver un port disponible commençant par 8888
PORT=8888
while netstat -tuln | grep -q ":$PORT "; do
  PORT=$((PORT+1))
done

# Lancer Jupyter Lab en arrière-plan
nohup pixi run jupyter lab \
  --ip=0.0.0.0 \
  --port=$PORT \
  --no-browser \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.root_dir=/mnt/prod \
  --ServerApp.config_file=${HOME}/.jupyter/jupyter_lab_config.py \
  --ServerApp.notebook_dir=/mnt/prod/docs/mentorat \
  --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
  --IdentityProvider.token='' \
  > jupyter.log 2>&1 &

echo "Jupyter Lab démarré sur le port $PORT"
echo "Accédez à Jupyter via : http://localhost:$PORT/"
echo "Répertoire initial : /mnt/prod/docs/mentorat"
echo "Assurez-vous de mapper le port via kubectl port-forward si nécessaire"

# Attendre que Jupyter soit prêt
sleep 2
echo
echo "Logs de démarrage :"
tail -n 10 jupyter.log
echo
echo "Pour arrêter Jupyter : pkill -f jupyter"
