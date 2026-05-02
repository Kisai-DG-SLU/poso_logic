#!/bin/bash
cd /mnt/prod
source .pixi/env/bin/activate 2>/dev/null || true
export PYTHONPATH=/mnt/prod:$PYTHONPATH
nohup python scripts/06_api_vllm.py > /tmp/api_dpo.log 2>&1 &
echo $! > /tmp/api_dpo.pid
echo "API démarrée avec PID: $(cat /tmp/api_dpo.pid)"
sleep 30
echo "Test de l'API:"
curl -s http://localhost:8000/health | python3 -m json.tool || echo "En attente du démarrage..."
