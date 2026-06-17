# ============================================================
# PosoLogic — Agent de Triage Médical CHSA
# Dockerfile de production pour déploiement vLLM + FastAPI
# vLLM 0.8.5 + torch 2.6.0+cu124 + modèle DPO merged
# ============================================================

# ---- Stage 1: Builder ----
FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# vLLM requires newer pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core dependencies with CUDA 12.4 extra index
RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    transformers==4.51.0 \
    peft>=0.8.0 \
    accelerate>=1.0.0

# Install vLLM 0.8.5 (CUDA 12.4 compatible)
RUN pip install --no-cache-dir vllm==0.8.5

# ---- Stage 2: Runtime ----
FROM nvidia/cuda:12.4-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    VLLM_USE_V1=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash poso && \
    mkdir -p /app /app/models /app/logs /app/mlruns && \
    chown -R poso:poso /app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# ---- Application ----

# API and scripts
COPY scripts/06_api_vllm.py /app/api.py
COPY scripts/06_api_dpo.py /app/api_dpo.py
COPY scripts/mlflow_tracker.py /app/mlflow_tracker.py

# Anonymisation (Presidio)
RUN pip install --no-cache-dir presidio-analyzer>=2.2.360 presidio-anonymizer>=2.2.360

# FastAPI + uvicorn
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# MLflow for metrics dashboard
RUN pip install --no-cache-dir mlflow>=2.14.0 matplotlib>=3.8.0

# Modèle DPO merged (poids LoRA fusionnés dans le modèle de base)
# Généré par scripts/merge_lora_to_vllm.py
COPY models/merged_dpo_vllm/ /app/models/merged_dpo/

# ---- Health check ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# ---- Security ----
# Switch to non-root user
USER poso

# ---- Runtime ----
EXPOSE 8000

# Default: start the API server
# Override with CMD for different modes:
#   docker run ... posologic:latest                    → API server
#   docker run ... posologic:latest mlflow             → MLflow dashboard
#   docker run ... posologic:latest evaluate           → Model evaluation

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"]
CMD ["serve"]
