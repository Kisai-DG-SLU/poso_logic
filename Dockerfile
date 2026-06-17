# ============================================================
# PosoLogic — Agent de Triage Médical CHSA
# Dockerfile de production pour déploiement vLLM + FastAPI
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

# Install core dependencies
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    transformers==4.51.0 \
    peft>=0.8.0 \
    accelerate>=1.0.0

# Install vLLM (CUDA 12.4 compatible)
RUN pip install --no-cache-dir vllm>=0.6.0

# ---- Stage 2: Runtime ----
FROM nvidia/cuda:12.4-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

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

# Model checkpoints (LoRA adapters)
# These are ~12 MB each, safe to include in image
COPY models/dpo_config.json /app/models/dpo_config.json
COPY models/sft_config.json /app/models/sft_config.json
COPY models/checkpoints/sft_final/ /app/models/checkpoints/sft_final/
COPY models/checkpoints/dpo_a2_optimized/final/ /app/models/checkpoints/dpo_final/

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
