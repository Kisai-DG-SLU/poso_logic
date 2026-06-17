"""
API FastAPI pour le modèle DPO fine-tuné - Déploiement vLLM haute performance
Livrable 3 : Endpoint de démonstration API déployé (vLLM)
Note : nécessite vLLM 0.8.5 installé (pip install vllm==0.8.5 ou Docker)
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uvicorn

logger = logging.getLogger(__name__)

os.environ['VLLM_USE_V1'] = '0'

VLLM_AVAILABLE = False
try:
    from vllm.platforms.cuda import CudaPlatform
    import vllm.platforms as vp
    vp._current_platform = CudaPlatform()

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)

    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    logger.info("vLLM détecté - mode haute performance")
except ImportError:
    logger.warning("vLLM non installé - utiliser Docker (cf. README)")

TRACE_LOG = Path("/app/logs/api_trace.jsonl")
TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="API Triage Médical CHSA - vLLM Haute Performance",
    description="Agent IA de triage médical spécialisé via DPO (Qwen3-1.7B) - Inférence vLLM",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = str(Path("/app/models/merged_dpo").resolve()) if Path("/app/models/merged_dpo").exists() else "/mnt/prod/models/merged_dpo_vllm"

llm = None
model_name_loaded = None


class TriageRequest(BaseModel):
    symptoms: List[str]
    antecedents: Optional[List[str]] = []
    constantes: Optional[Dict] = None
    notes: Optional[str] = ""
    patient_age: Optional[int] = None
    priority_guess: Optional[str] = None

    @field_validator('symptoms', mode='before')
    @classmethod
    def normalize_symptoms(cls, v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(item) for item in v]
        raise ValueError(f"symptoms must be a string or list, got {type(v)}")


class TriageResponse(BaseModel):
    priority_level: str
    recommendation: str
    confidence: float
    reasoning: str
    model_response: str


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7


class GenerateResponse(BaseModel):
    response: str
    model: str
    tokens_generated: int


@app.on_event("startup")
async def load_model():
    global llm, model_name_loaded
    if not VLLM_AVAILABLE:
        logger.warning("vLLM non disponible - l'API ne fonctionnera pas")
        return

    try:
        logger.info(f"Chargement du modèle vLLM: {MODEL_PATH}")
        llm = LLM(
            model=MODEL_PATH,
            max_model_len=1024,
            gpu_memory_utilization=0.70,
            enforce_eager=True,
            tensor_parallel_size=1,
        )
        model_name_loaded = "Qwen3-1.7B-DPO (merged)"
        logger.info(f"Modèle vLLM chargé: {model_name_loaded}")
    except Exception as e:
        logger.error(f"Erreur chargement vLLM: {e}")
        llm = None


@app.get("/")
async def root():
    return {
        "service": "Triage Médical CHSA - vLLM",
        "status": "operational" if VLLM_AVAILABLE and llm else "degraded",
        "model": model_name_loaded or "Qwen3-1.7B-DPO (merged)",
        "engine": "vLLM" if VLLM_AVAILABLE else "non disponible (pip install vllm==0.8.5 ou Docker)",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if VLLM_AVAILABLE and llm else "unhealthy",
        "vllm_available": VLLM_AVAILABLE,
        "model_loaded": llm is not None,
    }


def build_triage_prompt(request: TriageRequest) -> str:
    symptoms_text = ", ".join(request.symptoms)
    antecedents_text = ", ".join(request.antecedents) if request.antecedents else "aucun"
    return f"""Instruction: Tu es un assistant médical aux urgences. Évalue la priorité du patient suivant.

Symptômes: {symptoms_text}
Antécédents: {antecedents_text}
Âge: {request.patient_age if request.patient_age else 'non spécifié'}
Notes: {request.notes if request.notes else 'aucune'}

Donne une évaluation avec:
- Niveau de priorité (max/high/medium/low)
- Recommandation
- Confiance (0-1)
- Raisonnement

Réponse:"""


def parse_priority(response_text: str):
    priority = "medium"
    recommendation = "Consultation dans l'heure"
    confidence = 0.75
    reasoning = response_text[:1000]

    if "max" in response_text.lower() or "urgence vitale" in response_text.lower():
        priority = "max"
        recommendation = "Appeler le 15 - Urgence vitale"
        confidence = 0.90
    elif "high" in response_text.lower() or "urgent" in response_text.lower():
        priority = "high"
        recommendation = "Consultation urgente < 30 min"
        confidence = 0.80
    elif "low" in response_text.lower():
        priority = "low"
        recommendation = "Consultation différée < 24h"
        confidence = 0.70

    return priority, recommendation, confidence, reasoning


@app.post("/triage", response_model=TriageResponse)
async def evaluate_triage(request: TriageRequest):
    if not VLLM_AVAILABLE or llm is None:
        raise HTTPException(status_code=503, detail="vLLM non disponible - utiliser Docker ou pip install vllm==0.8.5")

    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        prompt = build_triage_prompt(request)
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=200,
        )

        outputs = llm.generate([prompt], sampling_params)
        response_text = outputs[0].outputs[0].text.strip()

        priority, recommendation, confidence, reasoning = parse_priority(response_text)

        latency_ms = (time.time() - start_time) * 1000

        trace_entry = {
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endpoint": "/triage",
            "latency_ms": round(latency_ms, 2),
            "engine": "vLLM",
            "priority_level": priority,
            "confidence": confidence,
            "tokens_generated": len(response_text.split()),
        }
        with open(TRACE_LOG, "a") as f:
            f.write(json.dumps(trace_entry) + "\n")

        return TriageResponse(
            priority_level=priority,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            model_response=response_text,
        )

    except Exception as e:
        logger.error(f"Erreur vLLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if not VLLM_AVAILABLE or llm is None:
        raise HTTPException(status_code=503, detail="vLLM non disponible")

    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        outputs = llm.generate([request.prompt], sampling_params)
        response_text = outputs[0].outputs[0].text.strip()
        tokens_generated = len(response_text.split())

        latency_ms = (time.time() - start_time) * 1000

        trace_entry = {
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endpoint": "/generate",
            "latency_ms": round(latency_ms, 2),
            "engine": "vLLM",
            "tokens_generated": tokens_generated,
        }
        with open(TRACE_LOG, "a") as f:
            f.write(json.dumps(trace_entry) + "\n")

        return GenerateResponse(
            response=response_text,
            model="Qwen3-1.7B-DPO-vLLM",
            tokens_generated=tokens_generated,
        )

    except Exception as e:
        logger.error(f"Erreur vLLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/traces")
async def get_traces(limit: int = 50):
    if not TRACE_LOG.exists():
        return {"traces": []}
    traces = []
    with open(TRACE_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return {"traces": traces[-limit:]}


if __name__ == "__main__":
    if not VLLM_AVAILABLE:
        logger.error("vLLM requis. Installez: pip install vllm==0.8.5")
        logger.error("Ou utilisez Docker: docker build -t posologic . && docker run --gpus all posologic")
        exit(1)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
