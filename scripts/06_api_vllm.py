"""
API FastAPI pour le modèle DPO fine-tuné - Agent de Triage Médical CHSA
Livrable 3 : Endpoint de démonstration API déployé
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Triage Médical CHSA - Modèle DPO",
    description="Agent IA de triage médical spécialisé via DPO (Qwen3-1.7B)",
    version="1.0.0",
)

# Configuration
BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER_PATH = "/mnt/prod/models/checkpoints/dpo_a2_optimized/final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Variables globales pour le modèle
model = None
tokenizer = None


class TriageRequest(BaseModel):
    symptoms: List[str]
    antecedents: Optional[List[str]] = []
    constantes: Optional[Dict] = None
    notes: Optional[str] = ""


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
    """Charge le modèle DPO au démarrage"""
    global model, tokenizer

    try:
        logger.info(f"Chargement du tokenizer depuis {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Chargement du modèle de base {BASE_MODEL}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        logger.info(f"Chargement de l'adaptateur LoRA depuis {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

        if not torch.cuda.is_available():
            model = model.to("cpu")

        model.eval()

        logger.info("Modèle DPO chargé avec succès !")

    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        import traceback

        traceback.print_exc()
        raise


@app.get("/")
async def root():
    return {
        "service": "Triage Médical CHSA - Modèle DPO",
        "status": "operational",
        "model": "Qwen3-1.7B + DPO LoRA",
        "device": DEVICE,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": DEVICE,
    }


@app.post("/triage", response_model=TriageResponse)
async def evaluate_triage(request: TriageRequest):
    """
    Évalue le niveau de priorité du patient en utilisant le modèle DPO
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        symptoms_text = ", ".join(request.symptoms)
        antecedents_text = (
            ", ".join(request.antecedents) if request.antecedents else "aucun"
        )

        prompt = f"""Instruction: Tu es un assistant médical aux urgences. Évalue la priorité du patient suivant.

Symptômes: {symptoms_text}
Antécédents: {antecedents_text}
Notes: {request.notes if request.notes else 'aucune'}

Donne une évaluation avec:
- Niveau de priorité (max/high/medium/low)
- Recommandation
- Confiance (0-1)
- Raisonnement

Réponse:"""

        # Tokenization
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Génération avec le modèle DPO
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Enlever le prompt pour ne garder que la réponse
        response_text = generated_text[len(prompt) :].strip()

        # Parsing simple de la réponse
        priority = "medium"
        recommendation = "Consultation dans l'heure"
        confidence = 0.75
        reasoning = response_text[:200]

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

        return TriageResponse(
            priority_level=priority,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            model_response=response_text,
        )

    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Génère une réponse avec le modèle fine-tuné
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        inputs = tokenizer(
            request.prompt, return_tensors="pt", truncation=True, max_length=512
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = response_text[len(request.prompt) :].strip()
        tokens_generated = len(tokenizer.encode(response_only))

        return GenerateResponse(
            response=response_only,
            model="Qwen3-1.7B-DPO-CHSA",
            tokens_generated=tokens_generated,
        )

    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
