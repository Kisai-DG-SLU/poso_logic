"""
API FastAPI pour le POC de triage médical (version démo, sans vLLM)
Étape 3 - Déploiement endpoint — version légère de démonstration

Points clés :
- Version minimale sans dépendance vLLM : parsing heuristique des symptômes
- Endpoints : / (status), /health, /triage (règles), /generate (placeholder)
- Port 8000 (vs 8001 pour la version vLLM)
- Utilisée pour les tests rapides et la validation du format API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional, List
import uvicorn

app = FastAPI(
    title="API Triage Médical CHSA",
    description="Agent IA de triage médical - Service des urgences",
    version="0.1.0"
)

class TriageRequest(BaseModel):
    symptoms: str | list[str]
    antecedents: Optional[List[str]] = []
    constantes: Optional[dict] = None
    notes: Optional[str] = ""

    @field_validator('symptoms', mode='before')
    @classmethod
    def normalize_symptoms(cls, v):
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return ", ".join(str(item) for item in v)
        raise ValueError(f"symptoms must be a string or list of strings, got {type(v)}")

class TriageResponse(BaseModel):
    priority_level: str
    recommendation: str
    confidence: float
    reasoning: str

@app.get("/")
async def root():
    return {"service": "Triage Médical CHSA", "status": "operational"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/triage", response_model=TriageResponse)
async def evaluate_triage(request: TriageRequest):
    """
    Évalue le niveau de priorité du patient
    """
    symptoms_text = request.symptoms
    
    priority = "medium"
    recommendation = "Consulta tion dans l'heure"
    confidence = 0.75
    reasoning = f"Patient présente: {symptoms_text}"
    
    if any(word in symptoms_text.lower() for word in ["douleur thoracique", "difficulté respiratoire", "perte conscience"]):
        priority = "max"
        recommendation = "Appeler le 15 - Urgence vitale"
        confidence = 0.90
        reasoning = "Signes d'urgence vitale détectés"
    
    elif any(word in symptoms_text.lower() for word in ["douleur", "fièvre haute", "saignement"]):
        priority = "high" 
        recommendation = "Consulta tion urgente < 30 min"
        confidence = 0.80
    
    return TriageResponse(
        priority_level=priority,
        recommendation=recommendation,
        confidence=confidence,
        reasoning=reasoning
    )

@app.post("/generate")
async def generate(request: dict):
    """
    Génère une réponse avec le modèle fine-tuné
    """
    prompt = request.get("prompt", "")
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt requis")
    
    return {
        "response": "[Intégration vLLM requise]",
        "model": "Qwen3-1.7B-CHSA",
        "status": "demo"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)