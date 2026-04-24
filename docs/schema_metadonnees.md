# Schema des Metadonnees - Dataset Triage Medical

## Schema JSON

```json
{
    "instruction": "str",
    "response": "str",
    "source": "str",
    "language": "str",
    "symptoms": ["list"],
    "antecedents": ["list"],
    "constantes": ["list"],
    "priority_level": "str",
    "confidence_score": "float",
    "category": "str"
}
```

## Description des champs

| Champ | Type | Description | Exemple |
|-------|------|-------------|--------|
| instruction | str | Question/instruction pour le modele | "Patient presente des douleurs thoraciques..." |
| response | str | Reponse attendue du modele | "Priorite: Urgente. Appeler le 15..." |
| source | str | Source du donnees | "frenchmedmcqa", "medquad" |
| language | str | Langue (fr/en) | "fr", "en" |
| symptoms | list | Symptomes decrits | ["douleur thoracique", "essoufflement"] |
| antecedents | list | Antecedents medicaux | ["diabete", "hypertension"] |
| constantes | list | Constantes vitales | ["PA: 140/90", "FC: 90"] |
| priority_level | str | Niveau de priorite | "max", "high", "medium", "low" |
| confidence_score | float | Score de confiance (0-1) | 0.85 |
| category | str | Categorie medicale | "cardiologie", "neurologie" |

## Niveaux de priorite (triage)

- **max** : Urgence vitale, immediate
- **high** : Urgence, < 15 min
- **medium** : Urgence relative, < 1h
- **low** : Differable, > 1h

## Format de sortie

- HuggingFace Dataset (.arrow)
- JSONL pour归档
- Train/Val/Test splits (80/10/10)