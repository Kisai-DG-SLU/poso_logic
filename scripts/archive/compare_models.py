"""
Script pour comparer le modèle de base, SFT et DPO
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import argparse

# Chemins des modèles
BASE_MODEL = "Qwen/Qwen3-1.7B"
SFT_MODEL = "/mnt/prod/models/checkpoints/sft_final"
DPO_MODEL = "/mnt/prod/models/checkpoints/dpo_minimal"

def load_model(model_path, device="cuda"):
    """Charge un modèle et son tokenizer"""
    print(f"Chargement du modèle {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Génère une réponse à partir d'un prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def compare_models(prompt, max_new_tokens=256):
    """Compare les réponses de tous les modèles"""
    results = {}
    
    # Charger et générer pour chaque modèle
    for name, path in [
        ("Base", BASE_MODEL),
        ("SFT", SFT_MODEL),
        ("DPO", DPO_MODEL)
    ]:
        try:
            model, tokenizer = load_model(path)
            response = generate_response(model, tokenizer, prompt, max_new_tokens)
            results[name] = response
            print(f"\n=== Réponse du modèle {name} ===\n{response}\n")
            
            # Libérer la mémoire
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Erreur avec le modèle {name}: {e}")
            results[name] = f"Erreur: {e}"
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Comparer les réponses des modèles")
    parser.add_argument("--prompt", type=str, default="Je suis un homme de 45 ans et j'ai une douleur thoracique qui irradie dans le bras gauche depuis 30 minutes. C'est une sensation d'oppression. J'ai aussi des sueurs. Que dois-je faire?",
                        help="Question médicale à poser")
    args = parser.parse_args()
    
    prompt = args.prompt
    print(f"Question: {prompt}")
    results = compare_models(prompt)

if __name__ == "__main__":
    main()
