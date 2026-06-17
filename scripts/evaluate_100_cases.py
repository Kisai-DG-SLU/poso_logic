"""
Évaluation clinique étendue sur 100 cas de triage médical
Génère les cas, les évalue via le modèle DPO, produit un rapport
"""
import json
import random
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER_PATH = "/mnt/prod/models/checkpoints/dpo_a2_optimized/final"
OUTPUT_DIR = Path("/mnt/prod/eval_reports")
NUM_CASES = 100
SEED = 42

SYMPTOM_TEMPLATES = {
    "max": [
        ("Arrêt cardiaque", "Patient inconscient, pas de respiration, pas de pouls depuis 5 minutes.", "Immédiat - niveau 1"),
        ("Détresse respiratoire aiguë", "Patient 60 ans, détresse respiratoire sévère, SpO2 82%, cyanose, impossibilité de parler. ATCD: BPCO.", "Immédiat - niveau 1"),
        ("AVC hémorragique", "Patient 75 ans, perte de conscience brutale, mydriase unilatérale, vomissements en jet, TA 220/110.", "Immédiat - niveau 1"),
        ("État de choc", "Patient 45 ans, pâleur extrême, TA 60/40, pouls filant, sueurs froides, hémorragie digestive.", "Immédiat - niveau 1"),
        ("Polytraumatisme", "Patient 30 ans, accident de la route, inconscient, fracture ouverte fémur, détresse respiratoire, GCS 7.", "Immédiat - niveau 1"),
        ("Embolie pulmonaire massive", "Patient 55 ans, douleur thoracique brutale, dyspnée massive, cyanose, TA 70/40, syncope.", "Immédiat - niveau 1"),
        ("Anaphylaxie stade 4", "Patient 25 ans, œdème de Quincke, stridor, urticaire généralisée, hypotension après injection antibiotique.", "Immédiat - niveau 1"),
        ("Tamponnade cardiaque", "Patient 65 ans, dyspnée croissante, hypotension, distension jugulaire, bruits cardiaques assourdis. ATCD: cancer pulmonaire.", "Immédiat - niveau 1"),
        ("Méningite fulminante", "Enfant 2 ans, fièvre 40°C, purpura extensif, raideur nuque, altération conscience, vomissements.", "Immédiat - niveau 1"),
        ("Dissection aortique", "Patient 55 ans, douleur thoracique déchirante irradiant dans le dos, TA asymétrique, HTA non traitée.", "Immédiat - niveau 1"),
    ],
    "high": [
        ("Infarctus du myocarde", "Patient 50 ans, douleur thoracique constrictive depuis 1h, irradiation bras gauche, sueurs, nausées, dyspnée.", "Urgent - < 30 min"),
        ("Occlusion intestinale", "Patient 70 ans, douleur abdominale violente, arrêt des matières et gaz, vomissements fécaloïdes, distension.", "Urgent - < 30 min"),
        ("Crise d'asthme aiguë", "Patient 35 ans, dyspnée sévère, wheezing, saturation 88%, fréquence respiratoire 35/min, utilisation muscles accessoires.", "Urgent - < 30 min"),
        ("Appendicite aiguë", "Patient 25 ans, douleur fosse iliaque droite, fièvre 38.5°C, défense, signe de Blumberg positif.", "Urgent - < 30 min"),
        ("Pneumothorax", "Patient 40 ans, douleur thoracique brutale, dyspnée, absence murmure vésiculaire à droite, Tachypnée.", "Urgent - < 30 min"),
        ("AVC ischémique", "Patient 72 ans, hémiplégie droite brutale, aphasie, déviation commissure, début il y a 45 minutes.", "Urgent - < 30 min"),
        ("Crise hypertensive", "Patient 60 ans, TA 240/130, céphalée intense, vision floue, nausées, épistaxis.", "Urgent - < 30 min"),
        ("Pancréatite aiguë", "Patient 45 ans, douleur épigastrique transfixiante, vomissements, fièvre, antécédent lithiase biliaire/alcool.", "Urgent - < 30 min"),
        ("Embolie pulmonaire", "Patient 48 ans, dyspnée d'apparition brutale, douleur thoracique, tachycardie, hémoptysie. ATCD: thrombose veineuse.", "Urgent - < 30 min"),
        ("Hémorragie digestive haute", "Patient 65 ans, hématémèse abondante, méléna, TA 100/60, pouls 110, pâleur. ATCD: ulcère gastrique.", "Urgent - < 30 min"),
        ("Fracture ouverte", "Patient 35 ans, fracture ouverte tibia, saignement modéré, douleur intense, déformation visible.", "Urgent - < 30 min"),
        ("Convulsions", "Patient enfant 4 ans, convulsion tonicoclonique généralisée depuis 10 minutes, fièvre 39°C, perte de connaissance.", "Urgent - < 30 min"),
        ("Hyperglycémie sévère", "Patient 30 ans, diabétique type 1, vomissements, douleur abdominale, respiration de Kussmaul, haleine cétonique.", "Urgent - < 30 min"),
        ("Rétention aiguë d'urine", "Patient 72 ans, impossibilité d'uriner depuis 8h, globe vésical douloureux, agitation. ATCD: hypertrophie prostatique.", "Urgent - < 30 min"),
        ("Déshydratation sévère", "Nourrisson 6 mois, diarrhée depuis 48h, pli cutané persistant, yeux creux, fontanelle déprimée.", "Urgent - < 30 min"),
    ],
    "medium": [
        ("Fracture simple", "Enfant 10 ans, chute à vélo, poignet gonflé et douloureux, déformation modérée, Mobilité doigts conservée.", "Dans les 2 heures - niveau 3"),
        ("Infection urinaire", "Femme 30 ans, brûlures mictionnelles, urines troubles, fièvre 38°C, douleur sus-pubienne.", "Dans les 2 heures - niveau 3"),
        ("Lombalgie aiguë", "Patient 40 ans, lumbago après effort, douleur lombaire sévère sans irradiation, pas de déficit neurologique.", "Dans les 2 heures - niveau 3"),
        ("Gastro-entérite", "Patient 25 ans, diarrhée aqueuse, vomissements, douleur abdominale diffuse, fièvre 38°C, sans signes de gravité.", "Dans les 2 heures - niveau 3"),
        ("Entorse cheville", "Patient 20 ans, entorse cheville en jouant au foot, gonflement modéré, douleur à l'appui, sans instabilité.", "Dans les 2 heures - niveau 3"),
        ("Migraine sévère", "Patient 35 ans, céphalée pulsatile unilatérale depuis 72h, photophobie, nausées, résistante au traitement habituel.", "Dans les 2 heures - niveau 3"),
        ("Bronchite aiguë", "Patient 50 ans, toux productive, fièvre 38.5°C, douleur thoracique modérée, crépitants auscultatoires.", "Dans les 2 heures - niveau 3"),
        ("Pharyngite érythémateuse", "Patient 22 ans, mal de gorge intense, fièvre 39°C, dysphagie, amygdales érythémateuses."  , "Dans les 2 heures - niveau 3"),
        ("Conjonctivite bactérienne", "Enfant 5 ans, œil rouge, pus, paupières collées au réveil, pas de douleur, pas de baisse vision.", "Dans les 2 heures - niveau 3"),
        ("Otite moyenne aiguë", "Enfant 3 ans, otalgie droite, fièvre 39°C, irritabilité, tympan congestif.", "Dans les 2 heures - niveau 3"),
    ],
    "low": [
        ("Renouvellement ordonnance", "Patient 55 ans, venu pour renouvellement traitement antihypertenseur, stable, TA bien contrôlée.", "Délai possible"),
        ("Verrues plantaires", "Patient 30 ans, verrues plantaires douloureuses, pas de fièvre, pas de rougeur, pas de signe infectieux.", "Délai possible"),
        ("Certificat médical", "Patient 28 ans, demande certificat médical pour reprise du sport, asymptomatique.", "Délai possible"),
        ("Bilan sanguin de routine", "Patient 60 ans, venu pour résultat prise de sang annuelle, asymptomatique.", "Délai possible"),
        ("Petite plaie superficielle", "Patient 35 ans, coupure superficielle doigt en cuisinant, saignement arrêté, propre.", "Délai possible"),
        ("Avis dermatologique", "Patient 45 ans, grain de beauté suspect, sans saignement, sans douleur, asymptomatique.", "Délai possible"),
        ("Consultation vaccination", "Enfant 2 ans, calendrier vaccinal, pas de fièvre, pas de symptôme.", "Délai possible"),
        ("Mycose cutanée", "Patient 30 ans, plaques rouges squameuses entre les orteils, prurit modéré, sans signe infectieux.", "Délai possible"),
        ("Réfraction visuelle", "Patient 25 ans, baisse acuité visuelle progressive, pas de douleur, pas de rougeur.", "Délai possible"),
        ("Constipation chronique", "Patient 50 ans, constipation depuis 2 semaines, pas de douleur abdominale, pas de vomissement.", "Délai possible"),
        ("Contrôle tensionnel", "Patient 65 ans, consultation de routine pour suivi HTA, TA 130/80 sous traitement.", "Délai possible"),
        ("Aptitude sportive", "Adolescent 14 ans, certificat d'aptitude au sport scolaire, asymptomatique.", "Délai possible"),
        ("Cors aux pieds", "Patient 55 ans, cors douloureux au pied, gêne à la marche, pas de diabète.", "Délai possible"),
        ("Besoin d'arrêt maladie", "Patient 35 ans, demande d'arrêt maladie pour fatigue, pas de fièvre, pas de signe clinique.", "Délai possible"),
        ("Conseil nutritionnel", "Patient 40 ans, conseils pour régime hypocalorique, IMC 28, pas de comorbidité.", "Délai possible"),
    ],
}

PRIORITY_ORDER = {"max": 0, "high": 1, "medium": 2, "low": 3}


def generate_cases(num_cases: int, seed: int = SEED) -> list:
    random.seed(seed)
    cases = []
    priorities = list(SYMPTOM_TEMPLATES.keys())
    weights = [0.15, 0.35, 0.30, 0.20]

    for i in range(num_cases):
        prio = random.choices(priorities, weights=weights)[0]
        template = random.choice(SYMPTOM_TEMPLATES[prio])
        cases.append({
            "id": i + 1,
            "priority": prio,
            "expected_priority": template[2],
            "instruction": template[1],
            "key_signs": [s.strip() for s in template[1].split(",")[:3]],
        })
    return cases


def format_prompt(instruction: str, prompt_lang: str = "fr") -> str:
    prefix = {
        "en": "You are a medical triage assistant. Evaluate the priority level.\n\nPatient presentation: ",
        "fr": "Vous êtes un assistant médical aux urgences. Évaluez le niveau de priorité.\n\nPrésentation du patient : ",
    }
    return f"{prefix[prompt_lang]}{instruction}\n\nRéponse (priorité, recommandation, raisonnement) :"


def load_model():
    print(f"Chargement du modèle {BASE_MODEL} avec adaptateur LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def evaluate_case(model, tokenizer, case: dict) -> dict:
    prompt = format_prompt(case["instruction"])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )
    latency = time.time() - start_time

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    generated = generated.strip()

    priority = "medium"
    if "max" in generated.lower() or "niveau 1" in generated.lower() or "immédiat" in generated.lower():
        priority = "max"
    elif "high" in generated.lower() or "urgent" in generated.lower() or "niveau 2" in generated.lower():
        priority = "high"
    elif "medium" in generated.lower() or "niveau 3" in generated.lower():
        priority = "medium"
    elif "low" in generated.lower() or "niveau 4" in generated.lower() or "délai" in generated.lower():
        priority = "low"

    correct = priority == case["priority"]
    return {
        "case_id": case["id"],
        "priority": case["priority"],
        "expected_priority": case["expected_priority"],
        "predicted_priority": priority,
        "correct": correct,
        "latency_seconds": round(latency, 3),
        "generated": generated[:200],
        "instruction": case["instruction"][:200],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Génération de {NUM_CASES} cas cliniques...")
    cases = generate_cases(NUM_CASES)
    print(f"  Distribution: max={sum(1 for c in cases if c['priority']=='max')}, "
          f"high={sum(1 for c in cases if c['priority']=='high')}, "
          f"medium={sum(1 for c in cases if c['priority']=='medium')}, "
          f"low={sum(1 for c in cases if c['priority']=='low')}")

    model, tokenizer = load_model()

    results = []
    errors = 0
    for i, case in enumerate(cases):
        print(f"  [{i+1}/{NUM_CASES}] Cas #{case['id']} ({case['priority']})...", end=" ")
        try:
            result = evaluate_case(model, tokenizer, case)
            results.append(result)
            status = "✅" if result["correct"] else "❌"
            print(f"{status} (latence: {result['latency_seconds']:.1f}s)")
        except Exception as e:
            print(f"❌ Erreur: {e}")
            errors += 1

    # Calculate stats
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total * 100 if total > 0 else 0
    avg_latency = sum(r["latency_seconds"] for r in results) / total if total > 0 else 0

    by_priority = {}
    for r in results:
        p = r["priority"]
        if p not in by_priority:
            by_priority[p] = {"total": 0, "correct": 0}
        by_priority[p]["total"] += 1
        if r["correct"]:
            by_priority[p]["correct"] += 1

    # Generate confusion-like stats
    confusion = {}
    for r in results:
        expected = r["priority"]
        predicted = r["predicted_priority"]
        key = f"{expected}→{predicted}"
        confusion[key] = confusion.get(key, 0) + 1

    report = {
        "num_cases": total,
        "errors": errors,
        "accuracy_pct": round(accuracy, 1),
        "average_latency_s": round(avg_latency, 2),
        "by_priority": {p: {"total": v["total"], "correct": v["correct"],
                            "pct": round(v["correct"]/v["total"]*100, 1)}
                        for p, v in by_priority.items()},
        "confusion": confusion,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = OUTPUT_DIR / "eval_100_cases_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Rapport sauvegardé: {report_path}")
    print(f"   Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"   Latence moyenne: {avg_latency:.2f}s")
    print(f"   Par priorité:")
    for p in ["max", "high", "medium", "low"]:
        if p in by_priority:
            bp = by_priority[p]
            print(f"     {p}: {bp['correct']}/{bp['total']} ({round(bp['correct']/bp['total']*100, 1)}%)")

    summary_path = OUTPUT_DIR / "eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Rapport d'évaluation clinique - PosoLogic\n")
        f.write(f"Date: {report['timestamp']}\n")
        f.write(f"Modèle: {BASE_MODEL} + DPO (LoRA)\n")
        f.write(f"Cas évalués: {total}\n")
        f.write(f"Accuracy: {accuracy:.1f}%\n")
        f.write(f"Latence moyenne: {avg_latency:.2f}s\n")
        f.write(f"Matrice de confusion:\n")
        for key, count in sorted(confusion.items()):
            f.write(f"  {key}: {count}\n")
    print(f"   Résumé: {summary_path}")


if __name__ == "__main__":
    main()
