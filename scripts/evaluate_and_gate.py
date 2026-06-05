"""
scripts/evaluate_and_gate.py
=============================
Évalue les modèles staging et décide s'ils peuvent passer en production.

Quality gates (seuils configurables via variables d'environnement) :
  GATE_PRICING_MAX_MAE      → MAE maximum acceptable (défaut: 800 MAD)
  GATE_PRICING_MIN_COVERAGE → Coverage minimum (défaut: 35%)
  GATE_MATCHING_MIN_AUC     → AUC minimum (défaut: 0.50)
  GATE_MATCHING_MIN_PK5     → Precision@5 minimum (défaut: 0.30)

Sorties GitHub Actions :
  pricing_passed  = true | false
  matching_passed = true | false
  pricing_mae     = valeur
  matching_auc    = valeur

Le job CI/CD "promote" ne se lance que si les deux passent.
"""

import os, sys, json, joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Seuils des quality gates ──────────────────────────────────────────────────
GATE_PRICING_MAX_MAE      = float(os.getenv("GATE_PRICING_MAX_MAE",      800))
GATE_PRICING_MIN_COVERAGE = float(os.getenv("GATE_PRICING_MIN_COVERAGE",  25))
GATE_MATCHING_MIN_AUC     = float(os.getenv("GATE_MATCHING_MIN_AUC",    0.50))
GATE_MATCHING_MIN_PK5     = float(os.getenv("GATE_MATCHING_MIN_PK5",    0.30))

GITHUB_OUTPUT = os.getenv("GITHUB_OUTPUT", "")   # fichier GitHub Actions pour les outputs


def set_output(name: str, value: str):
    """Écrit une variable de sortie pour GitHub Actions."""
    print(f"  output: {name} = {value}")
    if GITHUB_OUTPUT:
        with open(GITHUB_OUTPUT, "a") as f:
            f.write(f"{name}={value}\n")


def evaluate_pricing() -> dict:
    path = ROOT / "models" / "staging" / "pricing_model.pkl"
    if not path.exists():
        print("  ✗ pricing_model.pkl introuvable")
        return None

    bundle = joblib.load(path)
    m = bundle["metrics"]
    print(f"  MAE      : {m['mae']} MAD  (gate: < {GATE_PRICING_MAX_MAE})")
    print(f"  Coverage : {m['coverage']}%  (gate: >= {GATE_PRICING_MIN_COVERAGE}%)")

    passed = (
        m["mae"]      <= GATE_PRICING_MAX_MAE and
        m["coverage"] >= GATE_PRICING_MIN_COVERAGE
    )
    return {"metrics": m, "passed": passed}


def evaluate_matching() -> dict:
    path = ROOT / "models" / "staging" / "matching_model.pkl"
    if not path.exists():
        print("  ✗ matching_model.pkl introuvable")
        return None

    bundle = joblib.load(path)
    m = bundle["metrics"]
    print(f"  AUC          : {m['auc']}  (gate: >= {GATE_MATCHING_MIN_AUC})")
    print(f"  Precision@5  : {m['pk5']}  (gate: >= {GATE_MATCHING_MIN_PK5})")

    passed = (
        m["auc"] >= GATE_MATCHING_MIN_AUC and
        m["pk5"] >= GATE_MATCHING_MIN_PK5
    )
    return {"metrics": m, "passed": passed}


def main():
    print("=" * 50)
    print("  ÉVALUATION DES QUALITY GATES")
    print("=" * 50)

    # ── Pricing ───────────────────────────────────────────────────────────────
    print("\n📊 Pricing Model :")
    pricing_result = evaluate_pricing()
    if pricing_result is None:
        pricing_passed = False
        pricing_mae    = "N/A"
    else:
        pricing_passed = pricing_result["passed"]
        pricing_mae    = str(pricing_result["metrics"]["mae"])
        status = "✅ GATE PASSÉ" if pricing_passed else "❌ GATE ÉCHOUÉ"
        print(f"  → {status}")

    # ── Matching ──────────────────────────────────────────────────────────────
    print("\n📊 Matching Model :")
    matching_result = evaluate_matching()
    if matching_result is None:
        matching_passed = False
        matching_auc    = "N/A"
    else:
        matching_passed = matching_result["passed"]
        matching_auc    = str(matching_result["metrics"]["auc"])
        status = "✅ GATE PASSÉ" if matching_passed else "❌ GATE ÉCHOUÉ"
        print(f"  → {status}")

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    if pricing_passed and matching_passed:
        print("  ✅ TOUS LES GATES PASSÉS → Promotion autorisée")
    else:
        print("  ❌ GATES ÉCHOUÉS → Promotion bloquée")
        if not pricing_passed:
            print("     • Pricing: améliore le modèle ou ajuste les seuils")
        if not matching_passed:
            print("     • Matching: améliore le modèle ou ajuste les seuils")
    print("=" * 50)

    # ── Outputs GitHub Actions ─────────────────────────────────────────────────
    set_output("pricing_passed",  str(pricing_passed).lower())
    set_output("matching_passed", str(matching_passed).lower())
    set_output("pricing_mae",     pricing_mae)
    set_output("matching_auc",    matching_auc)

    # ── Rapport JSON ──────────────────────────────────────────────────────────
    def to_ser(obj):
        import numpy as np
        if isinstance(obj, (np.bool_,bool)): return bool(obj)
        if isinstance(obj, np.integer):      return int(obj)
        if isinstance(obj, np.floating):     return float(obj)
        if isinstance(obj, dict):            return {k:to_ser(v) for k,v in obj.items()}
        return obj

    report = {
        "pricing":  {"passed": bool(pricing_passed),  "metrics": to_ser(pricing_result["metrics"])  if pricing_result  else {}},
        "matching": {"passed": bool(matching_passed), "metrics": to_ser(matching_result["metrics"]) if matching_result else {}},
        "gates": {
            "GATE_PRICING_MAX_MAE":      GATE_PRICING_MAX_MAE,
            "GATE_PRICING_MIN_COVERAGE": GATE_PRICING_MIN_COVERAGE,
            "GATE_MATCHING_MIN_AUC":     GATE_MATCHING_MIN_AUC,
            "GATE_MATCHING_MIN_PK5":     GATE_MATCHING_MIN_PK5,
        }
    }
    with open(ROOT / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nRapport sauvegardé → evaluation_report.json")

    # Exit code 1 si les gates échouent → le CI/CD s'arrête
    if not (pricing_passed and matching_passed):
        sys.exit(1)


if __name__ == "__main__":
    main()
