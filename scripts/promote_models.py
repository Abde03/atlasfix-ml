"""
scripts/promote_models.py
==========================
Copie les modèles de staging/ vers production/.
Crée un fichier model_registry.json pour tracer l'historique.

Appelé par le CI/CD APRÈS que les quality gates ont passé.
"""

import os, sys, json, shutil, joblib
from pathlib import Path
from datetime import datetime, timezone

ROOT        = Path(__file__).resolve().parents[1]
STAGING_DIR = ROOT / "models" / "staging"
PROD_DIR    = ROOT / "models" / "production"
REGISTRY    = PROD_DIR / "model_registry.json"

PROD_DIR.mkdir(parents=True, exist_ok=True)


def load_registry() -> list:
    if REGISTRY.exists():
        with open(REGISTRY) as f:
            return json.load(f)
    return []


def save_registry(history: list):
    with open(REGISTRY, "w") as f:
        json.dump(history, f, indent=2)


def promote():
    print("=" * 50)
    print("  PROMOTION staging → production")
    print("=" * 50)

    history  = load_registry()
    promoted = []
    now      = datetime.now(timezone.utc).isoformat()

    for model_file in ["pricing_model.pkl", "matching_model.pkl"]:
        src = STAGING_DIR / model_file
        dst = PROD_DIR    / model_file

        if not src.exists():
            print(f"  ✗ {model_file} introuvable dans staging/")
            continue

        # Charger les métriques avant de copier
        bundle  = joblib.load(src)
        metrics = bundle.get("metrics", {})
        run_id  = bundle.get("run_id", "unknown")

        # Sauvegarder l'ancien modèle en backup (garde 3 versions)
        if dst.exists():
            backup = PROD_DIR / f"{model_file}.bak"
            shutil.copy2(dst, backup)

        shutil.copy2(src, dst)
        print(f"  ✓ {model_file} promu en production")
        print(f"    métriques : {metrics}")

        promoted.append({
            "model":      model_file,
            "promoted_at": now,
            "run_id":     run_id,
            "metrics":    metrics,
        })

    # Mettre à jour le registre
    history.append({
        "timestamp": now,
        "models":    promoted,
        "git_sha":   os.getenv("GITHUB_SHA", "local"),
        "triggered_by": os.getenv("GITHUB_ACTOR", "manual"),
    })
    # Garder uniquement les 20 dernières promotions
    save_registry(history[-20:])

    print()
    print(f"  Registre mis à jour → {REGISTRY}")
    print(f"  Historique : {len(history)} promotions enregistrées")
    print("=" * 50)
    print("  ✅ Promotion terminée")


if __name__ == "__main__":
    promote()
