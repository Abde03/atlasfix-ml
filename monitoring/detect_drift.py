"""
monitoring/detect_drift.py
===========================
Détecte si les nouvelles données de production sont différentes
des données d'entraînement (data drift).

Quand lancer ce script ?
  - Automatiquement chaque semaine (cron GitHub Actions)
  - Manuellement si tu suspectes un problème

Ce qu'il fait :
  1. Charge les données de référence (entraînement)
  2. Charge les nouvelles données de production (ou simule pour l'instant)
  3. Compare les distributions avec Evidently
  4. Génère un rapport HTML + un résumé JSON
  5. Si drift détecté → écrit un flag → CI/CD déclenche un retraining

Usage :
  python monitoring/detect_drift.py
  python monitoring/detect_drift.py --simulate-drift   ← pour tester
"""

import os, sys, json, argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR      = ROOT / "data" / "raw"
REPORTS_DIR   = ROOT / "monitoring" / "reports"
DRIFT_FLAG    = ROOT / "monitoring" / "drift_detected.flag"
HISTORY_FILE  = ROOT / "monitoring" / "drift_history.json"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Seuils
DRIFT_SHARE_THRESHOLD = 0.30   # si >30% des features driftent → alerte
FEATURE_DRIFT_PVALUE  = 0.05   # p-value pour le test statistique


def load_reference_data() -> pd.DataFrame:
    """Données d'entraînement = référence."""
    df = pd.read_csv(DATA_DIR / "accepted_offers.csv")
    return df[[
        "category", "city", "urgency", "desc_len",
        "n_photos", "budget_min", "budget_max",
        "artisan_rating", "artisan_score", "artisan_exp",
        "distance_km", "same_city", "accepted_price"
    ]]


def load_production_data(simulate_drift: bool = False) -> pd.DataFrame:
    """
    Charge les nouvelles données de production.

    En production réelle : Laravel écrit un fichier new_offers.csv
    à chaque nouvelle offre acceptée, ou tu l'exportes depuis PostgreSQL.

    Pour l'instant : on simule des données récentes.
    """
    prod_path = DATA_DIR / "new_offers.csv"

    if prod_path.exists():
        print(f"  Données réelles chargées depuis {prod_path}")
        return pd.read_csv(prod_path)

    # Simulation : génère des données récentes légèrement différentes
    print("  Simulation de données de production...")
    from faker import Faker
    fake = Faker("fr_FR")
    np.random.seed(int(datetime.now().timestamp()) % 1000)

    CATEGORIES = ["Plomberie","Électricité","Menuiserie","Maçonnerie",
                  "Peinture","Carrelage","Serrurerie","Jardinage",
                  "Nettoyage","Déménagement","Climatisation","Informatique"]
    CITIES     = ["Casablanca","Rabat","Fès","Marrakech","Agadir","Tanger","Meknès","Oujda"]
    URGENCIES  = ["normal","urgent","très urgent"]

    n = 200
    data = []
    for _ in range(n):
        cat = np.random.choice(CATEGORIES)
        city = np.random.choice(CITIES)

        # Si simulate_drift=True → les nouvelles données sont très différentes
        if simulate_drift:
            # Drift : les prix ont augmenté de 50%, les descriptions sont plus longues
            price_mult = 1.5
            desc_mult  = 2.5
            # Drift : nouvelle préférence pour certaines villes
            city = np.random.choice(["Casablanca", "Rabat"], p=[0.7, 0.3])
        else:
            price_mult = 1.0
            desc_mult  = 1.0

        base_prices = {
            "Plomberie":150,"Électricité":200,"Menuiserie":300,"Maçonnerie":400,
            "Peinture":200,"Carrelage":250,"Serrurerie":100,"Jardinage":150,
            "Nettoyage":100,"Déménagement":300,"Climatisation":500,"Informatique":100
        }
        bmax = {
            "Plomberie":900,"Électricité":1400,"Menuiserie":2200,"Maçonnerie":3500,
            "Peinture":1600,"Carrelage":2000,"Serrurerie":700,"Jardinage":900,
            "Nettoyage":600,"Déménagement":2200,"Climatisation":3200,"Informatique":900
        }
        p_base = base_prices.get(cat, 200)
        p_max  = bmax.get(cat, 1500)
        price  = int(np.random.randint(p_base, p_max) * price_mult)

        data.append({
            "category":      cat,
            "city":          city,
            "urgency":       np.random.choice(URGENCIES, p=[0.6, 0.3, 0.1]),
            "desc_len":      int(np.random.randint(30, 350) * desc_mult),
            "n_photos":      np.random.randint(0, 5),
            "budget_min":    int(p_base * 0.7),
            "budget_max":    int(p_max  * 1.3),
            "artisan_rating":round(min(5, max(1, np.random.normal(4.1, 0.6))), 1),
            "artisan_score": round(np.random.uniform(0.4, 1.0), 3),
            "artisan_exp":   np.random.randint(1, 25),
            "distance_km":   round(np.random.uniform(0, 50), 2),
            "same_city":     np.random.randint(0, 2),
            "accepted_price":price,
        })

    return pd.DataFrame(data)


def run_drift_detection(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Compare les distributions entre reference et current.
    Utilise des tests statistiques simples (KS test pour numérique,
    Chi2 pour catégoriel) — compatible avec toutes versions d'Evidently.
    """
    from scipy import stats

    results = {}
    drifted_features = []

    numeric_cols = ["desc_len", "n_photos", "budget_min", "budget_max",
                    "artisan_rating", "artisan_score", "artisan_exp",
                    "distance_km", "accepted_price"]

    categorical_cols = ["category", "city", "urgency", "same_city"]

    # Tests KS pour les features numériques
    for col in numeric_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values
        stat, pval = stats.ks_2samp(ref_vals, cur_vals)
        drifted = bool(pval < FEATURE_DRIFT_PVALUE)
        if drifted:
            drifted_features.append(col)
        results[col] = {
            "type":    "numeric",
            "test":    "KS",
            "statistic": round(float(stat), 4),
            "p_value":   round(float(pval), 4),
            "drifted":   drifted,
            "ref_mean":  round(float(ref_vals.mean()), 2),
            "cur_mean":  round(float(cur_vals.mean()), 2),
        }

    # Tests Chi2 pour les features catégorielles
    for col in categorical_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        ref_counts = reference[col].value_counts(normalize=True)
        cur_counts = current[col].value_counts(normalize=True)
        all_cats   = set(ref_counts.index) | set(cur_counts.index)
        ref_freq   = np.array([ref_counts.get(c, 0) for c in all_cats])
        cur_freq   = np.array([cur_counts.get(c, 0) for c in all_cats])
        # Normaliser pour Chi2
        if ref_freq.sum() > 0 and cur_freq.sum() > 0:
            expected = ref_freq * len(current)
            observed = cur_freq * len(current)
            # Éviter les zéros
            mask = expected > 0
            if mask.sum() >= 2:
                stat, pval = stats.chisquare(
                    observed[mask] + 0.5,
                    f_exp=expected[mask] + 0.5
                )
                drifted = bool(pval < FEATURE_DRIFT_PVALUE)
                if drifted:
                    drifted_features.append(col)
                results[col] = {
                    "type":    "categorical",
                    "test":    "Chi2",
                    "statistic": round(float(stat), 4),
                    "p_value":   round(float(pval), 4),
                    "drifted":   drifted,
                }

    drift_share = len(drifted_features) / max(len(results), 1)
    dataset_drift = drift_share > DRIFT_SHARE_THRESHOLD

    return {
        "dataset_drift":    dataset_drift,
        "drift_share":      round(drift_share, 3),
        "drifted_features": drifted_features,
        "n_features_tested":len(results),
        "feature_results":  results,
    }


def generate_html_report(reference, current, drift_summary: dict, path: Path):
    """Génère un rapport HTML lisible."""
    drifted = drift_summary["drifted_features"]
    status  = "🔴 DRIFT DÉTECTÉ" if drift_summary["dataset_drift"] else "🟢 STABLE"
    color   = "#e74c3c" if drift_summary["dataset_drift"] else "#27ae60"

    rows = ""
    for feat, info in drift_summary["feature_results"].items():
        d_color = "#e74c3c" if info["drifted"] else "#27ae60"
        d_text  = "OUI" if info["drifted"] else "non"
        extra   = f"mean: {info.get('ref_mean','?')} → {info.get('cur_mean','?')}" if info["type"]=="numeric" else ""
        rows += f"""
        <tr>
            <td>{feat}</td>
            <td>{info['type']}</td>
            <td>{info['test']}</td>
            <td>{info['p_value']}</td>
            <td style='color:{d_color};font-weight:bold'>{d_text}</td>
            <td style='color:#666'>{extra}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset='UTF-8'>
<title>AtlasFix — Drift Report</title>
<style>
  body {{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 20px}}
  h1   {{color:#2c3e50}}
  .status {{font-size:24px;font-weight:bold;color:{color};padding:15px;
            background:#f8f9fa;border-radius:8px;margin:20px 0}}
  table {{width:100%;border-collapse:collapse;margin-top:20px}}
  th    {{background:#2c3e50;color:white;padding:10px;text-align:left}}
  td    {{padding:8px;border-bottom:1px solid #ddd}}
  tr:hover {{background:#f5f5f5}}
  .summary {{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:20px 0}}
  .card    {{background:#f8f9fa;padding:15px;border-radius:8px;text-align:center}}
  .card h3 {{margin:0;font-size:28px;color:{color}}}
  .card p  {{margin:5px 0;color:#666}}
</style>
</head><body>
<h1>AtlasFix — Rapport de Drift</h1>
<p>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC</p>
<div class='status'>{status}</div>

<div class='summary'>
  <div class='card'>
    <h3>{drift_summary['drift_share']*100:.0f}%</h3>
    <p>Features en drift</p>
  </div>
  <div class='card'>
    <h3>{len(drifted)}</h3>
    <p>Features driftées</p>
  </div>
  <div class='card'>
    <h3>{len(reference)}/{len(current)}</h3>
    <p>Ref / Production</p>
  </div>
</div>

{"<p style='color:#e74c3c'>⚠️  Features driftées : " + ", ".join(drifted) + "</p>" if drifted else ""}

<h2>Détail par feature</h2>
<table>
  <tr><th>Feature</th><th>Type</th><th>Test</th><th>P-value</th><th>Drift ?</th><th>Évolution</th></tr>
  {rows}
</table>

<h2>Recommandation</h2>
{'<p>🔴 Le drift dépasse le seuil. Un <strong>retraining</strong> est recommandé.</p>' if drift_summary["dataset_drift"] else '<p>🟢 Les données de production restent similaires aux données d\'entraînement. Aucune action nécessaire.</p>'}
</body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def update_history(drift_summary: dict):
    """Ajoute le résultat au fichier d'historique JSON."""
    history = []
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            history = json.load(f)

    history.append({
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "dataset_drift": drift_summary["dataset_drift"],
        "drift_share":   drift_summary["drift_share"],
        "drifted":       drift_summary["drifted_features"],
    })

    # Garder les 52 dernières semaines
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-52:], f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-drift", action="store_true",
                        help="Simule un drift fort pour tester le système")
    args = parser.parse_args()

    print("=" * 55)
    print("  ATLASFIX — DÉTECTION DE DRIFT")
    print("=" * 55)

    # 1. Charger les données
    print("\n① Chargement des données...")
    reference = load_reference_data()
    current   = load_production_data(simulate_drift=args.simulate_drift)
    print(f"  Référence  : {len(reference)} lignes")
    print(f"  Production : {len(current)} lignes")
    if args.simulate_drift:
        print("  ⚠️  Mode simulation drift activé")

    # 2. Détecter le drift
    print("\n② Détection du drift...")
    drift_summary = run_drift_detection(reference, current)

    print(f"  Features testées : {drift_summary['n_features_tested']}")
    print(f"  Features driftées : {len(drift_summary['drifted_features'])}")
    print(f"  Drift share : {drift_summary['drift_share']*100:.1f}% (seuil: {DRIFT_SHARE_THRESHOLD*100:.0f}%)")

    if drift_summary["drifted_features"]:
        print(f"  Features concernées : {', '.join(drift_summary['drifted_features'])}")

    # 3. Rapport HTML
    print("\n③ Génération du rapport...")
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
    generate_html_report(reference, current, drift_summary, report_path)
    print(f"  Rapport → {report_path}")

    # 4. Historique
    update_history(drift_summary)
    print(f"  Historique → {HISTORY_FILE}")

    # 5. Flag pour CI/CD
    if drift_summary["dataset_drift"]:
        DRIFT_FLAG.write_text(json.dumps({
            "detected_at":  datetime.now(timezone.utc).isoformat(),
            "drift_share":  drift_summary["drift_share"],
            "drifted":      drift_summary["drifted_features"],
        }))
        print(f"\n  🔴 FLAG écrit → {DRIFT_FLAG}")
        print("     Le CI/CD va déclencher un retraining.")
    else:
        if DRIFT_FLAG.exists():
            DRIFT_FLAG.unlink()
        print("\n  🟢 Pas de drift significatif. Aucune action requise.")

    # 6. Résumé final
    print("\n" + "=" * 55)
    status = "🔴 DRIFT DÉTECTÉ — RETRAINING RECOMMANDÉ" if drift_summary["dataset_drift"] else "🟢 STABLE"
    print(f"  {status}")
    print("=" * 55)


if __name__ == "__main__":
    main()
