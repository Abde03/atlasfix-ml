"""
monitoring/track_performance.py
================================
Mesure la performance réelle des modèles sur les nouvelles données
et détecte si les métriques se dégradent dans le temps.

À lancer :
  - Chaque semaine (cron GitHub Actions)
  - Après chaque promotion de modèle

Ce qu'il fait :
  1. Évalue les modèles en production sur les nouvelles données
  2. Compare avec les métriques du run d'entraînement
  3. Sauvegarde dans un historique JSON
  4. Si dégradation détectée → écrit un flag → retraining déclenché
  5. Génère un graphe de tendance

Usage :
  python monitoring/track_performance.py
"""

import os, sys, json, joblib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.features import CATEGORIES, URGENCY_MAP, CITIES

PROD_DIR          = ROOT / "models" / "production"
DATA_DIR          = ROOT / "data" / "raw"
REPORTS_DIR       = ROOT / "monitoring" / "reports"
PERF_HISTORY_FILE = ROOT / "monitoring" / "performance_history.json"
DEGRADE_FLAG      = ROOT / "monitoring" / "degradation_detected.flag"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Seuils de dégradation acceptable (relatif aux métriques d'entraînement)
MAX_MAE_INCREASE_PCT    = 30   # +30% de MAE → alerte
MIN_AUC_DECREASE        = 0.08 # -0.08 d'AUC → alerte
MIN_COVERAGE_DECREASE   = 10   # -10% de coverage → alerte


def load_production_models() -> dict:
    pricing  = joblib.load(PROD_DIR / "pricing_model.pkl")
    matching = joblib.load(PROD_DIR / "matching_model.pkl")
    return {"pricing": pricing, "matching": matching}


def evaluate_pricing(bundle: dict, df: pd.DataFrame) -> dict:
    """Évalue le modèle pricing sur de nouvelles données."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from api.features import CATEGORIES, CITIES, URGENCY_MAP

    X = pd.DataFrame()
    X["category_enc"]     = pd.Categorical(df["category"], categories=CATEGORIES).codes
    X["city_enc"]         = pd.Categorical(df["city"],     categories=CITIES).codes
    X["urgency_enc"]      = df["urgency"].map(URGENCY_MAP).fillna(0).astype(int)
    X["budget_min"]       = df["budget_min"].fillna(200)
    X["budget_max"]       = df["budget_max"].fillna(1500)
    X["desc_len"]         = df["desc_len"].fillna(80)
    X["n_photos"]         = df["n_photos"].fillna(0)
    X["artisan_rating"]   = df["artisan_rating"].fillna(4.0)
    X["artisan_score"]    = df["artisan_score"].fillna(0.7)
    X["artisan_exp"]      = df["artisan_exp"].fillna(5)
    X["distance_km"]      = df["distance_km"].fillna(15)
    X["same_city"]        = df["same_city"].fillna(1).astype(int)
    X["budget_range"]     = X["budget_max"] - X["budget_min"]
    X["urgency_x_budget"] = X["urgency_enc"] * X["budget_max"]
    X["rating_x_exp"]     = X["artisan_rating"] * X["artisan_exp"]
    X["photos_x_desc"]    = X["n_photos"] * X["desc_len"]

    y = df["accepted_price"].values
    p25 = bundle["model_q25"].predict(X)
    p75 = bundle["model_q75"].predict(X)
    mid = (p25 + p75) / 2

    mae      = float(mean_absolute_error(y, mid))
    rmse     = float(np.sqrt(mean_squared_error(y, mid)))
    coverage = float(np.mean((y >= p25) & (y <= p75))) * 100

    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "coverage": round(coverage, 2)}


def evaluate_matching(bundle: dict, df_interactions: pd.DataFrame,
                      df_artisans: pd.DataFrame) -> dict:
    """Évalue le modèle matching sur de nouvelles interactions."""
    from sklearn.metrics import roc_auc_score

    artisan_feats = df_artisans[[
        "id","category","rating","global_score","years_experience",
        "hourly_rate","is_verified","response_time_h","response_rate","completion_rate"
    ]].rename(columns={"id": "artisan_id"})

    df = df_interactions.merge(artisan_feats, on="artisan_id", how="left")
    df["accepted"] = (df["type"] == "accept_offer").astype(int)

    from api.features import CATEGORIES
    X = pd.DataFrame(index=df.index)
    X["artisan_rating"]     = df["rating"].fillna(3.0)
    X["artisan_score"]      = df["global_score"].fillna(0.5)
    X["years_exp"]          = df["years_experience"].fillna(5)
    X["hourly_rate"]        = df["hourly_rate"].fillna(150)
    X["is_verified"]        = df["is_verified"].fillna(0).astype(int)
    X["response_time_h"]    = df["response_time_h"].fillna(12)
    X["response_rate"]      = df["response_rate"].fillna(0.7)
    X["completion_rate"]    = df["completion_rate"].fillna(0.8)
    cat_col = "category_x" if "category_x" in df.columns else "category"
    X["category_enc"]       = pd.Categorical(df[cat_col], categories=CATEGORIES).codes
    X["rating_x_verified"]  = X["artisan_rating"] * X["is_verified"]
    X["score_x_completion"] = X["artisan_score"]  * X["completion_rate"]
    X["exp_normalized"]     = (X["years_exp"] / 25.0).clip(0, 1)
    X["resp_speed_score"]   = (1.0 - (X["response_time_h"] / 24).clip(0, 1))

    y     = df["accepted"].values
    y_prob = bundle["model"].predict_proba(X)[:, 1]

    if len(np.unique(y)) < 2:
        return {"auc": None, "note": "Pas assez de classes"}

    auc = float(roc_auc_score(y, y_prob))
    top5_idx = np.argsort(y_prob)[::-1][:5]
    pk5 = float(np.mean(y[top5_idx]))

    return {"auc": round(auc, 4), "pk5": round(pk5, 4)}


def check_degradation(train_metrics: dict, current_metrics: dict,
                      model_name: str) -> list:
    """Retourne la liste des dégradations détectées."""
    issues = []

    if model_name == "pricing":
        if train_metrics.get("mae") and current_metrics.get("mae"):
            increase = (current_metrics["mae"] - train_metrics["mae"]) / max(train_metrics["mae"], 1) * 100
            if increase > MAX_MAE_INCREASE_PCT:
                issues.append(f"MAE augmentée de {increase:.1f}% ({train_metrics['mae']} → {current_metrics['mae']})")
        if train_metrics.get("coverage") and current_metrics.get("coverage"):
            decrease = train_metrics["coverage"] - current_metrics["coverage"]
            if decrease > MIN_COVERAGE_DECREASE:
                issues.append(f"Coverage baissée de {decrease:.1f}% ({train_metrics['coverage']}% → {current_metrics['coverage']}%)")

    if model_name == "matching":
        if train_metrics.get("auc") and current_metrics.get("auc"):
            decrease = train_metrics["auc"] - current_metrics["auc"]
            if decrease > MIN_AUC_DECREASE:
                issues.append(f"AUC baissée de {decrease:.3f} ({train_metrics['auc']} → {current_metrics['auc']})")

    return issues


def plot_trend(history: list, output_path: Path):
    """Génère un graphe de tendance des métriques dans le temps."""
    if len(history) < 2:
        return

    dates    = [h["timestamp"][:10] for h in history]
    mae_vals = [h.get("pricing_mae") for h in history]
    auc_vals = [h.get("matching_auc") for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("AtlasFix — Tendance des métriques en production", fontsize=13, fontweight="bold")

    # MAE Pricing
    valid_mae = [(d, v) for d, v in zip(dates, mae_vals) if v is not None]
    if valid_mae:
        x, y = zip(*valid_mae)
        axes[0].plot(x, y, "o-", color="coral", linewidth=2, markersize=6)
        axes[0].axhline(y=800, color="red", linestyle="--", alpha=0.5, label="Gate (800 MAD)")
        axes[0].set_title("Pricing — MAE (↓ meilleur)")
        axes[0].set_ylabel("MAD")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].legend()
        axes[0].fill_between(range(len(x)), y, alpha=0.1, color="coral")

    # AUC Matching
    valid_auc = [(d, v) for d, v in zip(dates, auc_vals) if v is not None]
    if valid_auc:
        x, y = zip(*valid_auc)
        axes[1].plot(x, y, "o-", color="steelblue", linewidth=2, markersize=6)
        axes[1].axhline(y=0.50, color="red", linestyle="--", alpha=0.5, label="Gate (0.50)")
        axes[1].set_title("Matching — AUC (↑ meilleur)")
        axes[1].set_ylabel("AUC")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].legend()
        axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Graphe tendance → {output_path}")


def main():
    print("=" * 55)
    print("  ATLASFIX — SUIVI DE PERFORMANCE")
    print("=" * 55)

    # 1. Charger les modèles de production
    print("\n① Chargement des modèles de production...")
    try:
        models = load_production_models()
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        print("  Lance d'abord le pipeline CI/CD pour promouvoir des modèles.")
        sys.exit(1)

    pricing_train_metrics  = models["pricing"]["metrics"]
    matching_train_metrics = models["matching"]["metrics"]
    print(f"  Pricing  métriques entraînement : {pricing_train_metrics}")
    print(f"  Matching métriques entraînement : {matching_train_metrics}")

    # 2. Charger nouvelles données
    print("\n② Chargement des nouvelles données...")
    offers       = pd.read_csv(DATA_DIR / "accepted_offers.csv")
    interactions = pd.read_csv(DATA_DIR / "interactions.csv")
    artisans     = pd.read_csv(DATA_DIR / "artisans.csv")

    # En production : charge les données récentes (ex: 30 derniers jours)
    # Pour l'instant on utilise 20% des données comme proxy "nouvelles données"
    offers_eval       = offers.sample(frac=0.2, random_state=42)
    interactions_eval = interactions.sample(frac=0.2, random_state=42)
    print(f"  Données évaluation : {len(offers_eval)} offres, {len(interactions_eval)} interactions")

    # 3. Évaluer
    print("\n③ Évaluation des modèles en production...")
    current_pricing  = evaluate_pricing(models["pricing"], offers_eval)
    current_matching = evaluate_matching(models["matching"], interactions_eval, artisans)
    print(f"  Pricing  actuel : {current_pricing}")
    print(f"  Matching actuel : {current_matching}")

    # 4. Détecter dégradation
    print("\n④ Vérification de la dégradation...")
    all_issues = []
    issues_p = check_degradation(pricing_train_metrics,  current_pricing,  "pricing")
    issues_m = check_degradation(matching_train_metrics, current_matching, "matching")
    all_issues = issues_p + issues_m

    if all_issues:
        for issue in all_issues:
            print(f"  ⚠️  {issue}")
        DEGRADE_FLAG.write_text(json.dumps({
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "issues":      all_issues,
        }))
        print(f"\n  🔴 FLAG dégradation → {DEGRADE_FLAG}")
    else:
        print("  ✅ Pas de dégradation détectée")
        if DEGRADE_FLAG.exists():
            DEGRADE_FLAG.unlink()

    # 5. Historique
    history = []
    if PERF_HISTORY_FILE.exists():
        with open(PERF_HISTORY_FILE) as f:
            history = json.load(f)

    history.append({
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "pricing_mae":   current_pricing.get("mae"),
        "pricing_coverage": current_pricing.get("coverage"),
        "matching_auc":  current_matching.get("auc"),
        "matching_pk5":  current_matching.get("pk5"),
        "degraded":      len(all_issues) > 0,
        "issues":        all_issues,
    })
    with open(PERF_HISTORY_FILE, "w") as f:
        json.dump(history[-52:], f, indent=2)

    # 6. Graphe tendance
    print("\n⑤ Génération du graphe de tendance...")
    plot_trend(history, REPORTS_DIR / "performance_trend.png")

    # 7. Résumé
    print("\n" + "=" * 55)
    print(f"  Pricing  MAE     : {current_pricing.get('mae')} MAD")
    print(f"  Pricing  Coverage: {current_pricing.get('coverage')}%")
    print(f"  Matching AUC     : {current_matching.get('auc')}")
    print(f"  Matching P@5     : {current_matching.get('pk5')}")
    if all_issues:
        print(f"\n  🔴 {len(all_issues)} problème(s) détecté(s) → retraining recommandé")
    else:
        print(f"\n  🟢 Modèles stables en production")
    print("=" * 55)


if __name__ == "__main__":
    main()
