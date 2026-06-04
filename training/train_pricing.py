"""
training/train_pricing.py
==========================
Script standalone d'entraînement du modèle Pricing.
Appelé par le CI/CD GitHub Actions (et lancable à la main).

Usage :
  python training/train_pricing.py

Produit :
  models/staging/pricing_model.pkl   ← bundle complet
"""

import os, sys, json, joblib, logging
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ajouter la racine au path pour importer api.features
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.features import CATEGORIES, CITIES, URGENCY_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "accepted_offers.csv"
MODEL_DIR = ROOT / "models" / "staging"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "mlflow_store").mkdir(exist_ok=True)
MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(ROOT / "mlflow_store" / "mlflow.db").replace("\\", "/"),
)


def train():
    # ── Données ───────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    log.info(f"Données : {len(df)} lignes")

    X = _df_to_features(df)
    y = df["accepted_price"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log.info(f"Train={len(X_train)} | Test={len(X_test)}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("atlasfix-pricing")

    params = dict(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        num_leaves=20, min_child_samples=15,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1,
    )

    with mlflow.start_run(run_name="train-pricing") as run:
        mlflow.log_params(params)

        # Entraîner Q25 et Q75
        log.info("Entraînement Q25...")
        m25 = lgb.LGBMRegressor(objective="quantile", alpha=0.25, **params)
        m25.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

        log.info("Entraînement Q75...")
        m75 = lgb.LGBMRegressor(objective="quantile", alpha=0.75, **params)
        m75.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

        # Métriques
        p25, p75 = m25.predict(X_test), m75.predict(X_test)
        mid      = (p25 + p75) / 2
        mae      = float(mean_absolute_error(y_test, mid))
        rmse     = float(np.sqrt(mean_squared_error(y_test, mid)))
        coverage = float(np.mean((y_test >= p25) & (y_test <= p75))) * 100
        width    = float(np.mean(p75 - p25))

        mlflow.log_metric("mae",      round(mae, 2))
        mlflow.log_metric("rmse",     round(rmse, 2))
        mlflow.log_metric("coverage", round(coverage, 2))
        mlflow.log_metric("width",    round(width, 2))

        log.info(f"MAE={mae:.1f} | RMSE={rmse:.1f} | Coverage={coverage:.1f}% | Width={width:.0f}")

        # Sauvegarder
        bundle = {
            "model_q25":    m25,
            "model_q75":    m75,
            "feature_cols": list(X.columns),
            "categories":   CATEGORIES,
            "cities":       CITIES,
            "urgency_map":  URGENCY_MAP,
            "metrics": {"mae": round(mae,2), "rmse": round(rmse,2),
                        "coverage": round(coverage,2), "width": round(width,2)},
            "run_id": run.info.run_id,
        }
        path = MODEL_DIR / "pricing_model.pkl"
        joblib.dump(bundle, path)
        mlflow.log_artifact(str(path))
        log.info(f"Modèle sauvegardé → {path}")

    return bundle["metrics"]


def _df_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adapte le DataFrame pour build_pricing_features."""
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
    return X


if __name__ == "__main__":
    metrics = train()
    print(json.dumps(metrics, indent=2))
