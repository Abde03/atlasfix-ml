"""
training/train_matching.py
===========================
Script standalone d'entraînement du modèle Matching.
Usage : python training/train_matching.py
"""

import os, sys, json, joblib, logging
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.features import CATEGORIES, URGENCY_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models" / "staging"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "mlflow_store").mkdir(exist_ok=True)
MLFLOW_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(ROOT / "mlflow_store" / "mlflow.db").replace("\\", "/"),
)


def precision_at_k(y_true, y_scores, k=5):
    idx = np.argsort(y_scores)[::-1][:k]
    return round(float(np.mean(y_true[idx])), 4)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
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
    return X


def train():
    artisans     = pd.read_csv(DATA_DIR / "artisans.csv")
    interactions = pd.read_csv(DATA_DIR / "interactions.csv")

    artisan_feats = artisans[[
        "id", "category", "city", "latitude", "longitude",
        "rating", "global_score", "years_experience", "hourly_rate",
        "is_verified", "response_time_h", "response_rate", "completion_rate"
    ]].rename(columns={"id": "artisan_id"})

    df = interactions.merge(artisan_feats, on="artisan_id", how="left")
    df["accepted"] = (df["type"] == "accept_offer").astype(int)

    log.info(f"Dataset : {len(df)} lignes | pos_rate={df['accepted'].mean()*100:.1f}%")

    X = build_features(df)
    y = df["accepted"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    spw   = int(n_neg / max(n_pos, 1))

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("atlasfix-matching")

    params = dict(
        objective="binary", n_estimators=500, learning_rate=0.05,
        max_depth=5, num_leaves=20, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, reg_alpha=0.1, reg_lambda=0.2,
        random_state=42, verbose=-1,
    )

    with mlflow.start_run(run_name="train-matching") as run:
        mlflow.log_params({**params, "n_train": len(X_train), "pos_rate": round(y.mean(), 3)})

        log.info("Entraînement classifieur...")
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric="auc",
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        log.info("Calibration...")
        calibrated = CalibratedClassifierCV(model, cv=3, method="sigmoid")
        calibrated.fit(X_train, y_train)

        y_prob = calibrated.predict_proba(X_test)[:, 1]
        auc    = round(roc_auc_score(y_test, y_prob), 4)
        ap     = round(average_precision_score(y_test, y_prob), 4)
        pk5    = precision_at_k(y_test, y_prob, k=5)
        pk10   = precision_at_k(y_test, y_prob, k=10)

        mlflow.log_metric("auc",             auc)
        mlflow.log_metric("avg_precision",   ap)
        mlflow.log_metric("precision_at_5",  pk5)
        mlflow.log_metric("precision_at_10", pk10)

        log.info(f"AUC={auc} | AP={ap} | P@5={pk5} | P@10={pk10}")

        bundle = {
            "model":         calibrated,
            "feature_cols":  list(X.columns),
            "categories":    CATEGORIES,
            "urgency_map":   URGENCY_MAP,
            "metrics":       {"auc": auc, "ap": ap, "pk5": pk5, "pk10": pk10},
            "run_id":        run.info.run_id,
        }
        path = MODEL_DIR / "matching_model.pkl"
        joblib.dump(bundle, path)
        mlflow.log_artifact(str(path))
        log.info(f"Modèle sauvegardé → {path}")

    return bundle["metrics"]


if __name__ == "__main__":
    metrics = train()
    print(json.dumps(metrics, indent=2))
