"""
tests/test_pipeline.py
=======================
Tests automatiques lancés par le CI/CD avant chaque entraînement.
Si un test échoue → le pipeline s'arrête → pas de nouveau modèle déployé.

Lance avec : pytest tests/ -v
"""

import os, sys, json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR   = ROOT / "data" / "raw"
MODEL_DIR  = ROOT / "models" / "staging"


# ════════════════════════════════════════════════════════════════════════════
# Tests des données
# ════════════════════════════════════════════════════════════════════════════

class TestData:

    def test_artisans_csv_existe(self):
        assert (DATA_DIR / "artisans.csv").exists(), \
            "artisans.csv manquant — lance python data/generate_data.py"

    def test_offers_csv_existe(self):
        assert (DATA_DIR / "accepted_offers.csv").exists(), \
            "accepted_offers.csv manquant"

    def test_interactions_csv_existe(self):
        assert (DATA_DIR / "interactions.csv").exists(), \
            "interactions.csv manquant"

    def test_artisans_colonnes(self):
        df = pd.read_csv(DATA_DIR / "artisans.csv")
        required = ["id", "category", "city", "latitude", "longitude",
                    "rating", "global_score", "years_experience",
                    "hourly_rate", "is_verified", "response_time_h"]
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Colonnes manquantes dans artisans.csv : {missing}"

    def test_artisans_pas_de_nulls_critiques(self):
        df = pd.read_csv(DATA_DIR / "artisans.csv")
        for col in ["category", "city", "rating"]:
            assert df[col].isnull().sum() == 0, f"Nulls dans artisans.csv colonne {col}"

    def test_offers_colonnes(self):
        df = pd.read_csv(DATA_DIR / "accepted_offers.csv")
        required = ["category", "city", "urgency", "accepted_price",
                    "artisan_rating", "budget_min", "budget_max"]
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Colonnes manquantes dans accepted_offers.csv : {missing}"

    def test_offers_prix_positifs(self):
        df = pd.read_csv(DATA_DIR / "accepted_offers.csv")
        assert (df["accepted_price"] > 0).all(), \
            "Des prix négatifs ou nuls dans accepted_offers.csv"

    def test_offers_volume_minimum(self):
        df = pd.read_csv(DATA_DIR / "accepted_offers.csv")
        assert len(df) >= 100, \
            f"Pas assez de données pour entraîner le pricing : {len(df)} < 100"

    def test_interactions_volume_minimum(self):
        df = pd.read_csv(DATA_DIR / "interactions.csv")
        assert len(df) >= 200, \
            f"Pas assez d'interactions : {len(df)} < 200"

    def test_interactions_contient_acceptations(self):
        df = pd.read_csv(DATA_DIR / "interactions.csv")
        n_accept = (df["type"] == "accept_offer").sum()
        assert n_accept >= 20, \
            f"Pas assez d'acceptations pour entraîner le matching : {n_accept} < 20"


# ════════════════════════════════════════════════════════════════════════════
# Tests du feature engineering
# ════════════════════════════════════════════════════════════════════════════

class TestFeatures:

    def test_build_pricing_features_shape(self):
        from api.features import build_pricing_features
        sample = {
            "category": "Plomberie", "city": "Casablanca", "urgency": "normal",
            "desc_len": 100, "n_photos": 2, "budget_min": 200, "budget_max": 1000,
            "artisan_rating": 4.0, "artisan_score": 0.7, "artisan_exp": 5,
            "distance_km": 10.0, "same_city": 1,
        }
        X = build_pricing_features(sample)
        assert X.shape[0] == 1, "build_pricing_features doit retourner 1 ligne"
        assert X.shape[1] == 16, f"Nombre de features incorrect : {X.shape[1]} != 16"

    def test_build_pricing_features_pas_de_nan(self):
        from api.features import build_pricing_features
        sample = {
            "category": "Électricité", "city": "Rabat", "urgency": "urgent",
            "desc_len": 80, "n_photos": 0, "budget_min": 300, "budget_max": 1200,
        }
        X = build_pricing_features(sample)
        assert not X.isnull().any().any(), "Des NaN dans les features pricing"

    def test_build_matching_features_shape(self):
        from api.features import build_matching_features
        artisan = {
            "rating": 4.2, "global_score": 0.8, "years_experience": 7,
            "hourly_rate": 180, "is_verified": 1, "response_time_h": 2.0,
            "response_rate": 0.9, "completion_rate": 0.95, "category": "Plomberie",
        }
        X = build_matching_features(artisan)
        assert X.shape[0] == 1
        assert X.shape[1] == 13, f"Nombre de features incorrect : {X.shape[1]} != 13"

    def test_haversine(self):
        from api.features import haversine
        # Distance Casablanca → Rabat ≈ 85 km
        dist = haversine(33.5731, -7.5898, 34.0209, -6.8416)
        assert 70 < dist < 110, f"Haversine incorrect : {dist} km (attendu ~85 km)"

    def test_categorie_inconnue_ne_plante_pas(self):
        """Une catégorie inconnue doit retourner -1 (Categorical codes) sans planter."""
        from api.features import build_pricing_features
        sample = {
            "category": "CatégorieInconnue", "city": "Casablanca", "urgency": "normal",
            "desc_len": 50, "n_photos": 0, "budget_min": 200, "budget_max": 1000,
        }
        X = build_pricing_features(sample)
        assert X.shape[0] == 1  # doit retourner quelque chose sans planter


# ════════════════════════════════════════════════════════════════════════════
# Tests des modèles (si déjà entraînés)
# ════════════════════════════════════════════════════════════════════════════

class TestModels:

    @pytest.fixture
    def pricing_bundle(self):
        path = MODEL_DIR / "pricing_model.pkl"
        if not path.exists():
            pytest.skip("pricing_model.pkl pas encore entraîné")
        import joblib
        return joblib.load(path)

    @pytest.fixture
    def matching_bundle(self):
        path = MODEL_DIR / "matching_model.pkl"
        if not path.exists():
            pytest.skip("matching_model.pkl pas encore entraîné")
        import joblib
        return joblib.load(path)

    def test_pricing_bundle_contient_les_cles(self, pricing_bundle):
        for key in ["model_q25", "model_q75", "feature_cols", "metrics"]:
            assert key in pricing_bundle, f"Clé manquante dans pricing bundle : {key}"

    def test_pricing_prediction_logique(self, pricing_bundle):
        from api.features import build_pricing_features
        sample = {
            "category": "Plomberie", "city": "Casablanca", "urgency": "normal",
            "desc_len": 100, "n_photos": 1, "budget_min": 200, "budget_max": 1000,
            "artisan_rating": 4.0, "artisan_score": 0.7, "artisan_exp": 5,
            "distance_km": 5.0, "same_city": 1,
        }
        X = build_pricing_features(sample)
        p_min = pricing_bundle["model_q25"].predict(X)[0]
        p_max = pricing_bundle["model_q75"].predict(X)[0]
        assert p_min >= 0,   f"p_min négatif : {p_min}"
        assert p_max >= p_min, f"p_max < p_min : {p_max} < {p_min}"

    def test_matching_bundle_contient_les_cles(self, matching_bundle):
        for key in ["model", "feature_cols", "metrics"]:
            assert key in matching_bundle, f"Clé manquante dans matching bundle : {key}"

    def test_matching_prediction_probabilite(self, matching_bundle):
        from api.features import build_matching_features
        artisan = {
            "rating": 4.5, "global_score": 0.85, "years_experience": 10,
            "hourly_rate": 200, "is_verified": 1, "response_time_h": 1.5,
            "response_rate": 0.95, "completion_rate": 0.98, "category": "Plomberie",
        }
        X = build_matching_features(artisan)
        prob = matching_bundle["model"].predict_proba(X)[0][1]
        assert 0.0 <= prob <= 1.0, f"Probabilité hors [0,1] : {prob}"

    def test_pricing_metriques_presentes(self, pricing_bundle):
        m = pricing_bundle["metrics"]
        for key in ["mae", "rmse", "coverage"]:
            assert key in m, f"Métrique manquante : {key}"
        assert m["mae"] > 0
        assert 0 <= m["coverage"] <= 100

    def test_matching_metriques_presentes(self, matching_bundle):
        m = matching_bundle["metrics"]
        for key in ["auc", "pk5"]:
            assert key in m, f"Métrique manquante : {key}"
        assert 0 <= m["auc"] <= 1
        assert 0 <= m["pk5"] <= 1
