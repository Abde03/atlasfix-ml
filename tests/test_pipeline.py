"""
tests/test_pipeline.py
=======================
Tests automatiques lancés par le CI/CD.
Lance avec : pytest tests/ -v
"""

import os, sys, json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR  = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models" / "staging"


# ════════════════════════════════════════════════════════════════════════════
# Tests des données
# ════════════════════════════════════════════════════════════════════════════

class TestData:

    def test_artisans_csv_existe(self):
        assert (DATA_DIR / "artisans.csv").exists()

    def test_offers_csv_existe(self):
        assert (DATA_DIR / "accepted_offers.csv").exists()

    def test_interactions_csv_existe(self):
        assert (DATA_DIR / "interactions.csv").exists()

    def test_artisans_colonnes(self):
        df = pd.read_csv(DATA_DIR / "artisans.csv")
        required = ["id", "category", "city", "latitude", "longitude",
                    "rating", "global_score", "years_experience",
                    "hourly_rate", "is_verified", "response_time_h",
                    "subscription"]   # ← NOUVEAU
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Colonnes manquantes dans artisans.csv : {missing}"

    def test_artisans_subscription_valeurs(self):
        """subscription doit contenir uniquement free/basic/premium/elite."""
        df = pd.read_csv(DATA_DIR / "artisans.csv")
        valid = {"free", "basic", "premium", "elite"}
        invalid = set(df["subscription"].unique()) - valid
        assert not invalid, f"Valeurs subscription invalides : {invalid}"

    def test_artisans_subscription_distribution(self):
        """free doit être le tier majoritaire."""
        df = pd.read_csv(DATA_DIR / "artisans.csv")
        counts = df["subscription"].value_counts(normalize=True)
        assert counts.get("free", 0) > 0.3, "free doit représenter >30% des artisans"

    def test_subscription_bias_dans_interactions(self):
        """
        Les artisans elite doivent avoir un taux d'acceptation plus élevé que free.
        Valide que le bias est correctement injecté dans les données.
        """
        # interactions.csv contient deja la colonne subscription (ajoutee dans generate_data.py)
        df = pd.read_csv(DATA_DIR / "interactions.csv")
        df["accepted"] = (df["type"] == "accept_offer").astype(int)

        rates = df.groupby("subscription")["accepted"].mean()
        if "elite" in rates and "free" in rates:
            assert rates["elite"] > rates["free"], \
                f"elite ({rates['elite']:.3f}) doit > free ({rates['free']:.3f})"

    def test_artisans_pas_de_nulls_critiques(self):
        df = pd.read_csv(DATA_DIR / "artisans.csv")
        for col in ["category", "city", "rating", "subscription"]:
            assert df[col].isnull().sum() == 0, f"Nulls dans artisans.csv colonne {col}"

    def test_offers_colonnes(self):
        df = pd.read_csv(DATA_DIR / "accepted_offers.csv")
        required = ["category", "city", "urgency", "accepted_price",
                    "artisan_rating", "budget_min", "budget_max"]
        missing = [c for c in required if c not in df.columns]
        assert not missing, f"Colonnes manquantes : {missing}"

    def test_offers_prix_positifs(self):
        df = pd.read_csv(DATA_DIR / "accepted_offers.csv")
        assert (df["accepted_price"] > 0).all()

    def test_offers_volume_minimum(self):
        df = pd.read_csv(DATA_DIR / "accepted_offers.csv")
        assert len(df) >= 100, f"Pas assez de données : {len(df)} < 100"

    def test_interactions_volume_minimum(self):
        df = pd.read_csv(DATA_DIR / "interactions.csv")
        assert len(df) >= 200

    def test_interactions_contient_acceptations(self):
        df = pd.read_csv(DATA_DIR / "interactions.csv")
        assert (df["type"] == "accept_offer").sum() >= 20


# ════════════════════════════════════════════════════════════════════════════
# Tests du feature engineering
# ════════════════════════════════════════════════════════════════════════════

class TestFeatures:

    def test_build_pricing_features_shape(self):
        from api.features import build_pricing_features
        sample = {
            "category": "Plomberie", "city": "Casablanca", "urgency": "normal",
            "desc_len": 100, "n_photos": 2, "budget_min": 200, "budget_max": 1000,
        }
        X = build_pricing_features(sample)
        assert X.shape[0] == 1
        assert X.shape[1] == 16

    def test_build_pricing_features_pas_de_nan(self):
        from api.features import build_pricing_features
        sample = {
            "category": "Électricité", "city": "Rabat", "urgency": "urgent",
            "desc_len": 80, "n_photos": 0, "budget_min": 300, "budget_max": 1200,
        }
        X = build_pricing_features(sample)
        assert not X.isnull().any().any()

    def test_build_matching_features_shape(self):
        from api.features import build_matching_features
        artisan = {
            "rating": 4.2, "global_score": 0.8, "years_experience": 7,
            "hourly_rate": 180, "is_verified": 1, "response_time_h": 2.0,
            "response_rate": 0.9, "completion_rate": 0.95, "category": "Plomberie",
            "subscription": "premium",   # ← NOUVEAU
        }
        X = build_matching_features(artisan)
        assert X.shape[0] == 1
        assert X.shape[1] == 16, f"Attendu 15 features, obtenu {X.shape[1]}"

    def test_matching_features_contient_subscription(self):
        """subscription_enc et ses dérivées doivent être dans les features."""
        from api.features import build_matching_features
        artisan = {
            "rating": 4.0, "global_score": 0.7, "years_experience": 5,
            "hourly_rate": 150, "is_verified": 1, "response_time_h": 3.0,
            "response_rate": 0.8, "completion_rate": 0.9, "category": "Peinture",
            "subscription": "basic",
        }
        X = build_matching_features(artisan)
        assert "subscription_enc" in X.columns
        assert "sub_x_score"      in X.columns
        assert "sub_x_verified"   in X.columns

    def test_subscription_enc_valeurs(self):
        """Vérifie que chaque tier est encodé dans le bon ordre."""
        from api.features import build_matching_features
        base = {
            "rating": 4.0, "global_score": 0.7, "years_experience": 5,
            "hourly_rate": 150, "is_verified": 1, "response_time_h": 3.0,
            "response_rate": 0.8, "completion_rate": 0.9, "category": "Plomberie",
        }
        enc = {}
        for sub in ["free", "basic", "premium", "elite"]:
            X = build_matching_features({**base, "subscription": sub})
            enc[sub] = int(X["subscription_enc"].iloc[0])

        assert enc["free"]    == 0
        assert enc["basic"]   == 1
        assert enc["premium"] == 2
        assert enc["elite"]   == 3

    def test_subscription_boost_ordre(self):
        """SUBSCRIPTION_BOOST doit croître : free < basic < premium < elite."""
        from api.features import SUBSCRIPTION_BOOST
        assert SUBSCRIPTION_BOOST["free"]    < SUBSCRIPTION_BOOST["basic"]
        assert SUBSCRIPTION_BOOST["basic"]   < SUBSCRIPTION_BOOST["premium"]
        assert SUBSCRIPTION_BOOST["premium"] < SUBSCRIPTION_BOOST["elite"]
        assert SUBSCRIPTION_BOOST["elite"]   <= 2.0, "boost elite ne doit pas dépasser ×2"

    def test_haversine(self):
        from api.features import haversine
        dist = haversine(33.5731, -7.5898, 34.0209, -6.8416)
        assert 70 < dist < 110

    def test_categorie_inconnue_ne_plante_pas(self):
        from api.features import build_pricing_features
        sample = {
            "category": "CatégorieInconnue", "city": "Casablanca", "urgency": "normal",
            "desc_len": 50, "n_photos": 0, "budget_min": 200, "budget_max": 1000,
        }
        X = build_pricing_features(sample)
        assert X.shape[0] == 1


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

    def test_pricing_bundle_cles(self, pricing_bundle):
        for key in ["model_q25", "model_q75", "feature_cols", "metrics"]:
            assert key in pricing_bundle

    def test_pricing_prediction_logique(self, pricing_bundle):
        from api.features import build_pricing_features
        sample = {
            "category": "Plomberie", "city": "Casablanca", "urgency": "normal",
            "desc_len": 100, "n_photos": 1, "budget_min": 200, "budget_max": 1000,
        }
        X = build_pricing_features(sample)
        p_min = pricing_bundle["model_q25"].predict(X)[0]
        p_max = pricing_bundle["model_q75"].predict(X)[0]
        assert p_min >= 0
        assert p_max >= p_min

    def test_matching_bundle_cles(self, matching_bundle):
        for key in ["model", "feature_cols", "metrics"]:
            assert key in matching_bundle

    def test_matching_bundle_contient_subscription_map(self, matching_bundle):
        """Le bundle doit inclure subscription_map pour traçabilité."""
        assert "subscription_map" in matching_bundle

    def test_matching_features_incluent_subscription(self, matching_bundle):
        """Le modèle entraîné doit avoir subscription dans ses features."""
        assert "subscription_enc" in matching_bundle["feature_cols"], \
            "subscription_enc manquant — réentraîne avec train_matching.py"
        assert "sub_x_score"      in matching_bundle["feature_cols"]
        assert "sub_x_verified"   in matching_bundle["feature_cols"]

    def test_matching_prediction_probabilite(self, matching_bundle):
        from api.features import build_matching_features
        artisan = {
            "rating": 4.5, "global_score": 0.85, "years_experience": 10,
            "hourly_rate": 200, "is_verified": 1, "response_time_h": 1.5,
            "response_rate": 0.95, "completion_rate": 0.98, "category": "Plomberie",
            "subscription": "elite",
        }
        X    = build_matching_features(artisan)
        prob = matching_bundle["model"].predict_proba(X)[0][1]
        assert 0.0 <= prob <= 1.0

    def test_subscription_boost_applied_correctly(self, matching_bundle):
        """
        Un artisan elite doit avoir un score final plus élevé qu'un artisan free
        ayant les mêmes autres caractéristiques.
        """
        from api.features import build_matching_features, SUBSCRIPTION_BOOST

        base = {
            "rating": 4.0, "global_score": 0.75, "years_experience": 7,
            "hourly_rate": 180, "is_verified": 1, "response_time_h": 2.0,
            "response_rate": 0.9, "completion_rate": 0.95, "category": "Plomberie",
        }
        scores = {}
        for sub in ["free", "basic", "premium", "elite"]:
            artisan     = {**base, "subscription": sub}
            X           = build_matching_features(artisan)
            raw         = matching_bundle["model"].predict_proba(X)[0][1]
            scores[sub] = min(1.0, raw * SUBSCRIPTION_BOOST[sub])

        assert scores["elite"]   > scores["free"],    \
            f"elite ({scores['elite']:.4f}) doit > free ({scores['free']:.4f})"
        assert scores["premium"] > scores["free"],    \
            f"premium ({scores['premium']:.4f}) doit > free ({scores['free']:.4f})"
        assert scores["basic"]   > scores["free"],    \
            f"basic ({scores['basic']:.4f}) doit > free ({scores['free']:.4f})"

    def test_pricing_metriques(self, pricing_bundle):
        m = pricing_bundle["metrics"]
        for key in ["mae", "rmse", "coverage"]:
            assert key in m
        assert m["mae"] > 0
        assert 0 <= m["coverage"] <= 100

    def test_matching_metriques(self, matching_bundle):
        m = matching_bundle["metrics"]
        for key in ["auc", "pk5"]:
            assert key in m
        assert 0 <= m["auc"] <= 1
        assert 0 <= m["pk5"] <= 1
