"""
features.py
===========
Fonctions de feature engineering partagées entre l'API et les notebooks.

⚠️  RÈGLE IMPORTANTE :
Ces fonctions doivent rester IDENTIQUES à celles des notebooks d'entraînement.
Si tu modifies build_matching_features() ici, modifie-la aussi dans 03_train_matching.ipynb.
"""

import pandas as pd
import numpy as np
import math

CATEGORIES = [
    "Plomberie", "Électricité", "Menuiserie", "Maçonnerie",
    "Peinture", "Carrelage", "Serrurerie", "Jardinage",
    "Nettoyage", "Déménagement", "Climatisation", "Informatique",
]
CITIES = [
    "Casablanca", "Rabat", "Fès", "Marrakech",
    "Agadir", "Tanger", "Meknès", "Oujda",
]
URGENCY_MAP = {"normal": 0, "urgent": 1, "très urgent": 2}

# ── Subscription ───────────────────────────────────────────────────────────────
SUBSCRIPTION_MAP   = {"free": 0, "basic": 1, "premium": 2, "elite": 3}

# Boost multiplicatif appliqué SUR le score brut du modèle dans le router.
# Séparé du modèle pour que l'admin puisse l'ajuster sans réentraîner.
SUBSCRIPTION_BOOST = {"free": 1.00, "basic": 1.15, "premium": 1.35, "elite": 1.60}


# ── Pricing features ──────────────────────────────────────────────────────────
def build_pricing_features(data: dict) -> pd.DataFrame:
    """
    Transforme un dict de requête en DataFrame de features pour le modèle pricing.
    (inchangé — subscription ne joue pas sur le prix)
    """
    df = pd.DataFrame([data])

    X = pd.DataFrame()
    X["category_enc"]     = pd.Categorical(df["category"], categories=CATEGORIES).codes
    X["city_enc"]         = pd.Categorical(df["city"],     categories=CITIES).codes
    X["urgency_enc"]      = df["urgency"].map(URGENCY_MAP).fillna(0).astype(int)
    X["budget_min"]       = df.get("budget_min", 200).fillna(200) if hasattr(df.get("budget_min", 200), "fillna") else pd.Series([data.get("budget_min", 200)])
    X["budget_max"]       = pd.Series([data.get("budget_max", 1500)])
    X["desc_len"]         = pd.Series([data.get("desc_len", 80)])
    X["n_photos"]         = pd.Series([data.get("n_photos", 0)])
    X["artisan_rating"]   = pd.Series([data.get("artisan_rating", 4.0)])
    X["artisan_score"]    = pd.Series([data.get("artisan_score", 0.7)])
    X["artisan_exp"]      = pd.Series([data.get("artisan_exp", 5)])
    X["distance_km"]      = pd.Series([data.get("distance_km", 10.0)])
    X["same_city"]        = pd.Series([int(data.get("same_city", 1))])
    X["budget_range"]     = X["budget_max"] - X["budget_min"]
    X["urgency_x_budget"] = X["urgency_enc"] * X["budget_max"]
    X["rating_x_exp"]     = X["artisan_rating"] * X["artisan_exp"]
    X["photos_x_desc"]    = X["n_photos"] * X["desc_len"]

    return X


# ── Matching features ─────────────────────────────────────────────────────────
def build_matching_features(artisan: dict) -> pd.DataFrame:
    """
    Transforme un dict artisan en DataFrame de features pour le modèle matching.
    Inclut subscription comme feature apprise par le modèle.
    """
    X = pd.DataFrame()

    # Features artisan de base (inchangées)
    X["artisan_rating"]     = pd.Series([artisan.get("rating", 3.0)])
    X["artisan_score"]      = pd.Series([artisan.get("global_score", 0.5)])
    X["years_exp"]          = pd.Series([artisan.get("years_experience", 5)])
    X["hourly_rate"]        = pd.Series([artisan.get("hourly_rate", 150)])
    X["is_verified"]        = pd.Series([int(artisan.get("is_verified", 0))])
    X["response_time_h"]    = pd.Series([artisan.get("response_time_h", 12)])
    X["response_rate"]      = pd.Series([artisan.get("response_rate", 0.7)])
    X["completion_rate"]    = pd.Series([artisan.get("completion_rate", 0.8)])
    X["category_enc"]       = pd.Categorical(
                                  [artisan.get("category", "Plomberie")],
                                  categories=CATEGORIES
                              ).codes

    # ── Subscription (NOUVEAU) ────────────────────────────────────────────────
    sub                     = str(artisan.get("subscription", "free"))
    X["subscription_enc"]   = pd.Series([SUBSCRIPTION_MAP.get(sub, 0)])

    # Features dérivées existantes
    X["rating_x_verified"]  = X["artisan_rating"] * X["is_verified"]
    X["score_x_completion"] = X["artisan_score"]  * X["completion_rate"]
    X["exp_normalized"]     = (X["years_exp"] / 25.0).clip(0, 1)
    X["resp_speed_score"]   = (1.0 - (X["response_time_h"] / 24).clip(0, 1))

    # Features dérivées avec subscription (NOUVEAU)
    # Ces deux features aident le modèle à capturer les interactions entre
    # l'abonnement et la qualité/vérification
    X["sub_x_score"]        = X["subscription_enc"] * X["artisan_score"]
    X["sub_x_verified"]     = X["subscription_enc"] * X["is_verified"]

    return X


# ── Utilitaires ───────────────────────────────────────────────────────────────
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en km entre deux points GPS."""
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return round(R * 2 * math.asin(math.sqrt(a)), 2)
