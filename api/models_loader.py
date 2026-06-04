"""
models_loader.py
================
Charge les fichiers .pkl depuis models/production/ (ou staging/ en dev).
Appelé une seule fois au démarrage de FastAPI.
"""

import os
import joblib

# Cherche d'abord en production, sinon staging (développement)
MODEL_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "models", "production"),
    os.path.join(os.path.dirname(__file__), "..", "models", "staging"),
]


def find_model(filename: str) -> str:
    """Cherche un fichier .pkl dans production puis staging."""
    for directory in MODEL_DIRS:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            print(f"  Modèle trouvé : {path}")
            return path
    raise FileNotFoundError(
        f"{filename} introuvable dans models/production/ ni models/staging/\n"
        f"Lance d'abord les notebooks 02 et 03 pour entraîner les modèles."
    )


def load_all_models() -> dict:
    """
    Charge et retourne tous les modèles dans un dict.
    Stocké dans app.state.models → accessible dans tous les routers.
    """
    return {
        "pricing":  joblib.load(find_model("pricing_model.pkl")),
        "matching": joblib.load(find_model("matching_model.pkl")),
    }
