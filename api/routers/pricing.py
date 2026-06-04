"""
pricing.py — Router FastAPI pour le module Pricing
===================================================
Deux endpoints :

  POST /pricing/estimate    → prédit [p_min, p_max] pour une demande
  POST /pricing/check-offer → vérifie si le prix d'une offre est raisonnable
"""

import numpy as np
from fastapi import APIRouter, Request, HTTPException

from api.schemas  import (PriceEstimateRequest, PriceEstimateResponse,
                           OfferCheckRequest,    OfferCheckResponse)
from api.features import build_pricing_features

router = APIRouter()


def _predict(bundle: dict, data: dict) -> tuple:
    """
    Appelle les deux modèles Q25 et Q75 et retourne (p_min, p_max, confidence).
    C'est ici que le vrai modèle LightGBM est utilisé.
    """
    X = build_pricing_features(data)

    p_min = float(bundle["model_q25"].predict(X)[0])
    p_max = float(bundle["model_q75"].predict(X)[0])

    # S'assurer que p_min <= p_max et positifs
    p_min = max(50.0, p_min)
    p_max = max(p_min + 50, p_max)

    # Confiance = inverse de la largeur relative
    width    = p_max - p_min
    midpoint = (p_min + p_max) / 2
    confidence = round(max(0.0, min(1.0, 1.0 - (width / max(midpoint, 1)))), 3)

    return round(p_min), round(p_max), confidence


@router.post("/estimate", response_model=PriceEstimateResponse)
def estimate_price(req: PriceEstimateRequest, request: Request):
    """
    Appelé par Laravel quand le client remplit son formulaire de demande.
    Laravel affiche la fourchette à l'utilisateur avant qu'il valide.

    Exemple de requête depuis Laravel :
      Http::post('http://ml-service:8001/pricing/estimate', [
          'category'   => $demand->service_type,
          'city'       => $demand->city,
          'urgency'    => $demand->urgency,
          'desc_len'   => strlen($demand->description),
          'n_photos'   => count($demand->photos),
          'budget_min' => $demand->budget_min,
          'budget_max' => $demand->budget_max,
      ])
    """
    try:
        bundle = request.app.state.models["pricing"]
        p_min, p_max, confidence = _predict(bundle, req.model_dump())
        return PriceEstimateResponse(
            p_min=p_min,
            p_max=p_max,
            confidence=confidence,
            source="model",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-offer", response_model=OfferCheckResponse)
def check_offer(req: OfferCheckRequest, request: Request):
    """
    Appelé par Laravel quand un artisan soumet une offre.
    Retourne si le prix est OK / trop haut / trop bas / suspect.

    Laravel affiche un avertissement si le statut n'est pas OK.
    """
    try:
        bundle = request.app.state.models["pricing"]
        p_min, p_max, _ = _predict(bundle, req.model_dump())

        offered = req.offered_price
        midpoint = (p_min + p_max) / 2
        deviation = round(((offered - midpoint) / max(midpoint, 1)) * 100, 1)

        if p_min <= offered <= p_max:
            status  = "OK"
            message = "Prix dans la fourchette estimée du marché."
        elif offered > p_max * 1.6:
            status  = "SUSPICIOUS"
            message = f"Prix {abs(deviation):.0f}% au-dessus du marché — vérification recommandée."
        elif offered > p_max:
            status  = "HIGH"
            message = f"Prix {abs(deviation):.0f}% au-dessus de la fourchette estimée."
        else:
            status  = "LOW"
            message = f"Prix {abs(deviation):.0f}% en dessous de la fourchette estimée."

        return OfferCheckResponse(
            status=status,
            p_min=p_min,
            p_max=p_max,
            deviation=deviation,
            message=message,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
