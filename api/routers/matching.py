"""
matching.py — Router FastAPI pour le module Matching
=====================================================
Endpoint : POST /matching/rank

Score final = P(accepted) du modèle LightGBM × SUBSCRIPTION_BOOST
  - Le modèle LightGBM apprend que subscription corrèle avec acceptance
  - Le boost multiplicatif garantit une séparation visible dans le ranking
  - L'admin peut ajuster SUBSCRIPTION_BOOST sans réentraîner
"""

import numpy as np
from fastapi import APIRouter, Request, HTTPException

from api.schemas  import MatchRequest, MatchResponse, RankedArtisan
from api.features import build_matching_features, haversine, SUBSCRIPTION_BOOST

router = APIRouter()


@router.post("/rank", response_model=MatchResponse)
def rank_artisans(req: MatchRequest, request: Request):
    """
    Appelé par Laravel à chaque publication d'une nouvelle demande.

    Exemple depuis Laravel :
      $artisans = Artisan::where('category', $demand->category)
                         ->where('is_available', true)
                         ->get();

      Http::post('http://ml-service:8001/matching/rank', [
          'demand_id'  => $demand->id,
          'category'   => $demand->category,
          'city'       => $demand->city,
          'latitude'   => $demand->latitude,
          'longitude'  => $demand->longitude,
          'top_n'      => 10,
          'artisans'   => $artisans->toArray(),  // doit inclure 'subscription'
      ])
    """
    try:
        bundle = request.app.state.models["matching"]
        model  = bundle["model"]

        if not req.artisans:
            return MatchResponse(
                demand_id=req.demand_id,
                artisans=[],
                model_used="lgbm-matching",
            )

        results = []
        for artisan in req.artisans:
            a = artisan.model_dump()

            # Feature engineering (identique à train_matching.py)
            X = build_matching_features(a)

            # ── Score du modèle LightGBM : P(accepted=1) ──────────────────────
            raw_score = float(model.predict_proba(X)[0][1])

            # ── Subscription boost ────────────────────────────────────────────
            # Appliqué APRÈS le modèle pour garder la séparation visible
            # et permettre à l'admin de l'ajuster sans réentraîner
            sub   = a.get("subscription", "free")
            boost = SUBSCRIPTION_BOOST.get(sub, 1.0)
            match_score = min(1.0, raw_score * boost)   # cap à 1.0

            # Distance géographique
            dist_km = haversine(
                req.latitude,   req.longitude,
                a["latitude"],  a["longitude"]
            )

            results.append({
                "id":           a["id"],
                "name":         a["name"],
                "city":         a["city"],
                "match_score":  round(match_score, 4),
                "raw_score":    round(raw_score, 4),   # utile pour debug/admin
                "rating":       a["rating"],
                "hourly_rate":  a["hourly_rate"],
                "is_verified":  a["is_verified"],
                "distance_km":  dist_km,
                "subscription": sub,                   # ← NOUVEAU
            })

        # Tri : score final décroissant, distance croissante comme tiebreaker
        results.sort(key=lambda x: (-x["match_score"], x["distance_km"]))

        # Ajouter le rang
        ranked = []
        for i, r in enumerate(results[: req.top_n]):
            ranked.append(RankedArtisan(**r, rank=i + 1))

        return MatchResponse(
            demand_id=req.demand_id,
            artisans=ranked,
            model_used="lgbm-matching-with-subscription",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
