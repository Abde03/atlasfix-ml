"""
matching.py — Router FastAPI pour le module Matching
=====================================================
Endpoint :

  POST /matching/rank → score et classe les artisans candidats pour une demande

Flux :
  1. Laravel pré-filtre les artisans (même catégorie, disponibles)
     et les envoie à ce endpoint
  2. Le modèle LightGBM prédit P(accepted=1) pour chaque artisan
  3. Les artisans sont triés par ce score (+ distance comme tiebreaker)
  4. Les top_n sont retournés à Laravel qui les affiche au client
"""

import numpy as np
from fastapi import APIRouter, Request, HTTPException

from api.schemas  import MatchRequest, MatchResponse, RankedArtisan
from api.features import build_matching_features, haversine

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
          'artisans'   => $artisans->toArray(),
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

            # Feature engineering (identique au notebook 03)
            X = build_matching_features(a)

            # Score du modèle LightGBM : P(accepted=1)
            match_score = float(model.predict_proba(X)[0][1])

            # Distance entre la demande et l'artisan
            dist_km = haversine(
                req.latitude, req.longitude,
                a["latitude"],  a["longitude"]
            )

            results.append({
                "id":          a["id"],
                "name":        a["name"],
                "city":        a["city"],
                "match_score": round(match_score, 4),
                "rating":      a["rating"],
                "hourly_rate": a["hourly_rate"],
                "is_verified": a["is_verified"],
                "distance_km": dist_km,
            })

        # Tri : score décroissant, puis distance croissante (tiebreaker)
        results.sort(key=lambda x: (-x["match_score"], x["distance_km"]))

        # Ajouter le rang
        ranked = []
        for i, r in enumerate(results[: req.top_n]):
            ranked.append(RankedArtisan(**r, rank=i + 1))

        return MatchResponse(
            demand_id=req.demand_id,
            artisans=ranked,
            model_used="lgbm-matching",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
