"""
AtlasFix ML Service — main.py
==============================
Lance avec : uvicorn api.main:app --reload --port 8001

Ce fichier :
  1. Charge les modèles .pkl au démarrage (une seule fois)
  2. Expose 3 groupes d'endpoints pour Laravel :
       POST /pricing/estimate
       POST /pricing/check-offer
       POST /matching/rank
       POST /chatbot/message
  3. Répond en JSON
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import pricing, matching, chatbot, health
from api.models_loader import load_all_models


# ── Démarrage : charger les modèles une seule fois en mémoire ─────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Chargement des modèles ML...")
    app.state.models = load_all_models()
    print("Modèles chargés ✓")
    print(f"  pricing  → {app.state.models['pricing']['metrics']}")
    print(f"  matching → {app.state.models['matching']['metrics']}")
    yield
    print("Arrêt du service ML.")


app = FastAPI(
    title="AtlasFix ML Service",
    description="Pricing · Matching · Chatbot — appelé par Laravel uniquement",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS : uniquement le backend Laravel peut appeler ce service
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",   # Laravel local
        "http://backend:8000",     # Laravel Docker
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(health.router,   tags=["Santé"])
app.include_router(pricing.router,  prefix="/pricing",  tags=["Pricing"])
app.include_router(matching.router, prefix="/matching", tags=["Matching"])
app.include_router(chatbot.router,  prefix="/chatbot",  tags=["Chatbot"])
