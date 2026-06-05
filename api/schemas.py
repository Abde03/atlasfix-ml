"""
schemas.py
==========
Définit la forme exacte des requêtes et réponses JSON.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class Urgency(str, Enum):
    normal      = "normal"
    urgent      = "urgent"
    tres_urgent = "très urgent"

class OfferStatus(str, Enum):
    ok         = "OK"
    high       = "HIGH"
    low        = "LOW"
    suspicious = "SUSPICIOUS"

class UserRole(str, Enum):
    client  = "client"
    artisan = "artisan"
    admin   = "admin"

class Subscription(str, Enum):
    free    = "free"
    basic   = "basic"
    premium = "premium"
    elite   = "elite"


# ── Pricing ───────────────────────────────────────────────────────────────────

class PriceEstimateRequest(BaseModel):
    category:   str     = Field(..., example="Plomberie")
    city:       str     = Field(..., example="Casablanca")
    urgency:    Urgency = Urgency.normal
    desc_len:   int     = Field(80,  ge=0)
    n_photos:   int     = Field(0,   ge=0)
    budget_min: float   = Field(200, ge=0)
    budget_max: float   = Field(1500,ge=0)

class PriceEstimateResponse(BaseModel):
    p_min:      float
    p_max:      float
    confidence: float
    source:     str

class OfferCheckRequest(BaseModel):
    category:      str
    city:          str
    urgency:       Urgency = Urgency.normal
    desc_len:      int     = 80
    n_photos:      int     = 0
    budget_min:    float   = 200
    budget_max:    float   = 1500
    offered_price: float   = Field(..., description="Prix proposé par l'artisan")

class OfferCheckResponse(BaseModel):
    status:    OfferStatus
    p_min:     float
    p_max:     float
    deviation: float
    message:   str


# ── Matching ──────────────────────────────────────────────────────────────────

class ArtisanInput(BaseModel):
    """Un artisan candidat envoyé par Laravel."""
    id:               int
    name:             str
    city:             str
    latitude:         float
    longitude:        float
    category:         str
    rating:           float
    global_score:     float
    years_experience: int
    hourly_rate:      int
    is_verified:      int
    response_time_h:  float
    response_rate:    float      = 0.8
    completion_rate:  float      = 0.9
    is_available:     int        = 1
    subscription:     Subscription = Subscription.free   # ← NOUVEAU

class MatchRequest(BaseModel):
    demand_id:  int
    category:   str
    city:       str
    latitude:   float
    longitude:  float
    artisans:   List[ArtisanInput]
    top_n:      int = Field(10, ge=1, le=50)

class RankedArtisan(BaseModel):
    id:           int
    name:         str
    city:         str
    match_score:  float
    raw_score:    float  = 0.0
    rank:         int
    rating:       float
    hourly_rate:  int
    is_verified:  int
    distance_km:  float
    subscription: str = "free"   # ← NOUVEAU


class MatchResponse(BaseModel):
    demand_id:  int
    artisans:   List[RankedArtisan]
    model_used: str


# ── Chatbot ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str      = Field(..., example="Comment publier une demande ?")
    user_role:  UserRole = UserRole.client
    session_id: str
    user_id:    int

class ChatResponse(BaseModel):
    response:   str
    intent:     str
    confidence: float
    session_id: str
