"""
generate_data.py
================
Génère les données synthétiques pour AtlasFix.
Lance avec : python data/generate_data.py

Produit 4 fichiers dans data/raw/ :
  - artisans.csv         → profils des artisans
  - clients.csv          → profils des clients
  - demands.csv          → demandes publiées
  - accepted_offers.csv  → offres acceptées (vérité terrain pour le pricing)
  - interactions.csv     → clics, vues, acceptations (pour le matching)
"""

import os
import csv
import math
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker("fr_FR")
random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_ARTISANS    = 300
N_CLIENTS     = 400
N_DEMANDS     = 800
N_INTERACTIONS = 3000

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Données métier ─────────────────────────────────────────────────────────────
CITIES = [
    ("Casablanca", 33.5731, -7.5898),
    ("Rabat",      34.0209, -6.8416),
    ("Fès",        34.0333, -5.0000),
    ("Marrakech",  31.6295, -7.9811),
    ("Agadir",     30.4278, -9.5981),
    ("Tanger",     35.7595, -5.8340),
    ("Meknès",     33.8935, -5.5473),
    ("Oujda",      34.6867, -1.9114),
]

CATEGORIES = [
    "Plomberie", "Électricité", "Menuiserie", "Maçonnerie",
    "Peinture", "Carrelage", "Serrurerie", "Jardinage",
    "Nettoyage", "Déménagement", "Climatisation", "Informatique",
]

# Prix de base par catégorie (min, max en MAD)
PRIX_BASE = {
    "Plomberie":     (150,  900),
    "Électricité":   (200, 1400),
    "Menuiserie":    (300, 2200),
    "Maçonnerie":    (400, 3500),
    "Peinture":      (200, 1600),
    "Carrelage":     (250, 2000),
    "Serrurerie":    (100,  700),
    "Jardinage":     (150,  900),
    "Nettoyage":     (100,  600),
    "Déménagement":  (300, 2200),
    "Climatisation": (500, 3200),
    "Informatique":  (100,  900),
}

URGENCES = ["normal", "urgent", "très urgent"]
URGENCE_MULT = {"normal": 1.0, "urgent": 1.25, "très urgent": 1.55}

TITRES = {
    "Plomberie":     ["Fuite sous l'évier", "Chauffe-eau en panne", "WC bouché", "Robinet qui goutte"],
    "Électricité":   ["Panne de courant", "Installer des prises", "Tableau à refaire", "Éclairage LED"],
    "Menuiserie":    ["Porte qui frotte", "Placard sur mesure", "Réparer parquet"],
    "Maçonnerie":    ["Fissures dans le mur", "Créer une ouverture", "Enduit à refaire"],
    "Peinture":      ["Repeindre le salon", "Rafraîchir façade", "Décoration chambre"],
    "Carrelage":     ["Poser carrelage salle de bain", "Joints à refaire", "Terrasse à carreler"],
    "Serrurerie":    ["Serrure bloquée", "Changer cylindre", "Installer interphone"],
    "Jardinage":     ["Taille de haie", "Système d'arrosage", "Engazonnement"],
    "Nettoyage":     ["Nettoyage après travaux", "Lavage vitres", "Désinfection locaux"],
    "Déménagement":  ["Déménagement 3 pièces", "Transport canapé", "Montage meubles"],
    "Climatisation": ["Clim qui ne refroidit plus", "Installer pompe à chaleur", "Chaudière à réviser"],
    "Informatique":  ["PC qui ne démarre plus", "Réseau WiFi", "Caméras surveillance"],
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def jitter(lat, lon, km=12):
    d = km / 111
    return round(lat + random.uniform(-d, d), 5), round(lon + random.uniform(-d, d), 5)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return round(R * 2 * math.asin(math.sqrt(a)), 2)

def random_date(days_ago=365):
    start = datetime.now() - timedelta(days=days_ago)
    return (start + timedelta(days=random.randint(0, days_ago))).strftime("%Y-%m-%d")

def phone():
    return random.choice(["06", "07"]) + "".join([str(random.randint(0,9)) for _ in range(8)])

def write_csv(filename, rows):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ {filename} ({len(rows)} lignes)")

# ── 1. Artisans ────────────────────────────────────────────────────────────────
print("Génération des artisans...")
artisans = []
for i in range(N_ARTISANS):
    city, base_lat, base_lon = random.choice(CITIES)
    lat, lon = jitter(base_lat, base_lon)
    cat = random.choice(CATEGORIES)

    # Uniform pour maximiser la variance — le modèle matching a besoin d'artisans très différents
    rating = round(random.uniform(1.5, 5.0), 1)
    n_reviews = random.randint(0, 130)
    exp = random.randint(1, 25)

    # Score global synthétique (sera recalculé par le vrai modèle)
    score = round(
        0.35 * (rating / 5)
        + 0.30 * random.uniform(0.5, 1.0)   # fiabilité simulée
        + 0.20 * (1 if random.random() > 0.4 else 0.4)   # confiance
        + 0.15 * min(exp / 20, 1.0),         # activité
        3
    )

    artisans.append({
        "id":                i + 1,
        "name":              fake.name(),
        "phone":             phone(),
        "city":              city,
        "latitude":          lat,
        "longitude":         lon,
        "category":          cat,
        "years_experience":  exp,
        "hourly_rate":       random.randint(80, 350),
        "rating":            rating,
        "n_reviews":         n_reviews,
        "response_time_h":   round(random.expovariate(1/3), 1),
        "response_rate":     round(random.uniform(0.5, 1.0), 2),
        "completion_rate":   round(random.uniform(0.7, 1.0), 2),
        "no_show_rate":      round(random.uniform(0.0, 0.15), 2),
        "is_verified":       random.choices([1, 0], weights=[0.6, 0.4])[0],
        "is_available":      random.choices([1, 0], weights=[0.8, 0.2])[0],
        "global_score":      score,
        "created_at":        random_date(730),
    })

write_csv("artisans.csv", artisans)

# ── 2. Clients ─────────────────────────────────────────────────────────────────
print("Génération des clients...")
clients = []
for i in range(N_CLIENTS):
    city, base_lat, base_lon = random.choice(CITIES)
    lat, lon = jitter(base_lat, base_lon)
    clients.append({
        "id":         i + 1,
        "name":       fake.name(),
        "phone":      phone(),
        "city":       city,
        "latitude":   lat,
        "longitude":  lon,
        "created_at": random_date(730),
    })

write_csv("clients.csv", clients)

# ── 3. Demandes + offres acceptées (vérité terrain pricing) ────────────────────
print("Génération des demandes et offres acceptées...")
demands = []
accepted_offers = []

artisans_by_cat = {}
for a in artisans:
    artisans_by_cat.setdefault(a["category"], []).append(a)

for i in range(N_DEMANDS):
    client = random.choice(clients)
    cat = random.choice(CATEGORIES)
    city_name, base_lat, base_lon = random.choice(CITIES)
    lat, lon = jitter(base_lat, base_lon)
    urgency = random.choices(URGENCES, weights=[0.6, 0.3, 0.1])[0]
    mult = URGENCE_MULT[urgency]
    p_min, p_max = PRIX_BASE[cat]
    desc_len = random.randint(30, 350)
    n_photos = random.randint(0, 5)
    status = random.choices(["open", "completed"], weights=[0.35, 0.65])[0]

    demand = {
        "id":          i + 1,
        "client_id":   client["id"],
        "title":       random.choice(TITRES[cat]),
        "category":    cat,
        "city":        city_name,
        "latitude":    lat,
        "longitude":   lon,
        "urgency":     urgency,
        "budget_min":  int(p_min * 0.7),
        "budget_max":  int(p_max * 1.3),
        "desc_len":    desc_len,
        "n_photos":    n_photos,
        "status":      status,
        "created_at":  random_date(365),
    }
    demands.append(demand)

    # Pour les demandes complétées → on crée une offre acceptée
    # C'est la VÉRITÉ TERRAIN pour entraîner le modèle de pricing
    if status == "completed":
        candidates = artisans_by_cat.get(cat, artisans)
        artisan = random.choice(candidates)
        real_price = int(random.randint(int(p_min * mult), int(p_max * mult)))
        dist = haversine(lat, lon, artisan["latitude"], artisan["longitude"])

        accepted_offers.append({
            # ── Features (inputs du modèle) ──
            "category":          cat,
            "city":              city_name,
            "urgency":           urgency,
            "desc_len":          desc_len,
            "n_photos":          n_photos,
            "budget_min":        demand["budget_min"],
            "budget_max":        demand["budget_max"],
            "artisan_rating":    artisan["rating"],
            "artisan_score":     artisan["global_score"],
            "artisan_exp":       artisan["years_experience"],
            "distance_km":       dist,
            "same_city":         1 if artisan["city"] == city_name else 0,
            # ── Target (ce qu'on veut prédire) ──
            "accepted_price":    real_price,
        })

write_csv("demands.csv", demands)
write_csv("accepted_offers.csv", accepted_offers)

# ── 4. Interactions (pour le matching collaboratif) ────────────────────────────
print("Génération des interactions...")
interactions = []
TYPES = ["view_profile", "message_sent", "accept_offer", "reject_offer", "ignore"]
POIDS = [0.35, 0.20, 0.15, 0.15, 0.15]

for i in range(N_INTERACTIONS):
    client = random.choice(clients)
    artisan = random.choice(artisans)

    # Quality signal — artisan quality drives accept probability so the model can learn
    q = (
        artisan["rating"] / 5.0   * 0.40
        + artisan["global_score"] * 0.30
        + artisan["response_rate"]* 0.20
        + artisan["is_verified"]  * 0.10
    )  # q ∈ [~0.20, ~0.95] with uniform rating

    p_accept = 0.02 + q * 0.46   # low-quality: ~0.11  high-quality: ~0.46
    p_reject = 0.27 - q * 0.22   # low-quality: ~0.22  high-quality: ~0.06
    p_msg    = 0.15 + q * 0.07
    p_view   = 0.25
    p_ignore = max(0.0, 1.0 - p_accept - p_reject - p_msg - p_view)

    itype = random.choices(TYPES, weights=[p_view, p_msg, p_accept, p_reject, p_ignore])[0]

    # Valeur numérique de l'interaction (pour le CF)
    value_map = {
        "view_profile": 0.2,
        "message_sent": 0.6,
        "accept_offer": 1.0,
        "reject_offer": -0.3,
        "ignore":       0.0,
    }

    interactions.append({
        "client_id":  client["id"],
        "artisan_id": artisan["id"],
        "category":   artisan["category"],
        "type":       itype,
        "value":      value_map[itype],
        "date":       random_date(365),
    })

write_csv("interactions.csv", interactions)

# ── Résumé ─────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("DONNÉES GÉNÉRÉES")
print("="*50)
print(f"  Artisans           : {len(artisans)}")
print(f"  Clients            : {len(clients)}")
print(f"  Demandes           : {len(demands)}")
print(f"  Offres acceptées   : {len(accepted_offers)}  ← vérité terrain pricing")
print(f"  Interactions       : {len(interactions)}  ← vérité terrain matching")
print(f"\n  Dossier            : {OUTPUT_DIR}")
print("="*50)
print("\nProchaine étape : ouvre notebooks/01_EDA.ipynb")
