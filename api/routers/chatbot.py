"""
chatbot.py — Router FastAPI pour le Chatbot multi-agents
=========================================================
Architecture RAG légère (sans dépendance OpenAI obligatoire) :
  1. Classifier l'intent du message
  2. Retrouver le document le plus pertinent dans la base de connaissances
  3. Retourner la réponse contextualisée

Pour ajouter un vrai LLM plus tard :
  - Mettre OPENAI_API_KEY dans .env
  - Décommenter le bloc _generate_with_llm()
"""

import os
from fastapi import APIRouter, Request
from api.schemas import ChatRequest, ChatResponse

router = APIRouter()

# ── Base de connaissances statique ─────────────────────────────────────────────
# Quand tu auras un LLM : remplace ces réponses par des appels à l'API
KB = {
    "client": [
        {
            "keywords": ["publier", "créer", "nouvelle demande", "poster", "demande"],
            "intent":   "publish_demand",
            "response": (
                "Pour publier une demande : connectez-vous → 'Nouvelle demande' → "
                "remplissez le titre, la description, la ville, votre budget et les photos. "
                "Le système vous affichera automatiquement une estimation de prix avant validation."
            ),
        },
        {
            "keywords": ["prix", "tarif", "combien", "estimation", "budget", "coût"],
            "intent":   "price_question",
            "response": (
                "L'estimation de prix est calculée par notre modèle d'IA sur la base "
                "des offres acceptées pour des demandes similaires dans votre ville. "
                "Elle est indicative — les artisans proposent leur propre prix."
            ),
        },
        {
            "keywords": ["accepter", "offre", "choisir", "artisan", "valider"],
            "intent":   "accept_offer",
            "response": (
                "Pour accepter une offre : allez dans 'Mes demandes' → sélectionnez votre demande "
                "→ comparez les offres (prix, note, score de l'artisan) → cliquez 'Accepter'. "
                "Les autres offres sont automatiquement rejetées."
            ),
        },
        {
            "keywords": ["avis", "noter", "évaluation", "note", "commentaire"],
            "intent":   "review",
            "response": (
                "Pour laisser un avis : après la prestation, vous recevrez une notification. "
                "Allez dans 'Mes demandes' → demande terminée → 'Laisser un avis'. "
                "Notez l'artisan sur 5 étoiles et laissez un commentaire."
            ),
        },
        {
            "keywords": ["réclamation", "problème", "litige", "plainte"],
            "intent":   "complaint",
            "response": (
                "Pour déposer une réclamation : Profil → 'Mes réclamations' → 'Nouvelle réclamation'. "
                "Décrivez le problème avec des photos si possible. "
                "Un administrateur traitera votre demande sous 48h."
            ),
        },
    ],
    "artisan": [
        {
            "keywords": ["offre", "envoyer", "proposer", "candidater", "répondre"],
            "intent":   "send_offer",
            "response": (
                "Pour envoyer une offre : consultez les demandes ouvertes filtrées par votre catégorie "
                "→ cliquez sur une demande → consultez la fourchette de prix estimée → "
                "cliquez 'Envoyer une offre' avec votre prix et un message personnalisé."
            ),
        },
        {
            "keywords": ["score", "réputation", "améliorer", "classement", "visibilité"],
            "intent":   "improve_score",
            "response": (
                "Votre score est calculé sur 4 axes : qualité (notes reçues), "
                "fiabilité (taux de réponse, complétion), confiance (vérification, ancienneté), "
                "activité (nombre de missions). Répondez rapidement et complétez vos missions "
                "pour améliorer votre score."
            ),
        },
        {
            "keywords": ["vérification", "badge", "documents", "certifié", "valider profil"],
            "intent":   "verify_profile",
            "response": (
                "Pour faire vérifier votre profil : Paramètres → 'Vérification' → "
                "soumettez votre CIN, justificatif de métier et attestation de formation. "
                "La vérification est traitée sous 72h et augmente votre visibilité dans les recommandations."
            ),
        },
        {
            "keywords": ["disponibilité", "horaire", "créneau", "agenda", "calendrier"],
            "intent":   "availability",
            "response": (
                "Pour gérer vos disponibilités : Tableau de bord → 'Mes disponibilités' → "
                "définissez vos créneaux semaine par semaine. "
                "Les demandes urgentes seront filtrées selon vos disponibilités."
            ),
        },
    ],
    "admin": [
        {
            "keywords": ["valider", "artisan", "profil", "attente", "vérifier"],
            "intent":   "validate_artisan",
            "response": (
                "Pour valider un profil artisan : Administration → 'Profils en attente' → "
                "examinez les documents soumis → cliquez 'Valider' ou 'Rejeter' avec un motif."
            ),
        },
        {
            "keywords": ["réclamation", "litige", "traiter", "résoudre"],
            "intent":   "manage_complaint",
            "response": (
                "Pour traiter une réclamation : Administration → 'Réclamations' → "
                "sélectionnez la réclamation → examinez les éléments des deux parties → "
                "clôturez avec une décision motivée."
            ),
        },
        {
            "keywords": ["métriques", "ia", "modèle", "matching", "pricing"],
            "intent":   "monitor_ai",
            "response": (
                "Les métriques ML sont accessibles dans Administration → 'Supervision IA'. "
                "Vous pouvez y consulter MAE du pricing, Precision@K du matching, "
                "et le F1-score du classifieur d'intents du chatbot."
            ),
        },
    ],
}

FALLBACK = "Je n'ai pas bien compris votre question. Pouvez-vous la reformuler ? " \
           "Vous pouvez me demander de l'aide sur les demandes, les offres, les prix ou votre profil."

# Sessions en mémoire (remplacer par Redis en production)
_sessions: dict = {}


def _classify(message: str, role: str) -> tuple[str, float, str]:
    """
    Retourne (intent, confidence, response) pour un message.
    Classifieur par mots-clés — remplace par un modèle ML si besoin.
    """
    msg = message.lower()
    kb  = KB.get(role, KB["client"])

    best_score    = 0
    best_entry    = None

    for entry in kb:
        score = sum(1 for kw in entry["keywords"] if kw in msg)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry and best_score > 0:
        confidence = min(1.0, 0.4 + best_score * 0.2)
        return best_entry["intent"], confidence, best_entry["response"]

    return "general_faq", 0.3, FALLBACK


@router.post("/message", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    """
    Appelé par Laravel pour chaque message utilisateur.

    Exemple depuis Laravel :
      Http::post('http://ml-service:8001/chatbot/message', [
          'message'    => $request->message,
          'user_role'  => auth()->user()->role,
          'session_id' => $request->session_id,
          'user_id'    => auth()->id(),
      ])
    """
    intent, confidence, response = _classify(req.message, req.user_role.value)

    # Historique de session (pour contexte multi-tours)
    history = _sessions.get(req.session_id, [])
    history.append({"role": "user", "content": req.message})
    history.append({"role": "bot",  "content": response})
    _sessions[req.session_id] = history[-10:]  # garder les 5 derniers échanges

    return ChatResponse(
        response=response,
        intent=intent,
        confidence=round(confidence, 2),
        session_id=req.session_id,
    )


@router.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Appelé par Laravel quand l'utilisateur ferme le widget chatbot."""
    _sessions.pop(session_id, None)
    return {"status": "ok"}
