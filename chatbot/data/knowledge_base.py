"""
chatbot/data/knowledge_base.py
================================
Base de connaissances d'AtlasFix.
Ce fichier contient TOUS les documents que le chatbot peut retrouver.

Structure de chaque document :
  - id       : identifiant unique
  - role     : à qui s'adresse ce doc (client / artisan / admin / tous)
  - intent   : l'intention couverte
  - question : formulation typique de l'utilisateur
  - answer   : la réponse à donner
  - keywords : mots-clés pour l'indexation

Plus tu ajoutes de documents ici → meilleur sera le chatbot.
Quand l'app sera live, tu pourras enrichir cette base avec de vraies FAQ.
"""

KNOWLEDGE_BASE = [

    # ═══════════════════════════════════════════════════════════════════
    # CLIENT — Publication de demande
    # ═══════════════════════════════════════════════════════════════════
    {
        "id":       "c_publish_1",
        "role":     "client",
        "intent":   "publish_demand",
        "question": "Comment publier une demande ?",
        "answer":   (
            "Pour publier une demande, connectez-vous à votre compte puis cliquez sur "
            "'Nouvelle demande'. Remplissez le titre, la description détaillée, choisissez "
            "la catégorie de service (plomberie, électricité, etc.), indiquez votre ville, "
            "votre budget indicatif et ajoutez des photos si possible. "
            "Notre système affichera automatiquement une estimation de prix avant que vous validiez."
        ),
        "keywords": ["publier", "créer", "nouvelle demande", "poster", "comment faire une demande"],
    },
    {
        "id":       "c_publish_2",
        "role":     "client",
        "intent":   "publish_demand",
        "question": "Quelles informations mettre dans ma demande ?",
        "answer":   (
            "Une bonne demande contient : un titre clair (ex: 'Fuite sous l'évier cuisine'), "
            "une description précise du problème, la localisation exacte, votre disponibilité, "
            "et des photos si possible. Plus votre demande est détaillée, plus vous recevrez "
            "des offres pertinentes et un prix estimé précis."
        ),
        "keywords": ["informations", "description", "que mettre", "détails", "photos"],
    },
    {
        "id":       "c_publish_3",
        "role":     "client",
        "intent":   "publish_demand",
        "question": "Puis-je modifier ma demande après publication ?",
        "answer":   (
            "Oui, vous pouvez modifier ou annuler votre demande tant qu'aucune offre "
            "n'a été acceptée. Allez dans 'Mes demandes', sélectionnez la demande, "
            "puis cliquez sur 'Modifier'. Si des artisans ont déjà envoyé des offres, "
            "ils seront notifiés de la modification."
        ),
        "keywords": ["modifier", "changer", "annuler", "éditer", "corriger demande"],
    },

    # ═══════════════════════════════════════════════════════════════════
    # CLIENT — Offres et matching
    # ═══════════════════════════════════════════════════════════════════
    {
        "id":       "c_offer_1",
        "role":     "client",
        "intent":   "manage_offers",
        "question": "Comment accepter une offre ?",
        "answer":   (
            "Allez dans 'Mes demandes' → sélectionnez votre demande → consultez les offres reçues. "
            "Comparez les prix, les profils et les scores des artisans, puis cliquez 'Accepter' "
            "sur l'offre de votre choix. Les autres offres sont automatiquement rejetées "
            "et l'artisan retenu est notifié immédiatement."
        ),
        "keywords": ["accepter offre", "choisir artisan", "valider offre", "sélectionner"],
    },
    {
        "id":       "c_offer_2",
        "role":     "client",
        "intent":   "manage_offers",
        "question": "Comment comparer les artisans recommandés ?",
        "answer":   (
            "AtlasFix vous recommande automatiquement les artisans les plus pertinents "
            "grâce à notre algorithme de Smart Matching. Pour chaque artisan vous verrez : "
            "son score de matching, sa note moyenne, son nombre d'avis, son tarif horaire, "
            "s'il est vérifié, et sa distance par rapport à vous. "
            "Cliquez sur un profil pour voir ses réalisations et ses avis détaillés."
        ),
        "keywords": ["comparer", "artisans recommandés", "matching", "choisir", "score"],
    },
    {
        "id":       "c_offer_3",
        "role":     "client",
        "intent":   "manage_offers",
        "question": "Je n'ai reçu aucune offre, que faire ?",
        "answer":   (
            "Si vous n'avez pas reçu d'offre après 24-48h, essayez : "
            "1) D'enrichir votre description avec plus de détails, "
            "2) D'augmenter légèrement votre budget si l'estimation IA suggère un tarif plus élevé, "
            "3) D'ajouter des photos du problème, "
            "4) De vérifier que la catégorie de service est bien choisie. "
            "Vous pouvez aussi contacter directement des artisans depuis leurs profils."
        ),
        "keywords": ["pas d'offre", "aucune offre", "personne ne répond", "que faire"],
    },

    # ═══════════════════════════════════════════════════════════════════
    # CLIENT — Prix et estimation
    # ═══════════════════════════════════════════════════════════════════
    {
        "id":       "c_price_1",
        "role":     "client",
        "intent":   "price_question",
        "question": "Comment fonctionne l'estimation de prix ?",
        "answer":   (
            "L'estimation de prix est calculée par notre modèle d'intelligence artificielle "
            "(LightGBM) entraîné sur l'historique des offres acceptées sur AtlasFix. "
            "Il prend en compte la catégorie de service, votre ville, l'urgence, "
            "la complexité décrite et les tarifs du marché local. "
            "La fourchette [prix min, prix max] représente 50% des transactions similaires. "
            "C'est une indication — les artisans proposent leur propre prix."
        ),
        "keywords": ["estimation prix", "tarif", "combien ça coûte", "prix", "fourchette", "IA prix"],
    },
    {
        "id":       "c_price_2",
        "role":     "client",
        "intent":   "price_question",
        "question": "Le prix proposé par un artisan est trop élevé, est-ce normal ?",
        "answer":   (
            "Chaque artisan fixe librement son prix. Si une offre dépasse significativement "
            "la fourchette estimée, AtlasFix l'indique avec un badge 'Prix élevé'. "
            "Cela peut s'expliquer par une spécialisation, du matériel haut de gamme, "
            "ou une forte demande. Vous pouvez négocier via la messagerie intégrée "
            "ou attendre d'autres offres."
        ),
        "keywords": ["prix trop élevé", "cher", "dépassement", "prix anormal", "négocier prix"],
    },

    # ═══════════════════════════════════════════════════════════════════
    # CLIENT — Avis et réclamations
    # ═══════════════════════════════════════════════════════════════════
    {
        "id":       "c_review_1",
        "role":     "client",
        "intent":   "review",
        "question": "Comment laisser un avis sur un artisan ?",
        "answer":   (
            "Après la fin d'une prestation, vous recevrez une notification vous invitant "
            "à évaluer l'artisan. Allez dans 'Mes demandes' → demande terminée → 'Laisser un avis'. "
            "Notez l'artisan de 1 à 5 étoiles et laissez un commentaire. "
            "Votre avis contribue au score de l'artisan et aide les futurs clients."
        ),
        "keywords": ["avis", "noter", "évaluer", "commentaire", "note", "étoiles"],
    },
    {
        "id":       "c_complaint_1",
        "role":     "client",
        "intent":   "complaint",
        "question": "Comment déposer une réclamation ?",
        "answer":   (
            "Si vous avez un problème avec une prestation : Profil → 'Mes réclamations' "
            "→ 'Nouvelle réclamation'. Décrivez le problème précisément, indiquez la demande "
            "concernée et joignez des photos si possible. "
            "Un administrateur examinera votre dossier sous 48h ouvrables et vous contactera."
        ),
        "keywords": ["réclamation", "problème", "litige", "plainte", "mauvaise prestation"],
    },

    # ═══════════════════════════════════════════════════════════════════
    # ARTISAN — Offres et candidatures
    # ═══════════════════════════════════════════════════════════════════
    {
        "id":       "a_offer_1",
        "role":     "artisan",
        "intent":   "send_offer",
        "question": "Comment envoyer une offre sur une demande ?",
        "answer":   (
            "Consultez la liste des demandes ouvertes dans votre catégorie. "
            "Cliquez sur une demande qui vous correspond → consultez la fourchette de prix "
            "estimée par notre IA → cliquez 'Envoyer une offre'. "
            "Saisissez votre prix, la durée estimée et un message personnalisé au client. "
            "Si votre prix s'écarte beaucoup de l'estimation, une alerte vous le signalera."
        ),
        "keywords": ["envoyer offre", "proposer", "candidater", "répondre demande", "faire offre"],
    },
    {
        "id":       "a_offer_2",
        "role":     "artisan",
        "intent":   "send_offer",
        "question": "Comment fixer mon prix pour une offre ?",
        "answer":   (
            "AtlasFix vous affiche la fourchette de prix du marché pour chaque demande. "
            "Pour maximiser vos chances d'être accepté : restez dans cette fourchette, "
            "personnalisez votre message (expliquez votre approche), "
            "et mettez en avant votre expérience spécifique sur ce type de travaux. "
            "Un prix légèrement inférieur à la médiane avec un bon message bat souvent "
            "une offre moins chère sans explication."
        ),
        "keywords": ["fixer prix", "quel prix", "tarif offre", "prix compétitif", "combien proposer"],
    },

    # ═══════════════════════════════════════════════════════════════════
    # ARTISAN — Score et profil
    # ═══════════════════════════════════════════════════════════════════
    {
        "id":       "a_score_1",
        "role":     "artisan",
        "intent":   "improve_score",
        "question": "Comment améliorer mon score sur AtlasFix ?",
        "answer":   (
            "Votre score est calculé sur 4 axes : "
            "1) Qualité (notes reçues — pondérées par récence), "
            "2) Fiabilité (taux de réponse aux messages, délai de réponse, taux de complétion), "
            "3) Confiance (vérification documentaire, ancienneté, absence de réclamations), "
            "4) Activité (fréquence d'offres, missions terminées). "
            "Actions prioritaires : répondez vite aux messages, complétez vos missions, "
            "faites vérifier votre profil."
        ),
        "keywords": ["améliorer score", "score", "réputation", "mieux classé", "visibilité", "points"],
    },
    {
        "id":       "a_score_2",
        "role":     "artisan",
        "intent":   "improve_score",
        "question": "Pourquoi je ne suis pas recommandé en premier ?",
        "answer":   (
            "Le classement dépend de votre score global et de la pertinence pour chaque demande. "
            "Pour remonter dans les recommandations : "
            "1) Faites vérifier votre profil (badge vérifié = boost significatif), "
            "2) Répondez aux messages en moins de 2h, "
            "3) Complétez toutes vos missions (taux de complétion > 95%), "
            "4) Accumulez des avis positifs. "
            "Le matching prend aussi en compte votre proximité géographique avec la demande."
        ),
        "keywords": ["pas en premier", "classement", "recommandation", "pourquoi pas choisi"],
    },
    {
        "id":       "a_verify_1",
        "role":     "artisan",
        "intent":   "verify_profile",
        "question": "Comment faire vérifier mon profil ?",
        "answer":   (
            "Allez dans Paramètres → 'Vérification de profil'. "
            "Soumettez : votre CIN (carte d'identité nationale), "
            "un justificatif de votre métier (diplôme, attestation de formation, "
            "ou certificat professionnel), et optionnellement des photos de réalisations. "
            "La vérification est traitée sous 72h. "
            "Le badge vérifié augmente fortement votre score et votre visibilité."
        ),
        "keywords": ["vérification", "badge vérifié", "documents", "certifié", "valider profil", "CIN"],
    },
    {
        "id":       "a_availability_1",
        "role":     "artisan",
        "intent":   "availability",
        "question": "Comment gérer mes disponibilités ?",
        "answer":   (
            "Allez dans votre tableau de bord → 'Mes disponibilités'. "
            "Définissez vos créneaux disponibles semaine par semaine. "
            "Pour les demandes urgentes, le système filtre automatiquement "
            "les artisans disponibles rapidement. "
            "Mettez-vous indisponible pendant vos congés pour ne pas recevoir "
            "de notifications inutiles."
        ),
        "keywords": ["disponibilité", "horaires", "créneaux", "agenda", "congés", "calendrier"],
    },

    # ═══════════════════════════════════════════════════════════════════
    # ADMIN — Modération et supervision
    # ═══════════════════════════════════════════════════════════════════
    {
        "id":       "ad_validate_1",
        "role":     "admin",
        "intent":   "validate_artisan",
        "question": "Comment valider un profil artisan ?",
        "answer":   (
            "Administration → 'Profils en attente'. "
            "Sélectionnez un profil → examinez les documents soumis (CIN, justificatif métier). "
            "Vérifiez l'authenticité des documents, la cohérence des informations, "
            "et l'absence de doublons. "
            "Cliquez 'Valider' pour activer le profil ou 'Rejeter' en précisant le motif. "
            "L'artisan est notifié automatiquement dans les deux cas."
        ),
        "keywords": ["valider artisan", "profil en attente", "vérification documents", "modération"],
    },
    {
        "id":       "ad_complaint_1",
        "role":     "admin",
        "intent":   "manage_complaint",
        "question": "Comment traiter une réclamation ?",
        "answer":   (
            "Administration → 'Réclamations' → sélectionnez la réclamation. "
            "Lisez le dossier complet (description, photos, échanges de messages). "
            "Contactez les deux parties si nécessaire via la messagerie admin. "
            "Choisissez une décision : résolu, rejeté, ou remboursement partiel. "
            "Documentez votre décision avec un motif. Les deux parties sont notifiées."
        ),
        "keywords": ["traiter réclamation", "résoudre litige", "décision admin", "clôturer"],
    },
    {
        "id":       "ad_ai_1",
        "role":     "admin",
        "intent":   "monitor_ai",
        "question": "Comment superviser les modules IA ?",
        "answer":   (
            "Administration → 'Supervision IA'. Vous y trouvez : "
            "1) Pricing : MAE actuelle, coverage des prédictions, évolution hebdomadaire, "
            "2) Matching : AUC, Precision@5, artisans les mieux rankés par catégorie, "
            "3) Chatbot : intents les plus fréquents, questions sans réponse, taux de résolution. "
            "Si un indicateur passe en rouge → un retraining automatique est déclenché. "
            "Vous pouvez aussi ajuster les pondérations α/β/γ du matching."
        ),
        "keywords": ["supervision IA", "métriques", "monitoring", "performance modèles", "retraining"],
    },

    # ═══════════════════════════════════════════════════════════════════
    # GÉNÉRAL — Compte et sécurité
    # ═══════════════════════════════════════════════════════════════════
    {
        "id":       "gen_account_1",
        "role":     "tous",
        "intent":   "account",
        "question": "Comment réinitialiser mon mot de passe ?",
        "answer":   (
            "Sur la page de connexion, cliquez 'Mot de passe oublié'. "
            "Entrez votre adresse email → vous recevrez un lien de réinitialisation valable 1h. "
            "Si vous ne recevez pas l'email, vérifiez vos spams "
            "ou contactez le support via le formulaire de contact."
        ),
        "keywords": ["mot de passe", "oublié", "réinitialiser", "connexion impossible", "login"],
    },
    {
        "id":       "gen_account_2",
        "role":     "tous",
        "intent":   "account",
        "question": "Comment supprimer mon compte ?",
        "answer":   (
            "Paramètres → 'Mon compte' → 'Supprimer mon compte'. "
            "Attention : cette action est irréversible. Toutes vos données, "
            "demandes, offres et historique seront supprimés. "
            "Si vous avez des missions en cours, vous devez les terminer ou les transférer "
            "avant de pouvoir supprimer votre compte."
        ),
        "keywords": ["supprimer compte", "désinscrire", "fermer compte", "quitter"],
    },
    {
        "id":       "gen_payment_1",
        "role":     "tous",
        "intent":   "payment",
        "question": "AtlasFix est-il gratuit ?",
        "answer":   (
            "L'inscription et l'utilisation de base sont gratuites pour les clients. "
            "Les artisans bénéficient d'un accès gratuit pour les X premières offres, "
            "puis un abonnement mensuel leur permet d'envoyer des offres illimitées. "
            "AtlasFix ne prend pas de commission sur les transactions entre clients et artisans."
        ),
        "keywords": ["gratuit", "payant", "prix abonnement", "commission", "coût plateforme"],
    },
]

# Tous les intents possibles (pour l'entraînement du classifieur)
ALL_INTENTS = list(set(doc["intent"] for doc in KNOWLEDGE_BASE))

# Tous les rôles
ALL_ROLES = ["client", "artisan", "admin", "tous"]

if __name__ == "__main__":
    print(f"Base de connaissances : {len(KNOWLEDGE_BASE)} documents")
    print(f"Intents : {ALL_INTENTS}")
    from collections import Counter
    roles = Counter(doc["role"] for doc in KNOWLEDGE_BASE)
    print(f"Répartition par rôle : {dict(roles)}")
