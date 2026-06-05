"""
chatbot/chatbot_service.py
===========================
Service complet du chatbot AtlasFix.

Pipeline : message -> intent classifier -> ChromaDB retrieval -> Ollama -> reponse

Comment lancer Ollama sur ton PC Windows :
  1. Telecharge Ollama : https://ollama.ai/download
  2. Installe et lance : ollama pull mistral
  3. Verifie : ollama run mistral "Bonjour"
  4. Lance le service : ollama serve  (http://localhost:11434)
"""

import os, sys, json, joblib, logging, requests
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chatbot.data.knowledge_base import KNOWLEDGE_BASE

logger = logging.getLogger(__name__)

OLLAMA_URL     = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "mistral")
EMBED_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DB_PATH = str(ROOT / "chatbot" / "vectordb")
MODEL_PATH     = ROOT / "chatbot" / "models" / "intent_classifier.pkl"

SYSTEM_PROMPTS = {
    "client": (
        "Tu es l'assistant AtlasFix pour les clients. "
        "Tu aides a publier des demandes, comprendre les offres et resoudre les problemes. "
        "Reponds en francais, de facon concise. Base-toi UNIQUEMENT sur le contexte fourni."
    ),
    "artisan": (
        "Tu es l'assistant AtlasFix pour les artisans. "
        "Tu aides a envoyer des offres, ameliorer le profil et developper l'activite. "
        "Reponds en francais avec des conseils pratiques."
    ),
    "admin": (
        "Tu es l'assistant AtlasFix pour les administrateurs. "
        "Tu aides a moderer la plateforme et superviser les modules IA. "
        "Reponds de facon factuelle."
    ),
}

# ─── Intent Classifier ────────────────────────────────────────────────────────

class IntentClassifier:
    def __init__(self):
        try:
            self._bundle = joblib.load(MODEL_PATH)
            logger.info(f"Intent classifier charge ({self._bundle['encoder_type']})")
        except FileNotFoundError:
            logger.warning("intent_classifier.pkl non trouve")
            self._bundle = None

    def classify(self, text: str) -> tuple:
        if self._bundle is None:
            return "general_faq", 0.3
        proba = self._bundle["pipeline"].predict_proba([text])[0]
        idx   = proba.argmax()
        return self._bundle["classes"][idx], round(float(proba[idx]), 3)

# ─── Vector Retriever ────────────────────────────────────────────────────────

class VectorRetriever:
    def __init__(self):
        self._collection = None
        self._encoder    = None
        self._mode       = "keyword"
        self._init()

    def _init(self):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            self._encoder    = SentenceTransformer(EMBED_MODEL)
            client           = chromadb.PersistentClient(path=VECTOR_DB_PATH)
            try:
                self._collection = client.get_collection("atlasfix_kb")
            except Exception:
                self._collection = client.create_collection("atlasfix_kb")
                self._index_kb()
            self._mode = "chroma"
            logger.info(f"ChromaDB pret : {self._collection.count()} docs")
        except Exception as e:
            logger.warning(f"ChromaDB non dispo : {e} — mode keyword")

    def _index_kb(self):
        texts, ids, metas = [], [], []
        for doc in KNOWLEDGE_BASE:
            texts.append(f"{doc['question']} {doc['answer']}")
            ids.append(doc["id"])
            metas.append({"role": doc["role"], "intent": doc["intent"], "question": doc["question"]})
        embeddings = self._encoder.encode(texts).tolist()
        self._collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metas)
        logger.info(f"{len(texts)} documents indexes")

    def retrieve(self, query: str, role: str, top_k: int = 2) -> list:
        if self._mode == "chroma":
            try:
                embedding = self._encoder.encode([query]).tolist()[0]
                where = {"role": {"$in": [role, "tous"]}}
                results = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=min(top_k, self._collection.count()),
                    where=where,
                )
                docs = []
                for i, meta in enumerate(results["metadatas"][0]):
                    kb_doc = next((d for d in KNOWLEDGE_BASE if d["id"] == results["ids"][0][i]), None)
                    docs.append({
                        "question": meta.get("question", ""),
                        "answer":   kb_doc["answer"] if kb_doc else "",
                        "intent":   meta.get("intent", ""),
                    })
                return docs
            except Exception as e:
                logger.warning(f"ChromaDB query error: {e}")

        # Fallback keyword
        query_lower = query.lower()
        scored = []
        for doc in KNOWLEDGE_BASE:
            if doc["role"] not in (role, "tous"):
                continue
            score = sum(1 for kw in doc["keywords"] if kw in query_lower)
            score += sum(1 for w in query_lower.split() if w in doc["question"].lower())
            scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k] if scored else [(0, KNOWLEDGE_BASE[0])]
        return [{"question": d["question"], "answer": d["answer"], "intent": d["intent"]} for _, d in top]

    def add_document(self, doc_id: str, question: str, answer: str, role: str, intent: str) -> bool:
        """
        Ajoute un nouveau document a la KB — appele quand un admin cree une nouvelle FAQ.
        C'est le mecanisme principal d'amelioration continue du chatbot.
        """
        if self._mode != "chroma":
            return False
        try:
            text      = f"{question} {answer}"
            embedding = self._encoder.encode([text]).tolist()[0]
            self._collection.add(
                documents=[text], embeddings=[embedding], ids=[doc_id],
                metadatas=[{"role": role, "intent": intent, "question": question}]
            )
            logger.info(f"Document ajoute : {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur ajout doc : {e}")
            return False

# ─── Ollama LLM ──────────────────────────────────────────────────────────────

class OllamaLLM:
    def is_available(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, system_prompt: str, context_docs: list,
                 user_message: str, history: list) -> str:
        context = "\n\n".join([
            f"Q: {doc['question']}\nR: {doc['answer']}"
            for doc in context_docs
        ])
        history_text = ""
        for msg in history[-6:]:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        prompt = (
            f"{system_prompt}\n\n"
            f"=== DOCUMENTATION ATLASFIX ===\n{context}\n\n"
            f"=== HISTORIQUE ===\n{history_text}\n"
            f"=== QUESTION ===\nUtilisateur: {user_message}\n\n"
            f"Assistant:"
        )
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=30,
            )
            return response.json()["response"].strip()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return None

# ─── Main ChatbotService ─────────────────────────────────────────────────────

class ChatbotService:
    """
    Orchestre les 3 composants : intent classifier + retriever + LLM.
    Expose une seule methode : respond()
    """

    def __init__(self):
        self.classifier = IntentClassifier()
        self.retriever  = VectorRetriever()
        self.llm        = OllamaLLM()
        self._sessions  = {}   # remplacer par Redis en prod
        logger.info(f"ChatbotService pret — LLM: {'Ollama' if self.llm.is_available() else 'RAG only'}")

    def respond(self, message: str, user_role: str,
                session_id: str, user_id: int) -> dict:
        """
        Point d'entree unique — appele par le router FastAPI.
        Retourne la reponse + metadata.
        """
        # 1. Classifier l'intent
        intent, confidence = self.classifier.classify(message)

        # 2. Recuperer les documents pertinents (RAG)
        context_docs = self.retriever.retrieve(message, user_role, top_k=2)

        # 3. Historique de session
        history = self._sessions.get(session_id, [])

        # 4. Generer la reponse
        system_prompt = SYSTEM_PROMPTS.get(user_role, SYSTEM_PROMPTS["client"])

        if self.llm.is_available():
            # Mode complet : Ollama
            response = self.llm.generate(system_prompt, context_docs, message, history)
            mode = "llm"
            if response is None:
                response = context_docs[0]["answer"] if context_docs else self._fallback()
                mode = "rag"
        else:
            # Mode RAG only : retourne le doc le plus pertinent
            response = context_docs[0]["answer"] if context_docs else self._fallback()
            mode = "rag"

        # 5. Mettre a jour l'historique
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": response})
        self._sessions[session_id] = history[-10:]

        return {
            "response":   response,
            "intent":     intent,
            "confidence": confidence,
            "mode":       mode,
            "session_id": session_id,
            "context_used": len(context_docs),
        }

    def add_faq(self, question: str, answer: str, role: str, intent: str) -> bool:
        """
        Ajoute une FAQ depuis le dashboard admin.
        C'est le mecanisme d'amelioration continue du chatbot dans le temps.
        """
        doc_id = f"admin_{hash(question) % 100000}"
        return self.retriever.add_document(doc_id, question, answer, role, intent)

    def get_unanswered_questions(self) -> list:
        """
        Retourne les questions avec faible confiance.
        L'admin peut les consulter et ajouter les reponses manquantes.
        """
        return [
            s for sess in self._sessions.values()
            for i, s in enumerate(sess)
            if s["role"] == "user" and i + 1 < len(sess)
        ]

    @staticmethod
    def _fallback() -> str:
        return (
            "Je n'ai pas trouve de reponse precise a votre question. "
            "Pouvez-vous reformuler ou contacter notre support via le formulaire de contact ?"
        )
