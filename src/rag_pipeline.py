"""
rag_pipeline.py
---------------
Runtime RAG pipeline for the Mental Health Chatbot.

Handles retrieval from Qdrant and generation via Groq.
Streaming is handled in app.py using the Groq client directly —
this module exposes the retrieval + prompt-building helpers,
plus a non-streaming generate_answer() for standalone testing.
"""

import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq
# ── 1. Load API keys ──────────────────────────────────────────────────────────
from config import (
    GROQ_MODEL,
    GROQ_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_CHUNKS,
    MAX_TOKENS_RESPONSE,
    TEMPERATURE_GENERATION
)

if not all([QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY]):
    import warnings
    warnings.warn(
        "Missing one or more API keys (QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY). "
        "Retrieval will be unavailable until the .env file is configured.",
        RuntimeWarning,
        stacklevel=2,
    )

# ── 2. Load embedding model (lazy) ───────────────────────────────────────────

_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model ready.")
    return _embedding_model

# ── 3. Connect to Qdrant (lazy) ───────────────────────────────────────────────

_qdrant_client = None

def _get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


# ── 4. Connect to Groq (lazy) ─────────────────────────────────────────────────

_groq_client = None

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY must be set in .env")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

# ── 5. Crisis safety check ────────────────────────────────────────────────────

CRISIS_KEYWORDS = [
    "suicide",
    "kill myself",
    "end my life",
    "self harm",
    "self-harm",
    "hurt myself",
    "don't want to live",
    "do not want to live",
    "want to die",
    "take my own life",
    "no reason to live",
]

CRISIS_RESPONSE = (
    "I hear you, and I am very concerned about your safety right now. "
    "Please reach out to a crisis support line immediately:\n\n"
    "- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/\n"
    "- Crisis Text Line (US): Text HOME to 741741\n"
    "- Befrienders Worldwide: https://www.befrienders.org\n\n"
    "You are not alone. A trained counselor is available to talk with you right now. "
    "If you are in immediate danger, please call your local emergency services."
)

def is_crisis(text: str) -> bool:
    """Return True if the text contains any crisis phrase."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in CRISIS_KEYWORDS)

# ── 6. Retrieval ──────────────────────────────────────────────────────────────

def retrieve_chunks(question: str, top_k: int = TOP_K_CHUNKS) -> list[dict]:
    """
    Embed the question and retrieve the top-k most similar chunks from Qdrant.
    Returns a list of dicts with keys: context, response, score.
    """
    query_vector = _get_embedding_model().encode(question).tolist()

    search_result = _get_qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    chunks = []
    for hit in search_result.points:
        chunks.append({
            "context": hit.payload.get("context", ""),
            "response": hit.payload.get("response", ""),
            "score": hit.score,
        })

    return chunks

# ── 7. Prompt builder ─────────────────────────────────────────────────────────

def build_prompt(question: str, chunks: list[dict], emotion: str = None, language: str = None) -> tuple[str, str]:
    """
    Build system and user prompts from the retrieved chunks.
    Returns (system_prompt, user_prompt).
    """
    retrieved_text = ""
    for i, chunk in enumerate(chunks):
        retrieved_text += (
            f"[Chunk {i + 1}] (similarity: {chunk['score']:.3f})\n"
            f"Original concern:\n{chunk['context']}\n\n"
            f"Counselor response:\n{chunk['response']}\n\n"
            f"---\n\n"
        )

    emotion_note = f"The detected emotion is: {emotion}.\n" if emotion else ""
    language_note = (
        f"IMPORTANT: You MUST reply in the same language as the user's message. "
        f"The detected language code is: '{language}'. "
        f"Do not switch to English or any other language under any circumstances.\n"
    ) if language else ""

    system_prompt = (
        "You are a supportive mental health assistant. "
        "Use only the retrieved context to answer the user. "
        "Be empathetic, calm, and clear. "
        "Carefully read the user's tone and adapt your response emotionally even if no emotion label is provided. "
        "Do not diagnose the user. "
        "Do not claim to be a therapist or doctor. "
        "If the context is not enough, say that clearly and give general supportive guidance."
    )

    user_prompt = (
        f"{language_note}"
        f"{emotion_note}"
        f"Retrieved context from the counseling knowledge base:\n\n"
        f"{retrieved_text}"
        f"User question:\n{question}\n\n"
        f"Answer:"
    )

    return system_prompt, user_prompt

# ── 8. Non-streaming answer generation (for standalone testing only) ──────────
# NOTE: app.py uses Groq's streaming API directly for real-time responses.
# This function is kept for __main__ testing convenience only.

def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """Call Groq and return the full generated answer (non-streaming)."""
    response = _get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=TEMPERATURE_GENERATION,
        max_tokens=MAX_TOKENS_RESPONSE,
    )
    return response.choices[0].message.content

# ── 9. Convenience wrapper (for testing) ──────────────────────────────────────

def mental_health_chatbot(question: str, emotion: str = None) -> str:
    """
    Non-streaming entry point for local testing.
    The real app uses streaming via app.py directly.
    """
    if is_crisis(question):
        return CRISIS_RESPONSE

    chunks = retrieve_chunks(question, top_k=TOP_K_CHUNKS)
    system_prompt, user_prompt = build_prompt(question, chunks, emotion=emotion)
    return generate_answer(system_prompt, user_prompt)


# ── 10. Quick local test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n--- Test 1: Anxiety ---")
    print(mental_health_chatbot("I keep having panic attacks and I cannot calm down.", emotion="anxiety"))

    print("\n--- Test 2: Depression ---")
    print(mental_health_chatbot("I feel hopeless and have no energy.", emotion="sadness"))

    print("\n--- Test 3: Crisis ---")
    print(mental_health_chatbot("I want to kill myself."))