"""
rag_runtime.py
--------------
Runtime RAG pipeline for Module 4 - Q&A Mental Health Chatbot.

This script ONLY does retrieval and generation.
It connects to the existing Qdrant collection that was populated by 04_QA_RAG.ipynb.
It never recreates the collection, re-embeds the dataset, or re-uploads vectors.
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# ── 1. Load API keys ──────────────────────────────────────────────────────────

load_dotenv()

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

if not all([QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY]):
    raise EnvironmentError(
        "Missing one or more API keys. "
        "Make sure QDRANT_URL, QDRANT_API_KEY, and GROQ_API_KEY are set in your .env file."
    )

# ── 2. Load the embedding model ───────────────────────────────────────────────
# Must be the same model used during indexing in 04_QA_RAG.ipynb.
# Using a different model would produce incompatible vectors.

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")

# ── 3. Connect to Qdrant Cloud ────────────────────────────────────────────────
# Connect to the existing collection. Do NOT create or recreate it here.

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "mental_health_chunks"

# ── 4. Connect to Groq ────────────────────────────────────────────────────────

groq_client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "openai/gpt-oss-20b"

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
    """Return True if the text contains any obvious crisis phrase."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in CRISIS_KEYWORDS)

# ── 6. Retrieval function ─────────────────────────────────────────────────────

def retrieve_chunks(question: str, top_k: int = 3) -> list[dict]:
    """
    Embed the question and retrieve the top-k most similar chunks from Qdrant.

    Returns a list of dicts with keys: context, response, score.
    """
    query_vector = embedding_model.encode(question).tolist()

    search_result = qdrant_client.query_points(
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

def build_prompt(question: str, chunks: list[dict], emotion: str = None) -> tuple[str, str]:
    """
    Build the system and user prompts from the retrieved chunks.

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

    system_prompt = (
        "You are a supportive mental health assistant. "
        "Use only the retrieved context to answer the user. "
        "Be empathetic, calm, and clear. "
        "Do not diagnose the user. "
        "Do not claim to be a therapist or doctor. "
        "If the context is not enough, say that clearly and give general supportive guidance."
    )

    user_prompt = (
        f"{emotion_note}"
        f"Retrieved context from the counseling knowledge base:\n\n"
        f"{retrieved_text}"
        f"User question:\n{question}\n\n"
        f"Answer:"
    )

    return system_prompt, user_prompt

# ── 8. Answer generation ──────────────────────────────────────────────────────

def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """Call Groq and return the generated answer string."""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    return response.choices[0].message.content

# ── 9. Final chatbot function ─────────────────────────────────────────────────

def mental_health_chatbot(question: str, emotion: str = None) -> str:
    """
    Main entry point for the runtime chatbot.

    Args:
        question: the user's message
        emotion:  optional emotion label from Module 2 (e.g. 'anxiety', 'sadness')

    Returns:
        A crisis safety message if crisis language is detected,
        otherwise a RAG-generated answer grounded in the counseling dataset.
    """
    # Step 1 - Crisis check
    if is_crisis(question):
        return CRISIS_RESPONSE

    # Step 2 - Retrieve relevant chunks from Qdrant
    chunks = retrieve_chunks(question, top_k=3)

    # Step 3 - Build prompts
    system_prompt, user_prompt = build_prompt(question, chunks, emotion=emotion)

    # Step 4 - Generate answer
    answer = generate_answer(system_prompt, user_prompt)

    return answer


# ── 10. Quick local test (only runs when script is called directly) ───────────

if __name__ == "__main__":
    print("\n--- Test 1: Anxiety ---")
    print(mental_health_chatbot("I keep having panic attacks and I cannot calm down.", emotion="anxiety"))

    print("\n--- Test 2: Depression ---")
    print(mental_health_chatbot("I feel hopeless and have no energy.", emotion="sadness"))

    print("\n--- Test 3: Crisis ---")
    print(mental_health_chatbot("I want to kill myself."))
