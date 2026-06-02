import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq
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

# Global module level client/model instantiations (Removes lazy loaders)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
groq_client = Groq(api_key=GROQ_API_KEY)

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
    text_lower = text.lower()
    return any(kw in text_lower for kw in CRISIS_KEYWORDS)

def retrieve_chunks(question: str, top_k: int = TOP_K_CHUNKS) -> list[dict]:
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

def build_prompt(question: str, chunks: list[dict], emotion: str = None, language: str = None) -> tuple[str, str]:
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

def generate_answer(system_prompt: str, user_prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=TEMPERATURE_GENERATION,
        max_tokens=MAX_TOKENS_RESPONSE,
    )
    return response.choices[0].message.content

def mental_health_chatbot(question: str, emotion: str = None) -> str:
    if is_crisis(question):
        return CRISIS_RESPONSE

    chunks = retrieve_chunks(question, top_k=TOP_K_CHUNKS)
    system_prompt, user_prompt = build_prompt(question, chunks, emotion=emotion)
    return generate_answer(system_prompt, user_prompt)


if __name__ == "__main__":
    print("\n--- Test 1: Anxiety ---")
    print(mental_health_chatbot("I keep having panic attacks and I cannot calm down.", emotion="anxiety"))

    print("\n--- Test 2: Depression ---")
    print(mental_health_chatbot("I feel hopeless and have no energy.", emotion="sadness"))

    print("\n--- Test 3: Crisis ---")
    print(mental_health_chatbot("I want to kill myself."))