# --- app.py ---

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.emotion_classifier import EmotionClassifier
from src.language_detector import LanguageDetector
from src.intent_classifier import IntentClassifier
from src.rag_pipeline import is_crisis, retrieve_chunks, build_prompt, CRISIS_RESPONSE
from config import (
    GROQ_MODEL,
    GROQ_API_KEY,
    EMOTION_CONFIDENCE_THRESHOLD,
    TOP_K_CHUNKS,
    MAX_TOKENS_TRANSLATION,
    MAX_TOKENS_RESPONSE,
    TEMPERATURE_TRANSLATION,
    TEMPERATURE_GENERATION
)
import asyncio
from groq import Groq

lang_detector = None
emotion_detector = None
intent_detector = None
groq_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global lang_detector, emotion_detector, intent_detector, groq_client

    # 1. Load Language Detector
    try:
        lang_detector = LanguageDetector()
    except Exception as e:
        print(f"Language detector load failed, using mock: {e}")
        class MockLanguageDetector:
            def predict(self, text: str) -> str: return "en"
        lang_detector = MockLanguageDetector()

    # 2. Load Emotion Classifier
    try:
        emotion_detector = EmotionClassifier()
    except Exception as e:
        print(f"Emotion classifier load failed, using mock: {e}")
        class MockEmotionClassifier:
            def predict(self, text: str) -> dict: return {"label": "neutral", "score": 1.0}
        emotion_detector = MockEmotionClassifier()

    # 3. Create the shared Groq Client first
    groq_client = Groq(api_key=GROQ_API_KEY)

    # 4. Load Intent Classifier (Sharing client)
    try:
        intent_detector = IntentClassifier(groq_client=groq_client)
    except Exception as e:
        print(f"Intent classifier load failed: {e}")
        raise

    # 5. FIX: Pre-warm the RAG Pipeline's lazy-loaders (Embedding Model + Qdrant connection)
    from src.rag_pipeline import _get_embedding_model, _get_qdrant_client, _get_groq_client
    print("Pre-warming embedding model and database clients...")
    _get_embedding_model()
    _get_qdrant_client()
    _get_groq_client(groq_client=groq_client) # Pass shared client
    print("All models pre-warmed and ready.")

    yield

app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    text: str

@app.post("/chat")
async def process_chat(request: QueryRequest):
    query = request.text
    if not query.strip():
        raise HTTPException(status_code=400, detail="Text query cannot be empty.")

    # Step 1: Detect Language
    language = lang_detector.predict(query)
    # Translate query to English for retrieval if needed
    retrieval_query = query
    if language != "en":
        translation_prompt = f"Translate the following text to English. Output ONLY the translation, nothing else:\n{query}"
        translation = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": translation_prompt}],
            temperature=TEMPERATURE_TRANSLATION,
            max_tokens=MAX_TOKENS_TRANSLATION,
        )
        retrieval_query = translation.choices[0].message.content.strip()

    # Step 2: Detect Emotion
    emotion_result = emotion_detector.predict(retrieval_query)
    emotion_label = emotion_result["label"]
    emotion_score = emotion_result["score"]

    if emotion_score < EMOTION_CONFIDENCE_THRESHOLD:
        emotion_label = "uncertain"
        prompt_emotion = None  # LLM will infer emotion from context
    else:
        prompt_emotion = emotion_label

    # Step 3: Classify Intent
    intent = intent_detector.classify_intent(retrieval_query)
    print(f"Detected intent: {intent} | Detected emotion: {emotion_label} ({emotion_score:.3f}) | Detected language: {language}")

    async def response_generator():
        # Emit metadata first
        metadata = {
            "type": "metadata",
            "detected_language": language,
            "detected_emotion": emotion_label,
            "detected_emotion_score": round(emotion_score, 3),
            "detected_intent": intent
        }
        yield json.dumps(metadata) + "\n"

        # Route based on intent
        if intent == "greeting":
            bot_response = "Hello! I am your supportive assistant. How are you feeling today?"
            yield json.dumps({"type": "chunk", "text": bot_response}) + "\n"

        elif intent == "goodbye":
            bot_response = "Goodbye. Take care of yourself, and remember that I am always here if you need to talk."
            yield json.dumps({"type": "chunk", "text": bot_response}) + "\n"

        elif intent == "gratitude":
            bot_response = "You are very welcome. I'm here to support you whenever you need."
            yield json.dumps({"type": "chunk", "text": bot_response}) + "\n"

        elif intent == "out_of_scope":
            bot_response = (
                "I apologize, but I am programmed specifically to assist with mental health concerns "
                "like stress, anxiety, and depression. I am unable to answer general questions outside of those topics."
            )
            yield json.dumps({"type": "chunk", "text": bot_response}) + "\n"

        else:
            if is_crisis(retrieval_query):
                yield json.dumps({"type": "chunk", "text": CRISIS_RESPONSE}) + "\n"
                return

            # Retrieve chunks and build prompts (from rag_pipeline.py functions)
            chunks = retrieve_chunks(retrieval_query, top_k=TOP_K_CHUNKS)
            system_prompt, user_prompt = build_prompt(query, chunks, emotion=prompt_emotion)

            # Emit retrieved chunks so the UI can display sources
            yield json.dumps({"type": "chunks", "data": chunks}) + "\n"

            # Stream the Groq response token by token
            try:
                stream = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE_GENERATION,
                    max_tokens=MAX_TOKENS_RESPONSE,
                    stream=True,  # FIX: enable streaming
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield json.dumps({"type": "chunk", "text": delta}) + "\n"
                        await asyncio.sleep(0)  # yield control to the event loop between chunks
            except Exception as e:
                yield json.dumps({"type": "chunk", "text": f"Error generating response: {e}"}) + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")