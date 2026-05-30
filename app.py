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
from rag_pipeline import RAGPipeline

lang_detector = None
emotion_detector = None
intent_detector = None
rag_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global lang_detector, emotion_detector, intent_detector, rag_pipeline

    # 1. Load Language Detector with custom fallback support
    try:
        lang_detector = LanguageDetector()
    except Exception as e:
        print(f"Language detector load failed, using mock interface: {e}")
        class MockLanguageDetector:
            def predict(self, text: str) -> str:
                return "en" # Fallback to English safely
        lang_detector = MockLanguageDetector()

    # 2. Load Emotion Classifier with fallback support
    try:
        emotion_detector = EmotionClassifier()
    except Exception as e:
        print(f"Emotion classifier load failed, using mock interface: {e}")
        class MockEmotionClassifier:
            def predict(self, text: str) -> dict:
                return {"label": "neutral", "score": 1.0}
        emotion_detector = MockEmotionClassifier()

    # 3. Load Intent and RAG engines
    try:
        intent_detector = IntentClassifier()
        rag_pipeline = RAGPipeline()
    except Exception as e:
        print(f"Failed to initialize core classifiers: {e}")

    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    text: str


@app.post("/chat")
async def process_chat(request: QueryRequest):
    query = request.text
    if not query.strip():
        raise HTTPException(status_code=400, detail="Text query cannot be empty.")

    # Step 1: Detect Language (returns standard ISO string via your class wrapper)
    language = lang_detector.predict(query)

    # Step 2: Detect Emotion (returns dict with label + confidence score)
    emotion_result = emotion_detector.predict(query)
    emotion_label = emotion_result["label"]
    emotion_score = emotion_result["score"]

    # CRITICAL FALLBACK: if model is uncertain, mark as ambiguous for safer RAG handling
    if emotion_score < 0.6:
        emotion_label = "uncertain"

    # Step 3: Classify Intent
    intent = intent_detector.classify_intent(query)

    async def response_generator():
        # Prepare metadata payload for the frontend
        metadata = {
            "type": "metadata",
            "detected_language": language,
            "detected_emotion": emotion_label,
            "detected_emotion_score": round(emotion_score, 3),
            "detected_intent": intent
        }
        yield json.dumps(metadata) + "\n"

        # Routing Logic based on classified intent
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
            bot_response = "I apologize, but I am programmed specifically to assist with mental health concerns like stress, anxiety, and depression. I am unable to answer general questions outside of those topics."
            yield json.dumps({"type": "chunk", "text": bot_response}) + "\n"

        else:  # asking_mental_health_question (Triggers streaming RAG pipeline)
            context = rag_pipeline.retrieve_context(query)
            for chunk in rag_pipeline.generate_response_stream(query, context, emotion_label, language):
                yield json.dumps({"type": "chunk", "text": chunk}) + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")