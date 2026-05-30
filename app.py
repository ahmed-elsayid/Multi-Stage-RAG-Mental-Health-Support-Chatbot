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

import asyncio
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

lang_detector = None
emotion_detector = None
intent_detector = None
groq_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global lang_detector, emotion_detector, intent_detector, groq_client

    # 1. Load Language Detector with fallback
    try:
        lang_detector = LanguageDetector()
    except Exception as e:
        print(f"Language detector load failed, using mock: {e}")
        class MockLanguageDetector:
            def predict(self, text: str) -> str:
                return "en"
        lang_detector = MockLanguageDetector()

    # 2. Load Emotion Classifier with fallback
    try:
        emotion_detector = EmotionClassifier()
    except Exception as e:
        print(f"Emotion classifier load failed, using mock: {e}")
        class MockEmotionClassifier:
            def predict(self, text: str) -> dict:
                return {"label": "neutral", "score": 1.0}
        emotion_detector = MockEmotionClassifier()

    # 3. Load Intent Classifier
    # FIX: was trying to instantiate a nonexistent RAGPipeline class alongside this
    try:
        intent_detector = IntentClassifier()
    except Exception as e:
        print(f"Intent classifier load failed: {e}")
        raise

    # 4. Initialize Groq client for streaming
    # FIX: streaming is handled here directly instead of through a nonexistent class method
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    text: str


GROQ_MODEL = "llama-3.1-8b-instant"


@app.post("/chat")
async def process_chat(request: QueryRequest):
    query = request.text
    if not query.strip():
        raise HTTPException(status_code=400, detail="Text query cannot be empty.")

    # Step 1: Detect Language
    language = lang_detector.predict(query)

    # Step 2: Detect Emotion
    emotion_result = emotion_detector.predict(query)
    emotion_label = emotion_result["label"]
    emotion_score = emotion_result["score"]

    if emotion_score < 0.6:
        emotion_label = "uncertain"

    # Step 3: Classify Intent
    intent = intent_detector.classify_intent(query)

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
            if is_crisis(query):
                yield json.dumps({"type": "chunk", "text": CRISIS_RESPONSE}) + "\n"
                return

            # Retrieve chunks and build prompts (from rag_pipeline.py functions)
            chunks = retrieve_chunks(query, top_k=3)
            system_prompt, user_prompt = build_prompt(query, chunks, emotion=emotion_label)

            # Stream the Groq response token by token
            try:
                stream = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=800,
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