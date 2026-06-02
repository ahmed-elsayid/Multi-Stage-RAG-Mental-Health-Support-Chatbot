import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from groq import Groq

from src.emotion_classifier import EmotionClassifier, MockEmotionClassifier
from src.language_detector import LanguageDetector, MockLanguageDetector
from src.intent_classifier import IntentClassifier
from src.rag_pipeline import is_crisis, retrieve_chunks, build_prompt, CRISIS_RESPONSE
from src import rag_pipeline
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
        print(f"Failed to load language detector, falling back to mock: {e}")
        lang_detector = MockLanguageDetector()

    # 2. Load Emotion Classifier
    try:
        emotion_detector = EmotionClassifier()
    except Exception as e:
        print(f"Failed to load emotion classifier, falling back to mock: {e}")
        emotion_detector = MockEmotionClassifier()

    # 3. Create global shared Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)

    # 4. Load Intent Classifier with shared client
    try:
        intent_detector = IntentClassifier(groq_client=groq_client)
    except Exception as e:
        print(f"Intent classifier load failed: {e}")
        raise

    # 5. Share global client reference with the RAG pipeline module
    rag_pipeline.groq_client = groq_client

    yield


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    text: str


@app.post("/chat")
async def process_chat(request: QueryRequest):
    query = request.text
    if not query.strip():
        raise HTTPException(status_code=400, detail="Text query cannot be empty.")

    language = lang_detector.predict(query)
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

    emotion_result = emotion_detector.predict(retrieval_query)
    emotion_label = emotion_result["label"]
    emotion_score = emotion_result["score"]

    if emotion_score < EMOTION_CONFIDENCE_THRESHOLD:
        emotion_label = "uncertain"
        prompt_emotion = None
    else:
        prompt_emotion = emotion_label

    intent = intent_detector.classify_intent(retrieval_query)

    async def response_generator():
        metadata = {
            "type": "metadata",
            "detected_language": language,
            "detected_emotion": emotion_label,
            "detected_emotion_score": round(emotion_score, 3),
            "detected_intent": intent
        }
        yield json.dumps(metadata) + "\n"

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

            chunks = retrieve_chunks(retrieval_query, top_k=TOP_K_CHUNKS)
            system_prompt, user_prompt = build_prompt(query, chunks, emotion=prompt_emotion)

            yield json.dumps({"type": "chunks", "data": chunks}) + "\n"

            try:
                stream = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE_GENERATION,
                    max_tokens=MAX_TOKENS_RESPONSE,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield json.dumps({"type": "chunk", "text": delta}) + "\n"
                        await asyncio.sleep(0)
            except Exception as e:
                yield json.dumps({"type": "chunk", "text": f"Error generating response: {e}"}) + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")