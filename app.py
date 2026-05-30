"""
app.py
------
FastAPI backend for the Multi-Stage RAG Mental Health Support Chatbot.

Endpoints:
  GET  /        - health check
  POST /chat    - Module 4 Q&A RAG endpoint
"""

from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_runtime import mental_health_chatbot

app = FastAPI(
    title="Multi-Stage RAG Mental Health Support Chatbot",
    description="RAG-based mental health support using Qdrant + Groq.",
    version="1.0.0",
)

# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    emotion: str = None   # optional, passed in from Module 2 emotion classifier

class ChatResponse(BaseModel):
    answer: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "project": "Multi-Stage RAG Mental Health Support Chatbot"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Module 4 - Q&A RAG endpoint.

    Accepts a user question and an optional emotion label from Module 2.
    Returns a grounded empathetic answer, or a safety message if crisis language is detected.
    """
    answer = mental_health_chatbot(
        question=request.question,
        emotion=request.emotion,
    )
    return ChatResponse(answer=answer)
