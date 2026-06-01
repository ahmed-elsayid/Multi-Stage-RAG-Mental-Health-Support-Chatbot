# config.py
from dotenv import load_dotenv
import os

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ── Models ────────────────────────────────────────────────────────────────────
GROQ_MODEL     = "openai/gpt-oss-120b"       # main generation + translation
INTENT_MODEL   = "llama-3.1-8b-instant"      # intent classification only

# ── Qdrant ────────────────────────────────────────────────────────────────────
COLLECTION_NAME    = "mental_health_chunks"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
TOP_K_CHUNKS       = 3

# ── Emotion Classifier ────────────────────────────────────────────────────────
EMOTION_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "saved_models", "emotion_distilbert_finetuned")
)
EMOTION_CONFIDENCE_THRESHOLD = 0.6           # below this → LLM infers emotion

# ── Language Detector ─────────────────────────────────────────────────────────
LANG_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "saved_models", "lang_detector.pkl")
)

# ── Generation ────────────────────────────────────────────────────────────────
MAX_TOKENS_RESPONSE    = 800
MAX_TOKENS_TRANSLATION = 200
MAX_TOKENS_INTENT      = 10
TEMPERATURE_GENERATION = 0.3
TEMPERATURE_INTENT     = 0.0
TEMPERATURE_TRANSLATION = 0.0