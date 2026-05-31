# 🧠 Multi-Stage RAG Mental Health Support Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot designed to provide empathetic and context-aware support for mental health topics such as anxiety, stress, depression, and coping strategies.

The system combines **Language Detection**, **Emotion Classification**, **Intent Routing**, and **Semantic Retrieval** to generate relevant and personalized responses.

---

## ✨ Features

- 🌍 Multilingual support with automatic query translation for non-English inputs
- 😊 Emotion-aware response generation (LLM infers emotion when confidence is low)
- 🎯 Intent classification and routing
- 📚 Retrieval-Augmented Generation (RAG)
- 🔎 Semantic search with Qdrant
- 🤖 Groq-powered response generation
- 🚨 Crisis detection with immediate safety resources
- ⚡ FastAPI streaming backend
- 💬 Streamlit chat interface

---

## 🏗️ System Architecture

```
User Query
    │
    ├── Language Detection
    │       └── Non-English? → Translate to English for retrieval
    │
    ├── Emotion Classification
    │       └── Low confidence? → LLM infers emotion from context
    │
    └── Intent Classification
              │
    ┌─────────┼──────────────────────┐
    │         │                      │
greeting   out_of_scope     asking_mental_health_question
goodbye    gratitude                 │
    │         │               Crisis detected?
    │         │              /               \
 Hardcoded    │            Yes               No
 Response     │             │                │
              │      Crisis Response    RAG Retrieval
              │                              │
              └──────────────┬───────────────┘
                             │
                      Groq LLM Response (Streaming)
                             │
                       Final Answer
```

---

## 📁 Project Structure

```
Multi-Stage-RAG-Mental-Health-Support-Chatbot/
│
├── notebooks/
│   ├── 1_Language_Detection.ipynb
│   ├── 2_Emotion_Classifier.ipynb
│   ├── 3_rag_initialization.ipynb
│   └── 04_QA_RAG.ipynb
│
├── src/
│   ├── language_detector.py
│   ├── emotion_classifier.py
│   ├── intent_classifier.py
│   └── rag_pipeline.py
│
├── saved_models/                        ← not tracked by git
│   ├── lang_detector.pkl
│   └── emotion_distilbert_finetuned/
│
├── config.py                            ← centralized configuration
├── app.py                               ← FastAPI backend (POST /chat)
├── ui.py                                ← Streamlit frontend
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🧠 Models & Components

| Component | Model / Tool |
|---|---|
| Language Detection | TF-IDF + CalibratedLinearSVC (custom trained) |
| Emotion Classification | DistilBERT fine-tuned |
| Intent Classification | Groq `llama-3.1-8b-instant` |
| Translation | Groq `openai/gpt-oss-120b` |
| Response Generation | Groq `openai/gpt-oss-120b` |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector Database | Qdrant Cloud |

### Design Choices

- **CalibratedLinearSVC + char_wb TF-IDF** for language detection — handles short, informal chat messages better than standard char n-grams, and provides confidence scores for fallback logic.
- **DistilBERT** captures emotional context efficiently. When confidence is below 0.6, the LLM infers emotion directly from the user's message.
- **Separate models for intent vs generation** — a fast 8b model handles the simple 5-class intent classification while the larger model handles nuanced response generation.
- **Query translation before retrieval** — non-English queries are translated to English before hitting Qdrant, since the knowledge base is English-only. The LLM still responds in the user's original language.
- **Patient contexts embedded** (not therapist responses) — improves retrieval quality by matching similar user situations rather than similar answers.

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/ahmed-elsayid/Multi-Stage-RAG-Mental-Health-Support-Chatbot.git
cd Multi-Stage-RAG-Mental-Health-Support-Chatbot
```

### Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

```bash
cp .env.example .env
```

Fill in your credentials in `.env`:

```
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
```

---

## 🗂️ Saved Models

The trained models are not tracked by git. Place them in the `saved_models/` directory before running:

```
saved_models/
├── lang_detector.pkl                    ← trained in notebook 1
└── emotion_distilbert_finetuned/        ← trained in notebook 2
```

To retrain, run the corresponding notebooks in the `notebooks/` folder.

---

## ▶️ Running the Application

### Start Backend

```bash
uvicorn app:app --reload
```

### Start Frontend

```bash
streamlit run ui.py
```

---

## 📚 RAG Pipeline

The knowledge base is built from the **Mental Health Counseling Conversations** dataset (3,508 rows after cleaning).

**Ingestion** (one-time, already done via notebook):
1. Loads and cleans the dataset
2. Embeds each `context` field using `all-MiniLM-L6-v2`
3. Uploads 384-dimensional vectors to Qdrant Cloud collection `mental_health_chunks`

**Runtime** (on every user message):
1. Detects language → translates to English if needed
2. Checks for crisis keywords → returns safety message if detected
3. Embeds the (translated) query and retrieves top-3 similar chunks from Qdrant
4. Builds a prompt from retrieved context + detected emotion
5. Streams the Groq response token by token back to the UI

### API Usage

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel anxious all the time"}'
```

Response is streamed as NDJSON:
```json
{"type": "metadata", "detected_language": "en", "detected_emotion": "fear", "detected_emotion_score": 0.91, "detected_intent": "asking_mental_health_question"}
{"type": "chunk", "text": "It sounds like you're dealing with..."}
{"type": "chunk", "text": " persistent anxiety..."}
```

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.

It is not a substitute for professional mental health care, diagnosis, or emergency services. If you or someone you know is in crisis, please contact a qualified mental health professional or local emergency services immediately.
