"""
ui_demo.py — ITI Student Project · NLP Final Task 2026
Multi-Stage RAG Mental Health Support Chatbot — Streamlit frontend.

Safe HTML rules:
  - CSS lives in one CSS constant, injected by load_css().
  - The banner and styled cards use standalone HTML blocks (pure text + static
    images via /app/static/). No Streamlit widgets live inside HTML blocks.
  - All layout, buttons, and chat use native Streamlit widgets.
  - Icons loaded via st.image() / safe_image() — never base64 or inline SVG.

Icon-loading rule:
  - Streamlit widgets  →  icons/ folder  (safe_image / st.image)
  - HTML blocks        →  /app/static/  (every HTML icon is mirrored there)

Backend:
  - Requires FastAPI backend running at http://localhost:8000
  - Start with: uvicorn app:app --reload
"""

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

from pathlib import Path
from datetime import datetime
from textwrap import dedent
import json
import html
import requests
import streamlit as st
import re

# ── 2. PAGE CONFIG  (must be the first Streamlit call) ───────────────────────

st.set_page_config(
    page_title="Mental Health Support Chatbot | ITI",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# st.logo() intentionally omitted — it placed a duplicate ITI logo in the
# Streamlit sidebar chrome. The logo only appears in the red banner below.

# ── 3. CONSTANTS ──────────────────────────────────────────────────────────────

# ── Colors ────────────────────────────────────────────────────────────────────
RED    = "#9B1C1F"   # ITI burgundy — primary brand, buttons, headers
NAVY   = "#173847"   # Dark navy — footer, headings
BG     = "#F5F5F5"   # Page background
TEXT   = "#1F2933"   # Dark body text
MUTED  = "#667085"   # Muted/caption text
BORDER = "#E4E7EC"   # Default card border

# ── Icon sizes (pixels) ───────────────────────────────────────────────────────
HERO_ICON_SIZE       = 96   # Hero icon beside the main title
ANALYSIS_ICON_SIZE = 54   # this is the on-screen size of the icon
ANALYSIS_ICON_BOX_SIZE = 67   # the container should be slightly bigger
HELP_ICON_SIZE       = 36    # "Need Immediate Help?" warning icon
SYSTEM_HEAD_ICON_SIZE = 30   # System Flow tab — section heading icon
SYSTEM_STEP_ICON_SIZE = 32   # System Flow tab — per-step pipeline icons
DISCLAIMER_ICON_SIZE  = 30   # Disclaimer tab — section heading icon
TAB_ICON_SIZE         = 28   # Tab bar icons (CSS-injected via /app/static/)

# ── Icon paths ────────────────────────────────────────────────────────────────
ICON_DIR = Path("static")  # Now unified

ICONS = {
    "hero":       ICON_DIR / "mental_health_support_icon.png",
    "language":   ICON_DIR / "language_icon.png",
    "emotion":    ICON_DIR / "emotion_icon.png",
    "intent":     ICON_DIR / "intent_icon.png",
    "route":      ICON_DIR / "route_icon.png",
    "help":       ICON_DIR / "immediately_icon.png",
    "system":     ICON_DIR / "system_icon.png",
    "disclaimer": ICON_DIR / "disclaimer_icon.png",
    "home":       ICON_DIR / "home_icon.png",
}

# ── Sample questions ──────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "I feel anxious all the time",
    "I feel very stressed",
    "I feel depressed and unmotivated",
    "I cannot sleep, I keep overthinking",
]

# ── Demo answers ──────────────────────────────────────────────────────────────
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm", "self-harm",
    "hurt myself", "don't want to live", "do not want to live",
    "want to die", "take my own life", "no reason to live",
]

CRISIS_ANSWER = (
    "I am really sorry you are feeling this way. Your safety matters right now. "
    "Please contact emergency services immediately, or reach out to a trusted person "
    "who can stay with you. You are important, and you do not have to go through this alone."
)

DEMO_ANSWERS = {
    "anxi": (
        "It is completely understandable to feel anxious. Start with slow breathing, "
        "naming things you can see around you, and writing down what triggered the feeling. "
        "Professional support can help if anxiety continues to affect your daily life."
    ),
    "stress": (
        "Stress can become overwhelming when everything feels urgent at once. Break the "
        "problem into smaller steps, choose one priority for today, and allow short breaks. "
        "A walk or breathing exercise can reduce pressure significantly."
    ),
    "depress": (
        "Depression makes even simple tasks feel heavy. Start very small: one routine, "
        "one short walk, one message to someone you trust. Reaching out to a mental "
        "health professional is an important and valid step."
    ),
    "sleep": (
        "Writing thoughts down before bed helps offload them from your mind. Try a "
        "consistent wind-down routine and reduce screen time before sleep. "
        "Professional guidance can help if this continues."
    ),
}

DEFAULT_ANSWER = (
    "I understand you are going through a difficult moment. Slowing down, naming what "
    "you feel, and taking one small supportive step can help. Speaking with someone you "
    "trust or a professional can make a real difference."
)

DEMO_CHUNKS = [
    {
        "context":  "I have been feeling anxious and do not know how to calm down.",
        "response": "Grounding and breathing exercises can help interrupt anxious thoughts.",
        "score":    0.91,
    },
    {
        "context":  "My mind keeps racing at night and I cannot sleep.",
        "response": "Writing thoughts down and a steady wind-down routine can help.",
        "score":    0.87,
    },
    {
        "context":  "I feel hopeless and have no motivation to do anything.",
        "response": "Small routines, gentle social support, and professional help support recovery.",
        "score":    0.83,
    },
]

# ── Pipeline steps (System Flow tab) ─────────────────────────────────────────
PIPELINE_STEPS = [
    ("1", False, "User Question",
    "Plain-language mental health question submitted through the interface."),     
    (None, False, "Language Detection",
     "Detects the input language — English, Arabic, and others. Module 1."),
    (None, False, "Emotion Classification",
     "Identifies emotional state: anxiety, sadness, stress, neutral. Module 2."),
    (None, False, "Intent Classification",
     "Classifies intent: mental health question, greeting, out-of-scope. Module 3."),
    (None, True,  "Crisis Safety Check",
     "Keyword check for crisis phrases — returns a direct safety message."),
    ("6", False, "Qdrant Retrieval",
    "Embeds the question and queries Qdrant Cloud for top-3 similar records. Module 4."),
    ("7", False, "Groq LLM",
    "Builds a grounded prompt and calls Groq openai/gpt-oss-20b (temperature 0.3)."),
    ("8", False, "Final Answer",
    "Empathetic, grounded answer returned with language, emotion, intent, and route."),
]

PIPELINE_STEP_ICONS = {
    "Language Detection":     ICONS["language"],
    "Emotion Classification": ICONS["emotion"],
    "Intent Classification":  ICONS["intent"],
    "Crisis Safety Check":    ICONS["help"],
}

# ── 4. CSS ────────────────────────────────────────────────────────────────────

CSS = f"""
<style>

html, body, [class*="css"] {{
    font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif;
}}
#MainMenu, footer, header, .stDeployButton {{ visibility:hidden; display:none; }}
.stApp {{ background-color: {BG}; }}
.block-container {{
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 1500px !important;
    width: 92vw !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

/* ── Buttons: default white pill ── */
.stButton > button {{
    background-color: white !important;
    color: #344054 !important;
    border: 1.5px solid #D0D5DD !important;
    border-radius: 20px !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    padding: 0.45rem 1rem !important;
    box-shadow: 0 1px 2px rgba(16,24,40,0.05) !important;
    transition: all 0.15s !important;
}}
.stButton > button:hover {{
    border-color: {RED} !important;
    color: {RED} !important;
    background-color: #FFF7F7 !important;
}}

/* ── Buttons: primary burgundy ── */
.stButton > button[kind="primary"] {{
    background-color: {RED} !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 3px 10px rgba(155,28,31,0.28) !important;
}}
.stButton > button[kind="primary"]:hover {{
    background-color: #7f1518 !important;
    color: white !important;
    border-color: transparent !important;
}}

/* ── Text area ── */

/* Main Streamlit text-area wrapper */
.stTextArea {{
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}}

/* BaseWeb wrapper — this is usually where the black border comes from */
.stTextArea div[data-baseweb="textarea"],
.stTextArea div[data-baseweb="base-input"] {{
    border: 1.5px solid #D0D5DD !important;
    border-radius: 12px !important;
    background-color: white !important;
    box-shadow: none !important;
    outline: none !important;
}}

/* Any inner wrapper inside the BaseWeb textarea */
.stTextArea div[data-baseweb="textarea"] > div,
.stTextArea div[data-baseweb="base-input"] > div {{
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background-color: white !important;
}}

/* Actual textarea */
.stTextArea textarea {{
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background-color: white !important;
    color: {TEXT} !important;
    border-radius: 12px !important;
    font-size: 0.91rem !important;
    line-height: 1.6 !important;
    resize: none !important;
    padding: 0.85rem 1rem !important;
    box-sizing: border-box !important;
}}

/* Focus state — burgundy, not black */
.stTextArea:focus-within div[data-baseweb="textarea"],
.stTextArea:focus-within div[data-baseweb="base-input"] {{
    border: 1.5px solid {RED} !important;
    box-shadow: 0 0 0 3px rgba(155,28,31,0.10) !important;
    outline: none !important;
}}

/* Remove browser black focus ring */
.stTextArea textarea:focus,
.stTextArea textarea:focus-visible,
.stTextArea div:focus,
.stTextArea div:focus-visible {{
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}}

.stTextArea textarea::placeholder {{
    color: #A8B0BC !important;
}}

label[data-testid="stWidgetLabel"] {{
    display: none !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: white !important;
    border-bottom: 1.5px solid {BORDER} !important;
    padding: 0 2rem !important;
    gap: 0.4rem !important;
    border-radius: 0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    justify-content: flex-end !important;
}}
.stTabs [data-baseweb="tab"] {{
    color: {MUTED} !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 0.9rem 1.7rem !important;
    border-bottom: 2.5px solid transparent !important;
    background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
    color: {RED} !important;
    font-weight: 700 !important;
    border-bottom: 2.5px solid {RED} !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {RED} !important;
    height: 2.5px !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background-color: {BG} !important;
    padding-top: 0 !important;
}}

/* ── Tab icons via CSS ::before (PNGs from /app/static/) ── */
.stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p::before {{
    content: "";
    display: inline-block;
    width: {TAB_ICON_SIZE}px;
    height: {TAB_ICON_SIZE}px;
    margin-right: 10px;
    vertical-align: -8px;
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
}}
.stTabs [data-baseweb="tab"]:nth-of-type(1)
    [data-testid="stMarkdownContainer"] p::before {{
    background-image: url("/app/static/home_icon.png");
}}
.stTabs [data-baseweb="tab"]:nth-of-type(2)
    [data-testid="stMarkdownContainer"] p::before {{
    background-image: url("/app/static/system_icon.png");
}}
.stTabs [data-baseweb="tab"]:nth-of-type(3)
    [data-testid="stMarkdownContainer"] p::before {{
    background-image: url("/app/static/disclaimer_icon.png");
}}

/* ── Cards (st.container border=True) ── */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 14px !important;
    border-color: #E8ECF0 !important;
    background-color: white !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}}

/* ── Sample question chips ── */
.st-key-s0 .stButton > button,
.st-key-s1 .stButton > button,
.st-key-s2 .stButton > button,
.st-key-s3 .stButton > button {{
    background-color: white !important;
    color: {NAVY} !important;
    border: 1.5px solid {RED} !important;
    border-radius: 30px !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 0.4rem 1rem !important;
    min-height: 40px !important;
    margin-bottom: 0.55rem !important;
    box-shadow: 0 1px 3px rgba(155,28,31,0.08) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
.st-key-s0 .stButton > button:hover,
.st-key-s1 .stButton > button:hover,
.st-key-s2 .stButton > button:hover,
.st-key-s3 .stButton > button:hover {{
    background-color: #FBECEC !important;
    color: {RED} !important;
    border: 2px solid {RED} !important;
    box-shadow: 0 2px 6px rgba(155,28,31,0.18) !important;
}}

/* ── Custom chat messages ── */
.chat-empty {{
    background: white;
    border: 1px solid #E8ECF0;
    border-radius: 10px;
    color: {MUTED};
    font-size: 0.86rem;
    padding: 0.9rem 1rem;
    margin-top: 0.7rem;
}}

.chat-row {{
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin: 0.75rem 0;
}}

.chat-avatar {{
    width: 42px;
    height: 42px;
    min-width: 42px;
    border-radius: 50%;
    position: relative;
    box-shadow: 0 1px 3px rgba(0,0,0,0.10);
}}

.chat-avatar-user {{
    background: {RED};
}}

.chat-avatar-user::before {{
    content: "";
    position: absolute;
    top: 9px;
    left: 14px;
    width: 10px;
    height: 10px;
    border: 2px solid white;
    border-radius: 50%;
}}

.chat-avatar-user::after {{
    content: "";
    position: absolute;
    left: 10px;
    bottom: 9px;
    width: 18px;
    height: 10px;
    border: 2px solid white;
    border-radius: 14px 14px 6px 6px;
}}

.chat-avatar-bot {{
    background: {NAVY};
}}

.chat-avatar-bot::before {{
    content: "";
    position: absolute;
    top: 12px;
    left: 10px;
    width: 18px;
    height: 14px;
    border: 2px solid white;
    border-radius: 5px;
}}

.chat-avatar-bot::after {{
    content: "";
    position: absolute;
    top: 17px;
    left: 15px;
    width: 4px;
    height: 4px;
    background: white;
    border-radius: 50%;
    box-shadow: 8px 0 0 white, 4px 7px 0 -1px white;
}}

.chat-bubble {{
    position: relative;
    flex: 1;
    border-radius: 8px;
    padding: 0.85rem 4.5rem 1.05rem 1rem;
    min-height: 48px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}

.chat-bubble-user {{
    background: #FFF2F2;
    border: 1px solid #F1D7D8;
}}

.chat-bubble-bot {{
    background: #F3F7FC;
    border: 1px solid #E1EAF4;
}}

.chat-bubble-crisis {{
    background: #FFF3F3;
    border: 1px solid #EFC2C3;
}}

.chat-text {{
    color: {TEXT};
    font-size: 0.92rem;
    line-height: 1.6;
    font-weight: 400;
}}

.chat-time {{
    position: absolute;
    right: 0.9rem;
    bottom: 0.65rem;
    color: {MUTED};
    font-size: 0.68rem;
    font-weight: 600;
    white-space: nowrap;
}}

/* ── Misc ── */
.stAlert {{ border-radius: 12px !important; }}
.stMarkdown h2 {{ color: {TEXT} !important; }}
.stMarkdown h3 {{ color: {NAVY} !important; }}
hr {{ border-color: {BORDER} !important; margin: 0.4rem 0 !important; }}
.streamlit-expanderHeader,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] > details > summary {{
    background-color: white !important;
    border-radius: 8px !important;
    color: {RED} !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    border: 1px solid {BORDER} !important;
}}
[data-testid="stExpander"] {{
    background-color: white !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
}}
[data-testid="stExpander"] > details {{
    background-color: white !important;
}}
/* Keep column padding light. We will control real spacing using explicit spacer columns. */
[data-testid="column"] {{ padding: 0 0.35rem !important; }}


/* ── Reusable label helpers ── */
.analysis-header {{
    display: block;
    color: {RED};
    font-size: 1.07rem;
    font-weight: 700;
    border-bottom: 2px solid {RED};
    padding-bottom: 0.45rem;
    margin-bottom: 0.6rem;
}}
.chat-label {{
    display: block;
    color: {RED};
    font-size: 0.95rem;
    font-weight: 800;
    padding-bottom: 0.55rem;
    margin-bottom: 0.75rem;
    border-bottom: 1.5px solid #D8B7B8;
}}

</style>
"""


def load_css() -> None:
    """Inject the single CSS block. Call once at the start of main()."""
    st.markdown(CSS, unsafe_allow_html=True)


# ── 5. BACKEND INTEGRATION ────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:8000/chat"

# Maps ISO 639-1 language codes returned by the backend to display names.
_LANG_NAMES = {
    "en": "English", "ar": "Arabic", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
    "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "tr": "Turkish", "pl": "Polish", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "no": "Norwegian", "cs": "Czech", "ro": "Romanian",
}

def is_crisis_message(text: str) -> bool:
    return any(kw in text.lower() for kw in CRISIS_KEYWORDS)


# --- chatbot_ui.py (demo_response generator) ---

def demo_response(question: str):
    """
    Call the FastAPI multi-stage backend and YIELD chunks, metadata, and retrieved documents
    in real-time. This prevents UI blocking.
    """
    try:
        resp = requests.post(
            BACKEND_URL,
            json={"text": question},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                packet = json.loads(raw_line)
                yield packet  # Yield every token/metadata packet immediately
            except (json.JSONDecodeError, ValueError):
                continue

    except requests.exceptions.ConnectionError:
        yield {
            "type": "error",
            "text": (
                "⚠️ **Backend not running.**\n\n"
                "Please start the FastAPI server in a separate terminal:\n\n"
                "```\nuvicorn app:app --reload\n```"
            )
        }
    except Exception as exc:
        yield {"type": "error", "text": f"⚠️ An unexpected error occurred: {exc}"}

# ── 6. SESSION STATE ──────────────────────────────────────────────────────────

def _blank_analysis() -> dict:
    return {"language": "—", "emotion": "—", "intent": "—", "route": "—"}


def initialize_state() -> None:
    """Initialise session state keys on first run."""
    if "history"  not in st.session_state: st.session_state.history  = []
    if "prefill"  not in st.session_state: st.session_state.prefill  = ""
    if "analysis" not in st.session_state: st.session_state.analysis = _blank_analysis()


def reset_chat() -> None:
    """Clear chat history, prefill, and analysis panel."""
    st.session_state.history  = []
    st.session_state.prefill  = ""
    st.session_state.analysis = _blank_analysis()


# ── 7. UI HELPERS ─────────────────────────────────────────────────────────────

def safe_image(path: Path, width: int = 40, fallback: str = "") -> None:
    """Render a PNG icon if the file exists; show fallback text if not."""
    if path.exists():
        st.image(str(path), width=width)
    elif fallback:
        st.markdown(fallback)


def render_banner() -> None:
    """Top red banner: ITI logo (via /app/static/) and title.
    No phone, email, or official ITI contact details."""
    st.markdown("""
<div style="
    background: #9B1C1F;
    padding: 1rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid #7B1618;
    margin: 0;
">
    <div style="display:flex; align-items:center; gap:14px;">
        <img src="/app/static/ITI.svg"
             alt="ITI"
             style="height:44px; filter:brightness(0) invert(1);"
             onerror="this.style.display='none'">
        <div>
            <div style="color:white; font-size:0.9rem; font-weight:700; letter-spacing:0.3px;">
                NLP Final Task 2026
            </div>
            <div style="color:rgba(255,255,255,0.6); font-size:0.7rem; letter-spacing:0.2px;">
                ITI Student Demo Project
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

def render_analysis_card(label: str, value: str, icon_key: str) -> None:
    """Analysis card: PNG icon on the left, label + value on the right."""
    icon_name = ICONS[icon_key].name
    safe_value = value or "—"

    html = f"""
<div style="
    background: white;
    border: 1px solid #E8ECF0;
    border-radius: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    padding: 1.05rem 1.05rem;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1rem;
    min-height: 86px;
    min-width: 0;
">
<div style="
    width:{ANALYSIS_ICON_BOX_SIZE}px;
    height:{ANALYSIS_ICON_BOX_SIZE}px;
    min-width:{ANALYSIS_ICON_BOX_SIZE}px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    overflow:hidden;
">
    <img src="/app/static/{icon_name}" alt=""
        style="
            width:{ANALYSIS_ICON_SIZE}px;
            height:{ANALYSIS_ICON_SIZE}px;
            object-fit:contain;
            display:block;
        "
        onerror="this.style.display='none'">
</div>

<div style="
    min-width:0;
    flex:1;
    overflow:hidden;
">
<div style="
    color:{MUTED};
    font-size:0.68rem;
    font-weight:800;
    letter-spacing:0.04px;
    white-space:nowrap;
    overflow:hidden;
    line-height:1.1;
">
{label.upper()}
</div>

<div style="
    color:{TEXT};
    font-size:0.82rem;
    font-weight:700;
    word-break:break-word;
    overflow-wrap:break-word;
    white-space:normal;
    line-height:1.3;
    margin-top:0.28rem;
">
{safe_value}
</div>
</div>
</div>
"""

    st.markdown(dedent(html), unsafe_allow_html=True)

def render_about_card() -> None:
    """Soft blue-lavender info card about the chatbot system."""
    st.markdown("""
<div style="
    background: linear-gradient(135deg, #E3EDF9 0%, #ECE8F5 100%);
    border: 1px solid #C9D8EA;
    border-radius: 14px;
    box-shadow: 0 1px 5px rgba(23,56,71,0.07);
    padding: 1.35rem 1.55rem;
    margin: 0 0.15rem;
">
    <div style="color:#9B1C1F; font-size:1rem; font-weight:700; margin-bottom:0.5rem;">
        About This Chatbot
    </div>
    <div style="color:#667085; font-size:0.86rem; line-height:1.6;">
        This system uses multiple NLP models to understand your question,
        retrieve relevant counselling knowledge, and generate a grounded,
        supportive answer.
    </div>
</div>
""", unsafe_allow_html=True)


def render_help_card() -> None:
    """Soft pink safety card. Icon size controlled by HELP_ICON_SIZE constant."""
    st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #FBE4E4 0%, #F9DADB 100%);
    border: 1px solid #EFC2C3;
    border-radius: 14px;
    box-shadow: 0 1px 5px rgba(155,28,31,0.09);
    padding: 1.25rem 1.4rem;
">
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;">
        <img src="/app/static/immediately_icon.png"
             alt=""
             style="width:{HELP_ICON_SIZE}px; height:{HELP_ICON_SIZE}px; object-fit:contain;"
             onerror="this.style.display='none'">
        <span style="color:#9B1C1F; font-size:1rem; font-weight:700;">
            Need Immediate Help?
        </span>
    </div>
    <div style="color:#667085; font-size:0.86rem; line-height:1.6; margin-bottom:0.6rem;">
        If you are in immediate danger or thinking about harming yourself,
        please contact emergency services or a trusted person right away.
    </div>
    <div style="color:#173847; font-size:0.9rem; font-weight:700;">
        You are important. You are not alone.
    </div>
</div>
""", unsafe_allow_html=True)

def _chat_text(text: str) -> str:
    """Escape text, then support simple markdown-like formatting inside HTML bubbles."""
    safe = html.escape(text or "")

    # Bold: **text**
    safe = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", safe)

    # New lines
    safe = safe.replace("\n\n", "<br><br>")
    safe = safe.replace("\n", "<br>")

    return safe


def render_chat_message(role: str, text: str, timestamp: str, is_crisis: bool = False) -> None:
    """Render one custom chat message row with controlled avatar and bubble colors."""
    safe_text = _chat_text(text)
    safe_time = html.escape(timestamp or "")

    if role == "user":
        avatar_class = "chat-avatar-user"
        bubble_class = "chat-bubble-user"
    else:
        avatar_class = "chat-avatar-bot"
        bubble_class = "chat-bubble-crisis" if is_crisis else "chat-bubble-bot"

    message_html = (
        f'<div class="chat-row">'
            f'<div class="chat-avatar {avatar_class}"></div>'
            f'<div class="chat-bubble {bubble_class}">'
                f'<div class="chat-text">{safe_text}</div>'
                f'<div class="chat-time">{safe_time}</div>'
            f'</div>'
        f'</div>'
    )

    st.markdown(message_html, unsafe_allow_html=True)


def render_chat_history() -> None:
    """Display the full conversation history from session state."""
    if not st.session_state.history:
        st.markdown(
            '<div class="chat-empty">Ask a question to get started.</div>',
            unsafe_allow_html=True,
        )
        return

    for item in st.session_state.history:
        render_chat_message(
            role="user",
            text=item["question"],
            timestamp=item["time"],
        )

        render_chat_message(
            role="assistant",
            text=item["result"]["answer"],
            timestamp=item["time"],
            is_crisis=item["result"]["is_crisis"],
        )

        chunks = item["result"].get("retrieved_chunks", [])
        if chunks:
            with st.expander(f"Retrieved Context  (Top {len(chunks)})", expanded=False):
                st.caption("Records retrieved from Qdrant Cloud — the LLM used only this context to answer.")
                for idx, chunk in enumerate(chunks, 1):
                    score_pct = int(chunk.get("score", 0) * 100)
                    context  = chunk.get("context",  "—")
                    response = chunk.get("response", "—")
                    st.markdown(
                        f"""
<div style="
    background:white;
    border:1px solid #E4E7EC;
    border-left: 4px solid #9B1C1F;
    border-radius:10px;
    padding:0.85rem 1rem;
    margin-bottom:0.6rem;
    color:#1F2933;
    font-size:0.82rem;
    line-height:1.55;
">
  <div style="font-weight:700;color:#9B1C1F;margin-bottom:0.45rem;">
    Chunk {idx} &nbsp;·&nbsp; {score_pct}% match
  </div>
  <div style="margin-bottom:0.3rem;">
    <span style="font-weight:600;color:#667085;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;">Concern</span><br>
    {context}
  </div>
  <div>
    <span style="font-weight:600;color:#667085;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;">Counsellor response</span><br>
    {response}
  </div>
</div>""",
                        unsafe_allow_html=True,
                    )

def render_footer() -> None:
    """Dark navy footer with project info and module list."""
    st.markdown("""
<div style="
    background: #173847;
    color: rgba(255,255,255,0.85);
    padding: 1.6rem 2.5rem;
    border-radius: 12px 12px 0 0;
    margin-top: 1.5rem;
    display: grid;
    grid-template-columns: 2.2fr 1fr 1.2fr;
    gap: 2rem;
    align-items: start;
">
    <div>
        <div style="font-weight:700; font-size:0.88rem; margin-bottom:0.5rem; color:white;">
            NLP Final Task 2026
        </div>
        <div style="font-weight:400; font-size:0.78rem;
                    color:rgba(255,255,255,0.55); margin-bottom:0.5rem;">
            ITI Student Demo Project
        </div>
        <p style="color:rgba(255,255,255,0.45); font-size:0.76rem; line-height:1.7; margin:0;">
            Multi-Stage RAG Mental Health Support Chatbot.<br>
            Frontend-only demo — backend integration coming soon.
        </p>
    </div>
    <div>
        <div style="font-weight:600; font-size:0.86rem; margin-bottom:0.65rem;">Quick Links</div>
        <div style="color:rgba(255,255,255,0.56); font-size:0.78rem; line-height:2.1;">
            About Us<br>Branches<br>News<br>Contact Us
        </div>
    </div>
    <div>
        <div style="font-weight:600; font-size:0.86rem; margin-bottom:0.65rem;">
            Project Modules
        </div>
        <div style="color:rgba(255,255,255,0.56); font-size:0.78rem; line-height:1.9;">
            Language Detection<br>
            Emotion Classification<br>
            Intent Classification<br>
            Q&amp;A RAG Pipeline
        </div>
    </div>
</div>
<div style="
    background: #122d3a;
    color: rgba(255,255,255,0.32);
    text-align: center;
    font-size: 0.71rem;
    padding: 0.55rem 2rem;
    border-top: 1px solid rgba(255,255,255,0.07);
">
    &#169; 2026 ITI Student Demo &middot; NLP Final Task 2026
    &middot; For educational and research purposes only.
</div>
""", unsafe_allow_html=True)


# ── 8. TAB RENDERERS ─────────────────────────────────────────────────────────

# --- chatbot_ui.py (_process_question engine) ---

def _process_question(question: str) -> None:
    q = question.strip()
    if not q:
        st.warning("Please type a question before submitting.")
        return

    language = "Unknown"
    emotion = "—"
    intent = "—"
    full_answer = ""
    retrieved_chunks = []
    is_crisis = False

    # 1. Set up real-time rendering containers
    # (Shows the user their user bubble immediately during generation)
    render_chat_message(role="user", text=q, timestamp=datetime.now().strftime("%I:%M %p"))

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        metadata_placeholder = st.empty()

        # 2. Iterate over our active generator stream
        for packet in demo_response(q):
            if packet.get("type") == "error":
                full_text = packet.get("text", "")
                response_placeholder.markdown(full_text)
                return

            elif packet.get("type") == "metadata":
                lang_code = packet.get("detected_language", "")
                language = _LANG_NAMES.get(lang_code, lang_code or "Unknown")
                emotion = packet.get("detected_emotion", "—").capitalize()
                raw_intent = packet.get("detected_intent", "—")

                # Format score values nicely
                score = packet.get("detected_emotion_score")
                score_suffix = f" ({score:.2f})" if score is not None else ""

                intent = {
                    "asking_mental_health_question": "Mental Health Q",
                    "out_of_scope": "Out of Scope",
                    "greeting": "Greeting",
                    "goodbye": "Goodbye",
                    "gratitude": "Gratitude",
                }.get(raw_intent, raw_intent.replace("_", " ").title())

                # Render metadata panel dynamically
                meta_info = f"🌍 Lang: {language.upper()} | 🎭 Emotion: {emotion}{score_suffix} | 🎯 Intent: {intent}"
                metadata_placeholder.caption(meta_info)

                # Dynamically push metadata updates to the right panel state
                st.session_state.analysis = {
                    "language": language,
                    "emotion": f"{emotion}{score_suffix}",
                    "intent": intent,
                    "route": "Crisis Safety" if is_crisis else (
                        "RAG Retrieval" if raw_intent == "asking_mental_health_question" else intent)
                }

            elif packet.get("type") == "chunks":
                retrieved_chunks = packet.get("data", [])

            elif packet.get("type") == "chunk":
                chunk_text = packet.get("text", "")
                full_answer += chunk_text

                # Render markdown as it arrives
                response_placeholder.markdown(full_answer)

    # 3. Save finalized state to state history
    is_crisis = is_crisis_message(q)
    route = "Crisis Safety" if is_crisis else ("RAG Retrieval" if intent == "Mental Health Q" else intent)

    st.session_state.history.append({
        "question": q,
        "result": {
            "answer": full_answer,
            "is_crisis": is_crisis,
            "route": route,
            "retrieved_chunks": retrieved_chunks,
            "language": language,
            "emotion": emotion,
            "intent": intent,
        },
        "time": datetime.now().strftime("%I:%M %p"),
    })

    st.session_state.prefill = ""
    st.rerun()

def render_chat_tab() -> None:
    """Tab 1 — Home / Chat: hero, sample questions, input, analysis panel.

    Spacing note:
    We use small spacer columns around the left and right content so buttons,
    input boxes, and analysis cards do not touch the parent container edges.
    """

    # Outer layout with small gutters around both sides
    gutter_l, col_left, mid_gap, col_right, gutter_r = st.columns(
        [0.05, 2.35, 0.12, 1.0, 0.05],
        gap="small"
    )

    # ── LEFT: chat area ───────────────────────────────────────────────────────
    with col_left:
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

        # Inner content area for hero
        hero_pad_l, hero_area, hero_pad_r = st.columns([0.03, 0.94, 0.03], gap="small")

        with hero_area:
            ic_col, ti_col = st.columns([0.14, 0.86], gap="medium")

            with ic_col:
                safe_image(ICONS["hero"], width=HERO_ICON_SIZE, fallback="❤️‍🩹")

            with ti_col:
                st.markdown(
                    f"<h2 style='color:{TEXT}; margin:0 0 0.25rem 0; padding:0;'>"
                    "Mental Health Support Chatbot</h2>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='color:#475467; font-size:0.93rem; line-height:1.55;"
                    " margin-top:0.1rem;'>"
                    "Grounded, empathetic, and context-aware support for anxiety, "
                    "depression, stress, and coping questions.</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='margin-top:0rem;'></div>", unsafe_allow_html=True)

        # Inner content area for sample questions
        sample_pad_l, sample_area, sample_pad_r = st.columns([0.04, 0.92, 0.04], gap="small")

        with sample_area:
            st.markdown(
                f"<div style='color:{NAVY}; font-size:0.95rem; font-weight:700;"
                " margin-top:-0.2rem; margin-bottom:0.45rem;'>Try a question</div>",
                unsafe_allow_html=True,
            )

            # 2 × 2 sample question grid with proper side padding
            r1c1, r1c2 = st.columns(2, gap="large")
            r2c1, r2c2 = st.columns(2, gap="large")

            for col, idx in [(r1c1, 0), (r1c2, 1), (r2c1, 2), (r2c2, 3)]:
                with col:
                    if st.button(
                        SAMPLE_QUESTIONS[idx],
                        key=f"s{idx}",
                        use_container_width=True
                    ):
                        st.session_state.prefill = SAMPLE_QUESTIONS[idx]
                        st.rerun()

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

        # Inner content area for input + buttons + chat
        input_pad_l, input_area, input_pad_r = st.columns([0.04, 0.92, 0.04], gap="small")

        with input_area:
            question = st.text_area(
                "Question",
                value=st.session_state.prefill,
                placeholder="Type your question here...",
                height=68,
                label_visibility="collapsed",
            )

            st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)

            b_ask, b_clear, _sp = st.columns([1.45, 1.25, 2.2], gap="medium")

            with b_ask:
                ask = st.button(
                    "Ask Chatbot",
                    type="primary",
                    icon=":material/send:",
                    use_container_width=True,
                )

            with b_clear:
                if st.button(
                    "Clear Chat",
                    icon=":material/delete:",
                    use_container_width=True
                ):
                    reset_chat()
                    st.rerun()

            if ask:
                _process_question(question)


            st.markdown("<div style='margin-top:1.1rem;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="chat-label">Chat</div>', unsafe_allow_html=True)
            render_chat_history()            

    # ── RIGHT: analysis panel ────────────────────────────────────────────────
    with col_right:
        st.markdown("<div style='margin-top:0.65rem;'></div>", unsafe_allow_html=True)

        # Inner right panel area so analysis cards do not touch the right border
        right_pad_l, right_area, right_pad_r = st.columns([0.02, 0.96, 0.02], gap="small")

        with right_area:
            st.markdown(
                '<span class="analysis-header">Analysis</span>',
                unsafe_allow_html=True,
            )

            _a = st.session_state.analysis

            a1, a2 = st.columns(2, gap="small")
            a3, a4 = st.columns(2, gap="small")

            with a1:
                render_analysis_card("Language", _a["language"], "language")
            with a2:
                render_analysis_card("Emotion", _a["emotion"], "emotion")
            with a3:
                render_analysis_card("Intent", _a["intent"], "intent")
            with a4:
                render_analysis_card("Route", _a["route"], "route")

            st.markdown("<div style='margin-top:1.1rem;'></div>", unsafe_allow_html=True)
            render_about_card()

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            render_help_card()

def render_system_flow_tab() -> None:
    """Tab 2 — System Flow: pipeline architecture + notes."""
    flow_col, note_col = st.columns([2, 1], gap="large")

    with flow_col:
        st.write("")

        header_html = (
            f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:0.25rem;">'
            f'<img src="/app/static/system_icon.png" alt="" '
            f'style="width:{SYSTEM_HEAD_ICON_SIZE}px; height:{SYSTEM_HEAD_ICON_SIZE}px; object-fit:contain;" '
            f'onerror="this.style.display=\'none\'">'
            f'<h3 style="color:{NAVY}; margin:0; font-size:1.55rem;">Pipeline Architecture</h3>'
            f'</div>'
            f'<div style="color:#667085; font-size:0.88rem; margin-bottom:1rem;">'
            f'Every user question passes through this multi-stage pipeline.'
            f'</div>'
        )

        st.markdown(header_html, unsafe_allow_html=True)

        for bullet, is_crisis, title, desc in PIPELINE_STEPS:
            step_icon = PIPELINE_STEP_ICONS.get(title)
            suffix = " — crisis bypass" if is_crisis else ""

            if step_icon:
                icon_name = step_icon.name
                icon_html = (
                    f'<img src="/app/static/{icon_name}" alt="" '
                    f'style="width:{SYSTEM_STEP_ICON_SIZE}px; height:{SYSTEM_STEP_ICON_SIZE}px; object-fit:contain;" '
                    f'onerror="this.style.display=\'none\'">'
                )
            else:
                icon_html = (
                    f'<div style="'
                    f'width:{SYSTEM_STEP_ICON_SIZE}px; '
                    f'height:{SYSTEM_STEP_ICON_SIZE}px; '
                    f'border-radius:50%; '
                    f'background:#FBECEC; '
                    f'color:{RED}; '
                    f'display:flex; '
                    f'align-items:center; '
                    f'justify-content:center; '
                    f'font-size:1rem; '
                    f'font-weight:800;'
                    f'">{bullet}</div>'
                )

            border_color = "#EFC2C3" if is_crisis else "#E8ECF0"
            bg_color = "#FFF7F7" if is_crisis else "white"
            title_color = RED if is_crisis else NAVY

            step_html = (
                f'<div style="'
                f'background:{bg_color}; '
                f'border:1px solid {border_color}; '
                f'border-radius:14px; '
                f'padding:0.9rem 1rem; '
                f'margin-bottom:0.85rem; '
                f'box-shadow:0 1px 4px rgba(0,0,0,0.06); '
                f'display:flex; '
                f'align-items:center; '
                f'gap:14px;'
                f'">'
                    f'<div style="'
                    f'width:42px; '
                    f'min-width:42px; '
                    f'display:flex; '
                    f'align-items:center; '
                    f'justify-content:center;'
                    f'">'
                        f'{icon_html}'
                    f'</div>'

                    f'<div style="min-width:0;">'
                        f'<div style="'
                        f'color:{title_color}; '
                        f'font-size:0.95rem; '
                        f'font-weight:800; '
                        f'margin-bottom:0.25rem;'
                        f'">'
                            f'{title}{suffix}'
                        f'</div>'

                        f'<div style="'
                        f'color:#475467; '
                        f'font-size:0.82rem; '
                        f'line-height:1.45;'
                        f'">'
                            f'{desc}'
                        f'</div>'
                    f'</div>'
                f'</div>'
            )

            st.markdown(step_html, unsafe_allow_html=True)

    with note_col:
        st.write("")
        st.markdown("### Notes")

        st.info(
            "**Indexing vs. Retrieval**\n\n"
            "`04_QA_RAG.ipynb` runs **once** to embed and upload 3,508 counselling "
            "records to Qdrant Cloud.\n\n"
            "`rag_runtime.py` connects at query time — no re-uploading needed.\n\n"
            "Embeddings: `all-MiniLM-L6-v2` · 384 dims · cosine similarity.",
            icon="ℹ️",
        )

        st.success(
            "**Module 4 — Q&A RAG**\n\n"
            "Dataset: 3,508 counselling records\n\n"
            "Vector DB: Qdrant Cloud\n\n"
            "LLM: Groq `openai/gpt-oss-20b`",
            icon="✅",
        )
def render_disclaimer_tab() -> None:
    """Tab 3 — Disclaimer: educational notice, crisis resources, project info."""
    disc_col, _ = st.columns([2, 1], gap="large")

    with disc_col:
        st.write("")

        header_html = (
            f'<div style="display:flex; align-items:center; gap:12px; margin-bottom:0.55rem;">'
            f'<img src="/app/static/disclaimer_icon.png" alt="" '
            f'style="width:{DISCLAIMER_ICON_SIZE}px; height:{DISCLAIMER_ICON_SIZE}px; '
            f'object-fit:contain; opacity:0.92; filter:grayscale(0.2);" '
            f'onerror="this.style.display=\'none\'">'
            f'<h3 style="color:{NAVY}; margin:0; font-size:1.55rem;">Disclaimer</h3>'
            f'</div>'
            f'<div style="color:#667085; font-size:0.88rem; margin-bottom:1.2rem;">'
            f'Please read before using this chatbot.'
            f'</div>'
        )

        st.markdown(header_html, unsafe_allow_html=True)

        education_icon_html = (
            f'<div style="'
            f'width:24px; '
            f'height:24px; '
            f'min-width:24px; '
            f'border:1.5px solid #D9A441; '
            f'border-radius:50%; '
            f'color:#92400E; '
            f'background:#FFF3C4; '
            f'display:flex; '
            f'align-items:center; '
            f'justify-content:center; '
            f'font-size:0.82rem; '
            f'font-weight:900; '
            f'font-family:Georgia, serif; '
            f'line-height:1;'
            f'">!</div>'
        )

        education_html = (
            f'<div style="'
            f'background:#FFF8DB; '
            f'border:1px solid #F3D98B; '
            f'border-radius:14px; '
            f'box-shadow:0 1px 4px rgba(0,0,0,0.05); '
            f'padding:1.15rem 1.25rem; '
            f'margin-bottom:1rem;'
            f'">'
                f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:0.7rem;">'
                    f'{education_icon_html}'
                    f'<div style="color:#92400E; font-size:0.98rem; font-weight:800;">'
                        f'Educational & Research Purpose Only'
                    f'</div>'
                f'</div>'
                f'<div style="color:#475467; font-size:0.88rem; line-height:1.65;">'
                    f'This chatbot was developed as an academic NLP project by ITI students. '
                    f'It does not provide medical diagnosis, therapy, or professional advice. '
                    f'Responses are generated from anonymised counselling conversations and '
                    f'should not replace professional mental health care.'
                f'</div>'
            f'</div>'
        )

        st.markdown(education_html, unsafe_allow_html=True)

        crisis_icon_html = (
            f'<div style="'
            f'width:24px; '
            f'height:24px; '
            f'min-width:24px; '
            f'border:1.5px solid {RED}; '
            f'border-radius:50%; '
            f'color:{RED}; '
            f'background:#FBECEC; '
            f'display:flex; '
            f'align-items:center; '
            f'justify-content:center; '
            f'font-size:0.82rem; '
            f'font-weight:900; '
            f'font-family:Georgia, serif; '
            f'line-height:1;'
            f'">!</div>'
        )

        crisis_html = (
            f'<div style="'
            f'background:#FDE4E4; '
            f'border:1px solid #EFC2C3; '
            f'border-radius:14px; '
            f'box-shadow:0 1px 4px rgba(155,28,31,0.08); '
            f'padding:1.15rem 1.25rem; '
            f'margin-bottom:1rem;'
            f'">'
                f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:0.7rem;">'
                    f'{crisis_icon_html}'
                    f'<div style="color:{RED}; font-size:0.98rem; font-weight:800;">'
                        f'Crisis & Emergency Notice'
                    f'</div>'
                f'</div>'
                f'<div style="color:#475467; font-size:0.88rem; line-height:1.75;">'
                    f'If you are in immediate danger, do not rely on this chatbot. '
                    f'Contact emergency services or a trusted person immediately.'
                    f'<br><br>'
                    f'<strong style="color:{NAVY};">IASP:</strong> iasp.info/resources/Crisis_Centres'
                    f'<br><br>'
                    f'<strong style="color:{NAVY};">Crisis Text Line (US):</strong> Text HOME to 741741'
                    f'<br><br>'
                    f'<strong style="color:{NAVY};">Befrienders Worldwide:</strong> befrienders.org'
                f'</div>'
            f'</div>'
        )

        st.markdown(crisis_html, unsafe_allow_html=True)

        project_html = (
            f'<div style="'
            f'background:white; '
            f'border:1px solid #E8ECF0; '
            f'border-radius:14px; '
            f'box-shadow:0 1px 4px rgba(0,0,0,0.05); '
            f'padding:1.15rem 1.25rem; '
            f'margin-bottom:0.85rem;'
            f'">'
                f'<div style="color:{NAVY}; font-size:0.98rem; font-weight:800; margin-bottom:0.75rem;">'
                    f'Project Information'
                f'</div>'
                f'<div style="color:#475467; font-size:0.88rem; line-height:1.8;">'
                    f'<strong>Dataset:</strong> Mental Health Counseling Conversations (3,508 records)<br>'
                    f'<strong>Embeddings:</strong> all-MiniLM-L6-v2 · 384 dimensions · cosine similarity<br>'
                    f'<strong>Vector Database:</strong> Qdrant Cloud<br>'
                    f'<strong>LLM:</strong> Groq openai/gpt-oss-20b · Temperature 0.3<br>'
                    f'<strong>Modules:</strong> Language Detection · Emotion Classification · '
                    f'Intent Classification · Q&A RAG'
                f'</div>'
            f'</div>'
        )

        st.markdown(project_html, unsafe_allow_html=True)

        footer_note_html = (
            f'<div style="color:#667085; font-size:0.8rem; line-height:1.5;">'
            f'This is a student demo project. It is not an official ITI product or service.'
            f'</div>'
        )

        st.markdown(footer_note_html, unsafe_allow_html=True)
# ── 9. MAIN ───────────────────────────────────────────────────────────────────

def main() -> None:
    load_css()
    initialize_state()
    render_banner()

    tab_chat, tab_flow, tab_disc = st.tabs([
        "Home / Chat",
        "System Flow",
        "Disclaimer",
    ])

    with tab_chat:
        render_chat_tab()

    with tab_flow:
        render_system_flow_tab()

    with tab_disc:
        render_disclaimer_tab()

    render_footer()


if __name__ == "__main__":
    main()
