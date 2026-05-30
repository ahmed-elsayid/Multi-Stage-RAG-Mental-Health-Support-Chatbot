# --- ui.py ---

import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Mental Health Support AI",
    page_icon="🧠",
    layout="centered"
)

# ── Sidebar Configurations ────────────────────────────────────────────────────
with st.sidebar:
    st.title("📌 Navigation & Resources")
    st.write("This chatbot utilizes local NLP classification models combined with a RAG pipeline to guide you safely.")

    st.divider()

    # 1. Emergency Helpline Quick Access (Aligns with your pipeline's safety checks)
    st.subheader("🚨 Crisis Resources")
    st.error(
        "If you are in immediate danger or experiencing severe distress, "
        "please connect with professional services immediately."
    )
    st.markdown("""
    - **US/Canada:** Call or Text **988**
    - **Crisis Text Line:** Text **HOME** to **741741**
    - **UK:** Call **111** (NHS services)
    - **International:** Find local centers at [Befrienders Worldwide](https://www.befrienders.org)
    """)

    st.divider()

    # 2. Reset Button for Local Testing/Defense Demos
    if st.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main Chat Interface ────────────────────────────────────────────────────────
st.title("🧠 Supportive AI Companion")
st.write("An empathetic assistant designed to provide mental health information and guidance.")

# Session State for history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "meta" in msg:
            st.caption(msg["meta"])

user_input = st.chat_input("How can I help you today?")

if user_input:
    # Append User Input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Render Assistant Interface (Streaming placeholders)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        metadata_placeholder = st.empty()

        try:
            # Connect using streaming mode (stream=True)
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={"text": user_input},
                stream=True,
                timeout=30
            )

            if response.status_code == 200:
                full_text = ""
                meta_info = ""

                # Read response chunks as they arrive
                for line in response.iter_lines():
                    if line:
                        payload = json.loads(line.decode("utf-8"))

                        # Handle metadata line
                        if payload.get("type") == "metadata":
                            # Render confidence score beside the emotion label
                            score = payload.get("detected_emotion_score")
                            score_suffix = f" ({score:.2f})" if score is not None else ""

                            # Clean up intent names for cleaner presentation
                            clean_intent = payload['detected_intent'].replace('_', ' ').title()

                            meta_info = (
                                f"🌍 Lang: {payload['detected_language'].upper()} | "
                                f"🎭 Emotion: {payload['detected_emotion']}{score_suffix} | "
                                f"🎯 Intent: {clean_intent}"
                            )
                            # Render metadata immediately
                            metadata_placeholder.caption(meta_info)

                        # Handle text chunk lines
                        elif payload.get("type") == "chunk":
                            full_text += payload["text"]
                            # Dynamically write markdown chunks as they land
                            response_placeholder.markdown(full_text)

                # Store fully constructed result in session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_text,
                    "meta": meta_info
                })

            else:
                st.error(f"Error communicating with backend. (Status code: {response.status_code})")
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the API server. "
                "Please verify that your backend server is running locally (e.g., uvicorn app:app --reload)"
            )
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")