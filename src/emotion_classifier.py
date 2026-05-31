import os  # FIX: Added missing import
import torch
from transformers import pipeline

from config import EMOTION_MODEL_PATH
class EmotionClassifier:
    def __init__(self):
        # Choose GPU if available, fallback to CPU
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "text-classification",
            model=EMOTION_MODEL_PATH,
            tokenizer=EMOTION_MODEL_PATH,
            device=device,
            truncation=True,
            max_length=512
        )

    def predict(self, text: str) -> dict:
        """
        Returns a dict: {"label": str, "score": float}
        """
        result = self.classifier(text)[0]
        return {
            "label": result["label"],
            "score": float(result["score"])
        }


# --- Test ---
if __name__ == "__main__":
    hard_tests = {
        "I guess I'm fine. Really. Totally fine.": "sadness",
        "Oh great, another meeting. Just what I needed.": "anger",
        "My heart is pounding and I can't stop sweating.": "fear",
        "I could just hug you forever.": "love",
        "I never expected this to happen. Wow.": "surprise",
        "Why does everyone leave? I must be the problem.": "sadness",
        "Don't talk to me. Just leave me alone.": "anger",
        "I'm literally shaking right now.": "fear",
        "That's it? That's all you have to say?": "anger",
        "I feel like a ghost in my own home.": "sadness",
    }

    try:
        classifier = EmotionClassifier()
        print("Emotion Classifier loaded successfully from:", EMOTION_MODEL_PATH)
        for text, expected_emotion in hard_tests.items():
            prediction = classifier.predict(text)
            predicted_emotion = prediction["label"]
            confidence = prediction["score"]
            print(f"Text: '{text}' | Predicted: {predicted_emotion} ({confidence:.2f}) | Expected: {expected_emotion}")
    except Exception as e:
        print(f"Error loading model from {EMOTION_MODEL_PATH}: {e}")