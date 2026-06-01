import os
import joblib

from config import LANG_MODEL_PATH
class LanguageDetector:
    def __init__(self, model_path=LANG_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Language model file not found at: {self.model_path}. "
                "Please place the saved 'lang_detector.pkl' file inside the 'saved_models/' directory."
            )
        self.model = joblib.load(self.model_path)
        print(f"Language model loaded successfully from: {self.model_path}")



    def predict(self, text: str) -> str:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot run inference.")
        return self.model.predict([text])[0]



if __name__ == "__main__":
    try:
        detector = LanguageDetector()
        test_texts = [
            "can you help me with my anxiety?",
            "¿Cómo estás?",
            "Bonjour, comment ça va?",
            "Hallo, wie geht's dir?"
        ]
        for text in test_texts:
            lang = detector.predict(text)
            print(f"Text: '{text}' -> Detected Language: '{lang}'")
    except Exception as e:
        print(f"Error during testing: {e}")




