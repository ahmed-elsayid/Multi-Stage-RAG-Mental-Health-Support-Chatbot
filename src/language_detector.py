import os
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "saved_models", "lang_detector.pkl"))

class LanguageDetector:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
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

    def predict(self, text: str) -> str:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot run inference.")
        return self.model.predict([text])[0]


if __name__ == "__main__":
    try:
        detector = LanguageDetector()
        print("Language detector model loaded successfully.")
        print("Detected language:", detector.predict("Hello, how are you?"))
    except Exception as e:
        print("Initialization failed:", e)