# Module 3: Intent Classification via Groq

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class IntentClassifier:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"

    def classify_intent(self, user_query: str):
        prompt = f"""
        Classify the user's input into exactly ONE of the following intents:
        - greeting (e.g., "Hi", "Hello", "Hey there")
        - goodbye (e.g., "Bye", "Goodbye", "See you later")
        - gratitude (e.g., "Thank you", "Thanks", "I appreciate it")
        - asking_mental_health_question (e.g., questions about anxiety, sadness, coping strategies, panic attacks, depression)
        - out_of_scope (e.g., general knowledge questions, coding, sports, weather, math)

        Input: "{user_query}"

        Output ONLY the class name (one of: greeting, goodbye, gratitude, asking_mental_health_question, out_of_scope). Do not output punctuation or other text.
        """

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an objective classification system."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                max_tokens=10,
                temperature=0.0
            )
            intent = response.choices[0].message.content.strip().lower()
            print(f"Intent classification response: '{intent}'")
            return intent
        except Exception as e:
            print(f"Error classifying intent: {e}")
            return "asking_mental_health_question"

if __name__ == "__main__":
    classifier = IntentClassifier()
    test_queries = [
        "Hi, how are you?",
        "Thanks for your help!",
        "What are some coping strategies for anxiety?",
        "What's the weather like today?",
        "Goodbye!"
    ]
    for query in test_queries:
        intent = classifier.classify_intent(query)
        print(f"Query: '{query}' -> Classified Intent: '{intent}'")