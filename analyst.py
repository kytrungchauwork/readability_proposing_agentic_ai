from phobert_singleton import phobert_model

class Analyst:
    def run(self, text):
        result = phobert_model.predict(text)

        return {
            "text": text,
            "label": result["label"],
            "confidence": result["confidence"]
        }