import json

class Reviewer:
    def __init__(self, phobert_model):
        self.model = phobert_model

    def run(self, text, target_level):
        pred = self.model.predict(text)

        detected = float(pred["label"])
        confidence = pred["confidence"]

        match = (detected == float(target_level))

        return {
            "match": match,
            "detected_level": detected,
            "target_level": float(target_level),
            "confidence": confidence,
            "reason": "PhoBERT-based validation"
        }