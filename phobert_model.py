from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class PhoBERTClassifier:
    def __init__(self, model_path: str):
        print(f"🔥 Loading PhoBERT from: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        # mapping label
        self.labels = [0.0, 1.0, 2.0]

        print("✅ PhoBERT loaded!")

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # classification softmax
        probs = torch.softmax(outputs.logits, dim=1)

        confidence, idx = torch.max(probs, dim=1)

        label = self.labels[idx.item()]  # 👉 float 0.0 / 1.0 / 2.0

        return {
            "label": label,
            "confidence": float(confidence.item())
        }