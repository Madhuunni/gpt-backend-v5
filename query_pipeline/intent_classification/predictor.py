"""FastAPI adapter around the original ContextClassifier and TF-IDF loader."""
import os
import numpy as np
import torch
from .classifier import load, normalize


class IntentClassifier:
    def __init__(self, artifact_dir, threshold=None):
        self.model, self.vectorizer, self.metadata = load(artifact_dir)
        self.threshold = float(os.getenv("INTENT_THRESHOLD", "0.7")) if threshold is None else threshold
        if not 0 <= self.threshold <= 1:
            raise ValueError("INTENT_THRESHOLD must be between 0 and 1")

    @torch.inference_mode()
    def predict(self, query: str) -> str:
        x = torch.from_numpy(self.vectorizer.transform([normalize(query)]).toarray().astype(np.float32))
        if not torch.any(x):
            return ""
        score, index = self.model(x).softmax(dim=1)[0].max(dim=0)
        return self.metadata["labels"][index.item()] if score.item() >= self.threshold else ""
