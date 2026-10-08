"""Adapt the original IDExtractor / IDTagger result to the HTTP contract."""
from .common import tokenize
from .extractor import IDExtractor


class IdentifierExtractor(IDExtractor):
    def __init__(self, artifact_dir, threshold=None):
        super().__init__(model_dir=artifact_dir, device="cpu", threshold=threshold)

    def predict(self, query: str) -> dict[str, str]:
        # The original tagger supports at most 512 tokens.
        if len(tokenize(query)) > 512:
            return {}
        result = super().predict(query)
        return result if any(result.values()) else {}
