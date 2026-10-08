"""CLI inference using exactly the artifact loaders and graph used by FastAPI."""
import argparse
import json
from pathlib import Path

import torch
from app.graph import build_graph, evaluate
from intent_classification.predictor import IntentClassifier
from identifier_extraction.predictor import IdentifierExtractor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--artifacts", default=str(Path(__file__).resolve().parent / "artifacts"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.artifacts)
    graph = build_graph(IntentClassifier(root/"intent_classification"), IdentifierExtractor(root/"identifier_extraction"))
    print(json.dumps(evaluate(graph,args.query),indent=2))
