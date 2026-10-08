"""Local supervised context classifier. No pretrained weights or LLM APIs."""
import argparse
import copy
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def normalize(text):
    # IDs are irrelevant to context classification. This does NOT extract IDs.
    text = re.sub(r"\b\d+\b", " numbertoken ", text.lower())
    text = text.replace("#", " number ")
    return " ".join(re.findall(r"[a-z]+", text))


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_data(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("intents"), list):
        raise ValueError('Expected {"intents": [{"context": ..., "query_patterns": [...]}]}')
    rows = {}
    for item in data["intents"]:
        label = item.get("context")
        patterns = item.get("query_patterns")
        if not isinstance(label, str) or not label.strip() or not isinstance(patterns, list) or not patterns:
            raise ValueError("Every intent needs a nonempty context and query_patterns list")
        for pattern in patterns:
            if not isinstance(pattern, str) or not normalize(pattern):
                raise ValueError("Every query pattern must contain English words")
            text = normalize(pattern)
            if text in rows and rows[text] != label:
                raise ValueError(f"Conflicting labels for normalized pattern: {text}")
            rows[text] = label
    if not rows:
        raise ValueError("No patterns found")
    return list(rows.items())


class ContextClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)  # raw logits; CrossEntropyLoss applies log-softmax


def reject_overlap(first, second):
    if set(x for x, _ in first) & set(x for x, _ in second):
        raise ValueError("Normalized patterns overlap between datasets; use genuinely new wording")


def split_train_validation(rows, seed):
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted({y for _, y in rows}):
        group = [r for r in rows if r[1] == label]
        if len(group) < 4:
            raise ValueError(f"{label}: need at least 4 UNIQUE normalized patterns for auto-split")
        rng.shuffle(group)
        count = max(1, round(0.2 * len(group)))
        val.extend(group[:count])
        train.extend(group[count:])
    return train, val


def tensors(vectorizer, rows, labels):
    unknown = {y for _, y in rows} - set(labels)
    if unknown:
        raise ValueError(f"Contexts absent from training: {sorted(unknown)}")
    x = torch.from_numpy(vectorizer.transform([x for x, _ in rows]).toarray().astype(np.float32))
    y = torch.tensor([labels.index(y) for _, y in rows], dtype=torch.long)
    return x, y


def train(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    rows = read_data(args.data)
    labels = sorted({y for _, y in rows})
    if len(labels) < 2:
        raise ValueError("Only one context found. A multiclass classifier needs at least two contexts; add realistic other-context examples. A one-class softmax always returns 100%.")
    if args.validation:
        train_rows, val_rows = rows, read_data(args.validation)
        reject_overlap(train_rows, val_rows)
        if {y for _, y in val_rows} != set(labels):
            raise ValueError("Validation must include every training context and no new contexts")
    else:
        train_rows, val_rows = split_train_validation(rows, args.seed)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, dtype=np.float32)
    vectorizer.fit([x for x, _ in train_rows])  # fit ONLY on training data
    x_train, y_train = tensors(vectorizer, train_rows, labels)
    x_val, y_val = tensors(vectorizer, val_rows, labels)
    model = ContextClassifier(x_train.shape[1], len(labels))
    counts = torch.bincount(y_train, minlength=len(labels)).float()
    criterion = nn.CrossEntropyLoss(weight=len(y_train) / (len(labels) * counts))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = criterion(model(x_val), y_val).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"epoch={epoch} validation_loss={val_loss:.4f}")
        if stale >= 25:
            break
    model.load_state_dict(best_state)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "weights.pt")
    metadata = {
        "format_version": 1, "normalization_version": 1,
        "labels": labels, "hidden_dim": 64,
        "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
        "idf": vectorizer.idf_.tolist(),
        "development_hashes": [fingerprint(x) for x, _ in train_rows + val_rows],
        "seed": args.seed, "best_epoch": best_epoch,
        "torch_version": str(torch.__version__)
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model.eval()
    with torch.inference_mode():
        predictions = model(x_val).argmax(1).tolist()
    report = metrics(y_val.tolist(), predictions, labels)
    report.update({"train_examples": len(train_rows), "validation_examples": len(val_rows), "best_epoch": best_epoch})
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Validation selected the model. Use separate held-out data for final evaluation.")


def load(directory):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if meta["format_version"] != 1 or meta["normalization_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    vec = TfidfVectorizer(ngram_range=(1, 2), vocabulary=meta["vocabulary"], dtype=np.float32)
    vec.idf_ = np.array(meta["idf"], dtype=np.float32)
    model = ContextClassifier(len(meta["vocabulary"]), len(meta["labels"]), meta["hidden_dim"])
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, vec, meta


def metrics(actual, predicted, labels):
    ids = list(range(len(labels)))
    return {
        "accuracy": accuracy_score(actual, predicted),
        "classification_report": classification_report(actual, predicted, labels=ids, target_names=labels, output_dict=True, zero_division=0),
        "confusion_matrix_labels": labels,
        "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(actual, predicted, labels=ids).tolist()
    }


def evaluate(args):
    model, vec, meta = load(args.model)
    rows = read_data(args.data)
    if any(fingerprint(x) in set(meta["development_hashes"]) for x, _ in rows):
        raise ValueError("Test data overlaps training/validation after normalization; use unseen queries")
    x, y = tensors(vec, rows, meta["labels"])
    with torch.inference_mode():
        predicted = model(x).argmax(1).tolist()
    report = metrics(y.tolist(), predicted, meta["labels"])
    report["examples"] = len(rows)
    report["errors"] = [{"query": text, "expected": label, "predicted": meta["labels"][p]} for (text, label), p in zip(rows, predicted) if meta["labels"][p] != label]
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")


def predict(args):
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    model, vec, meta = load(args.model)
    x = torch.from_numpy(vec.transform([normalize(args.query)]).toarray().astype(np.float32))
    with torch.inference_mode():
        probs = model(x).softmax(dim=1)[0]
    score, index = probs.max(dim=0)
    candidate = meta["labels"][index.item()]
    reason = "no_known_features" if not torch.any(x) else "low_score" if score.item() < args.threshold else None
    print(json.dumps({"query": args.query, "context": candidate if reason is None else None,
                      "candidate_context": candidate, "score": round(score.item(), 6),
                      "needs_review": reason is not None, "reason": reason}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--validation", help="Optional explicit validation file in the same JSON format")
    p.add_argument("--output", default="artifacts")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=train)
    p = commands.add_parser("evaluate")
    p.add_argument("--model", default="artifacts")
    p.add_argument("--data", required=True)
    p.add_argument("--output")
    p.set_defaults(func=evaluate)
    p = commands.add_parser("predict")
    p.add_argument("--model", default="artifacts")
    p.add_argument("--query", required=True)
    p.add_argument("--threshold", type=float, default=0.0, help="Optional uncalibrated softmax cutoff; tune on validation data")
    p.set_defaults(func=predict)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
