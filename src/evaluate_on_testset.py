"""Evaluate the saved transformer on the fixed 300-row test set and write reports/evaluation.md.

Run from the project root:
    python -m src.evaluate_on_testset

Test labels were produced by an LLM (Claude) and are NOT human-verified, so the scores
below mean "agreement with LLM-made labels".
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "clause-classifier"
TEST = ROOT / "data" / "eval" / "test_set_labeled.csv"
REPORT = ROOT / "reports" / "evaluation.md"


def main():
    if not (MODEL_DIR / "config.json").exists():
        raise SystemExit(f"No model found in {MODEL_DIR}. Unzip the Colab download into models/ first.")
    df = pd.read_csv(TEST)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    labels = [id2label[i] for i in sorted(id2label)]

    preds = []
    with torch.no_grad():
        for i in range(0, len(df), 32):
            batch = tok(list(df.text.astype(str).values[i:i + 32]), truncation=True, max_length=256,
                        padding=True, return_tensors="pt")
            preds.append(model(**batch).logits.argmax(-1).numpy())
    pred = [id2label[i] for i in np.concatenate(preds)]
    true = df.claude_label.tolist()

    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average="macro", zero_division=0)
    rep = classification_report(true, pred, labels=labels, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(true, pred, labels=labels), index=labels, columns=labels)

    text = (
        "# Clause classifier evaluation\n\n"
        f"- Test rows: {len(df)} (random paragraphs from {df.policy_id.nunique()} policy documents, never used for training)\n"
        "- Test labels: made by an LLM (Claude), not human-verified; scores mean agreement with those labels\n"
        f"- **Accuracy: {acc:.4f}**\n- **Macro-F1: {f1:.4f}**\n\n"
        f"## Per-class report\n```\n{rep}\n```\n\n"
        f"## Confusion matrix (rows = true label, columns = predicted)\n```\n{cm.to_string()}\n```\n"
    )
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved {REPORT}")


if __name__ == "__main__":
    main()
