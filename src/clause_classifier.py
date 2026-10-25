"""Clause classifier: fine-tuned transformer (preferred), sklearn model, or keyword fallback.

Labels are the nine classes used in the training data and in the Colab notebook:
claims, coverage, definition, exclusion, limit, other, premium, renewal, waiting_period.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "clause_classifier.joblib"
TRANSFORMER_PATH = ROOT / "models" / "clause-classifier"  # produced by the Colab notebook
MAX_LEN = 256

LABELS = [
    "claims", "coverage", "definition", "exclusion", "limit",
    "other", "premium", "renewal", "waiting_period",
]

KEYWORDS = {
    "waiting_period": ["waiting period", "pre-existing disease", "initial waiting", "cooling off", "months from the commencement"],
    "exclusion": ["excluded", "exclusion", "excludes", "excluding", "not covered", "shall not cover",
                  "not payable", "not eligible", "will not be covered", "expenses shall be excluded"],
    "coverage": ["covered", "coverage", "benefit", "hospitalisation", "hospitalization", "medical expenses"],
    "claims": ["claim", "notify", "intimation", "documents required", "reimbursement", "cashless"],
    "limit": ["sub-limit", "sublimit", "sum insured", "deductible", "co-payment", "co-pay", "room rent limit"],
    "premium": ["premium", "premium payable", "premium amount", "premium payment", "instalment", "installment"],
    "renewal": ["renewal", "renewed", "policy renewal", "grace period", "portability", "cancellation", "free look"],
    "definition": ["means", "defined as", "definition", "shall mean"],
}


@lru_cache(maxsize=1)
def _transformer_bundle():
    """Load (tokenizer, model, id->label). Labels come from labels.json, not from config.id2label."""
    if not (TRANSFORMER_PATH / "config.json").exists():
        return None
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(TRANSFORMER_PATH))
        model = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_PATH)).eval()
        f = TRANSFORMER_PATH / "labels.json"
        if f.exists():
            label2id = json.loads(f.read_text(encoding="utf-8"))
            id2label = {int(v): k for k, v in label2id.items()}
        else:
            id2label = {i: l for i, l in enumerate(LABELS)}
        return tok, model, id2label
    except Exception:
        return None


@lru_cache(maxsize=1)
def _sklearn_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def keyword_classify(text: str) -> tuple[str, float]:
    """Classify using keyword counts (last-resort fallback, not a trained model)."""
    t = re.sub(r"\s+", " ", text.lower())
    scores = {label: sum(t.count(k) for k in terms) for label, terms in KEYWORDS.items()}
    label, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        return "other", 0.0
    total = sum(scores.values())
    return label, min(0.95, score / max(total, 1))


def predict_clause(text: str) -> tuple[str, float, str]:
    """Return (label, confidence, method). Uses the best available model; no hand-written overrides."""
    bundle = _transformer_bundle()
    if bundle is not None:
        try:
            import torch
            tok, model, id2label = bundle
            batch = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(model(**batch).logits[0], dim=-1)
            idx = int(probs.argmax())
            return id2label[idx], float(probs[idx]), "fine-tuned transformer"
        except Exception:
            pass

    model = _sklearn_model()
    if model is not None:
        try:
            label = model.predict([text])[0]
            conf = float(max(model.predict_proba([text])[0])) if hasattr(model, "predict_proba") else 0.0
            return str(label), conf, "TF-IDF + Logistic Regression"
        except Exception:
            pass

    label, conf = keyword_classify(text)
    return label, conf, "keyword fallback (not a trained model)"
