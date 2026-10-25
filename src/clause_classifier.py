"""Clause classifier: fine-tuned DistilBERT, trained sklearn model, or keyword fallback."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import re
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "clause_classifier.joblib"
TRANSFORMER_PATH = ROOT / "models" / "distilbert-clause-classifier"

KEYWORDS = {
    "waiting_period": ["waiting period", "pre-existing disease", "initial waiting", "months from the commencement"],
    "exclusion": [
    "excluded",
    "exclusion",
    "excludes",
    "excluding",
    "not covered",
    "shall not cover",
    "not payable",
    "not eligible",
    "will not be covered",
    "expenses shall be excluded",
],
    "coverage": ["covered", "coverage", "benefit", "hospitalisation", "hospitalization", "medical expenses"],
    "claims": ["claim", "notify", "intimation", "documents required", "reimbursement"],
    "limit": ["sub-limit", "sublimit", "sum insured", "deductible", "co-payment", "co-pay", "room rent limit"],
    "premium": [
    "premium",
    "premium payable",
    "premium amount",
    "premium payment",
    "instalment",
    "installment",
    "taxes",
],
    "renewal": [
    "renewal",
    "renewed",
    "policy may be renewed",
    "policy renewal",
    "renewal of the policy",
    "renewal premium",
    "grace period",
    "portability",
    "cancellation",
],
    "definition": ["means", "defined as", "definition", "shall mean"],
}

@lru_cache(maxsize=1)
def _transformer_classifier():
    if not TRANSFORMER_PATH.exists():
        return None
    try:
        from transformers import pipeline
        return pipeline("text-classification", model=str(TRANSFORMER_PATH), tokenizer=str(TRANSFORMER_PATH), truncation=True)
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
    """Classify using keyword counts as a fallback."""
    t = re.sub(r"\s+", " ", text.lower())

    # Ignore obvious administrative/contact details.
    if any(term in t for term in [
        "mailing address",
        "registered office address",
        "toll free no",
        "toll-free number",
        "cin:",
        "irdai reg. no",
    ]):
        return "other", 0.90

    scores = {
        label: sum(t.count(keyword) for keyword in terms)
        for label, terms in KEYWORDS.items()
    }

    label, score = max(scores.items(), key=lambda item: item[1])

    if score == 0:
        return "other", 0.0

    total = sum(scores.values())
    return label, min(0.95, score / max(total, 1))


def predict_clause(text: str) -> tuple[str, float, str]:
    t = re.sub(r"\s+", " ", text.lower())
    if any(term in t for term in [
        "mailing address",
        "registered office address",
        "toll free no",
        "toll-free number",
        "cin:",
        "irdai reg. no",
    ]):
        return "other", 0.90, "administrative-text rule"

    # Correct clear, common patterns before consulting the model.
    if any(phrase in t for phrase in [
        "claim documents", "documents required for claim",
        "claim procedure", "claim intimation", "reimbursement claim",
        "submit the claim", "submit claim",
    ]):
        return "claims", 0.80, "rule-based correction"

    if any(phrase in t for phrase in [
        "renewed after", "policy renewal", "renewal of the policy",
        "may be renewed", "renewal means", "at the time of renewal",
    ]):
        return "renewal", 0.80, "rule-based correction"

    if any(phrase in t for phrase in [
        "excluded from coverage", "not covered under this policy",
        "shall not be covered", "is excluded from",
    ]):
        return "exclusion", 0.80, "rule-based correction"

    transformer = _transformer_classifier()
    if transformer is not None:
        try:
            result = transformer(text[:4000])[0]
            return (
                str(result["label"]).replace("LABEL_", ""),
                float(result.get("score", 0.0)),
                "fine-tuned DistilBERT",
            )
        except Exception:
            pass

    model = _sklearn_model()
    if model is not None:
        try:
            label = model.predict([text])[0]
            confidence = (
                float(max(model.predict_proba([text])[0]))
                if hasattr(model, "predict_proba") else 0.0
            )
            return str(label), confidence, "TF-IDF + Logistic Regression"
        except Exception:
            pass

    label, confidence = keyword_classify(text)
    return label, confidence, "keyword fallback (not a trained model)"

    transformer = _transformer_classifier()
    if transformer is not None:
        try:
            result = transformer(text[:4000])[0]
            return str(result["label"]).replace("LABEL_", ""), float(result.get("score", 0.0)), "fine-tuned DistilBERT"
        except Exception:
            pass
    model = _sklearn_model()
    if model is not None:
        try:
            label = model.predict([text])[0]
            confidence = float(max(model.predict_proba([text])[0])) if hasattr(model, "predict_proba") else 0.0
            return str(label), confidence, "TF-IDF + Logistic Regression"
        except Exception:
            pass
    label, confidence = keyword_classify(text)
    return label, confidence, "keyword fallback (not a trained model)"
