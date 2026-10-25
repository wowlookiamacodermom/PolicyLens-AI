
"""Simple, evidence-grounded policy summaries for PolicyLens AI."""
from __future__ import annotations

import re


TOPICS = {
    "Coverage": [
        "coverage", "covered", "benefit", "benefits", "sum insured",
        "hospitalisation", "hospitalization", "medical expenses",
    ],
    "Waiting periods": [
        "waiting period", "waiting periods", "pre-existing disease",
        "pre existing disease", "30-day waiting", "thirty day waiting",
    ],
    "Exclusions": [
        "exclusion", "exclusions", "excluded", "not covered",
        "shall not be covered",
    ],
    "Limits and cost sharing": [
        "sub-limit", "sublimit", "co-payment", "copayment",
        "deductible", "sum insured", "maximum limit", "room rent limit",
    ],
    "Claims": [
        "claim procedure", "claims procedure", "claim intimation",
        "reimbursement", "cashless", "pre-authorisation",
        "pre-authorization", "claim documents",
    ],
}


def _sentences(text: str) -> list[str]:
    """Split extracted PDF text into sentence-like pieces."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+|(?<=•)\s+", text)
    return [part.strip(" •\t-") for part in parts if len(part.strip()) >= 25]


def summarize_policy(pages: list[dict], max_per_topic: int = 3) -> dict:
    """
    Return a simple, topic-organized summary based only on uploaded policy text.
    Each item includes its source page for verification.
    """
    result = {}

    for topic, keywords in TOPICS.items():
        matches = []
        seen = set()

        for page in pages:
            page_number = page.get("page", "?")
            page_text = page.get("text", "")

            for sentence in _sentences(page_text):
                lower = sentence.lower()
                matched = [word for word in keywords if word in lower]

                if not matched:
                    continue

                # Prefer clauses containing several topic-specific terms.
                score = len(matched)
                normalized = lower[:250]

                if normalized in seen:
                    continue
                seen.add(normalized)

                matches.append({
                    "text": sentence,
                    "page": page_number,
                    "score": score,
                })

        matches.sort(key=lambda item: item["score"], reverse=True)

        result[topic] = [
            {"text": item["text"], "page": item["page"]}
            for item in matches[:max_per_topic]
        ]

    return result

"""Simple, evidence-grounded policy summaries for PolicyLens AI."""

import re


TOPICS = {
    "Coverage": [
        "coverage", "covered", "benefit", "benefits", "sum insured",
        "hospitalisation", "hospitalization", "medical expenses",
    ],
    "Waiting periods": [
        "waiting period", "waiting periods", "pre-existing disease",
        "pre existing disease", "30-day waiting", "thirty day waiting",
    ],
    "Exclusions": [
        "exclusion", "exclusions", "excluded", "not covered",
        "shall not be covered",
    ],
    "Limits and cost sharing": [
        "sub-limit", "sublimit", "co-payment", "copayment",
        "deductible", "sum insured", "maximum limit", "room rent limit",
    ],
    "Claims": [
        "claim procedure", "claims procedure", "claim intimation",
        "reimbursement", "cashless", "pre-authorisation",
        "pre-authorization", "claim documents",
    ],
}


def _sentences(text: str) -> list[str]:
    """Split extracted PDF text into sentence-like pieces."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+|(?<=•)\s+", text)
    return [part.strip(" •\t-") for part in parts if len(part.strip()) >= 25]


def summarize_policy(pages: list[dict], max_per_topic: int = 3) -> dict:
    """
    Return a simple, topic-organized summary based only on uploaded policy text.
    Each item includes its source page for verification.
    """
    result = {}

    for topic, keywords in TOPICS.items():
        matches = []
        seen = set()

        for page in pages:
            page_number = page.get("page", "?")
            page_text = page.get("text", "")

            for sentence in _sentences(page_text):
                lower = sentence.lower()
                matched = [word for word in keywords if word in lower]

                if not matched:
                    continue

                # Prefer clauses containing several topic-specific terms.
                score = len(matched)
                normalized = lower[:250]

                if normalized in seen:
                    continue
                seen.add(normalized)

                matches.append({
                    "text": sentence,
                    "page": page_number,
                    "score": score,
                })

        matches.sort(key=lambda item: item["score"], reverse=True)

        result[topic] = [
            {"text": item["text"], "page": item["page"]}
            for item in matches[:max_per_topic]
        ]

    return result
