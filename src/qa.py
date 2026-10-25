
"""Evidence-first question answering for PolicyLens AI."""
from __future__ import annotations

from functools import lru_cache
import re


@lru_cache(maxsize=1)
def _qa_pipeline():
    """Load extractive QA if the model is available."""
    try:
        from transformers import pipeline

        return pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
        )
    except Exception:
        return None


def _sentences(text: str) -> list[str]:
    """Split extracted policy text into readable sentences."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if len(part.strip()) > 25]


def _normalise(text: str) -> str:
    """Normalise text for duplicate detection."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _topic_terms(question: str) -> list[str]:
    """Return useful policy phrases for the question topic."""
    q = question.lower()
    terms = []

    topics = {
        "waiting": [
            "waiting period",
            "waiting periods",
            "pre-existing disease",
            "pre existing disease",
            "specific disease",
            "specified illness",
            "initial waiting period",
            "no waiting period",
            "continuous coverage",
        ],
        "exclusion": [
            "exclusion",
            "excluded",
            "not covered",
            "shall not be covered",
        ],
        "claim": [
            "claim procedure",
            "claim documents",
            "claim intimation",
            "reimbursement",
            "cashless",
        ],
        "limit": [
            "sum insured",
            "sub-limit",
            "deductible",
            "co-payment",
            "maximum limit",
        ],
        "renewal": [
            "renewal",
            "renewed",
            "grace period",
            "policy period",
        ],
        "premium": [
            "premium payable",
            "premium amount",
            "premium payment",
        ],
    }

    for topic, phrases in topics.items():
        if topic in q:
            terms.extend(phrases)

    return terms


def _relevant_sentences(question: str, retrieved, limit: int = 5):
    """Rank relevant sentences and remove duplicate wording."""
    q = question.lower()
    question_words = {
        word for word in re.findall(r"[a-z0-9]+", q)
        if len(word) >= 3
    }
    topic_terms = _topic_terms(question)
    candidates = []
    seen = set()

    for passage, retrieval_score in retrieved:
        for sentence in _sentences(passage.text):
            normalised = _normalise(sentence)

            if not normalised or normalised in seen:
                continue

            # Ignore very short fragments and administrative boilerplate.
            if any(
                phrase in normalised
                for phrase in [
                    "registered office address",
                    "toll free number",
                    "toll free no",
                    "cin ",
                    "irdai reg no",
                    "mailing address",
                ]
            ):
                continue

            sentence_words = set(normalised.split())
            overlap = len(question_words & sentence_words)

            phrase_matches = sum(
                1 for phrase in topic_terms
                if phrase in normalised
            )

            # Prefer policy phrases that directly match the requested topic.
            score = (
                overlap * 0.15
                + phrase_matches * 1.5
                + min(float(retrieval_score), 1.0)
            )

            if score <= 0:
                continue

            seen.add(normalised)
            candidates.append(
                (
                    score,
                    sentence,
                    int(passage.page),
                    float(retrieval_score),
                )
            )

    candidates.sort(key=lambda item: item[0], reverse=True)

    # Remove near-duplicates, including repeated clauses with small differences.
    selected = []
    selected_tokens = []

    for item in candidates:
        tokens = set(_normalise(item[1]).split())

        is_duplicate = False
        for previous_tokens in selected_tokens:
            union = tokens | previous_tokens
            similarity = (
                len(tokens & previous_tokens) / len(union)
                if union else 0
            )
            if similarity >= 0.82:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(item)
            selected_tokens.append(tokens)

        if len(selected) >= limit:
            break

    return selected



def _evidence(retrieved, limit: int = 3) -> list[dict]:
    """Build concise evidence records and skip near-duplicate passages."""
    results = []
    seen_tokens = []

    for passage, score in retrieved:
        text = re.sub(r"\s+", " ", passage.text).strip()
        normalised = _normalise(text)

        if not normalised:
            continue

        tokens = set(normalised.split())

        # Skip passages that are almost the same as one already selected.
        duplicate = False
        for previous_tokens in seen_tokens:
            union = tokens | previous_tokens
            similarity = (
                len(tokens & previous_tokens) / len(union)
                if union else 0
            )
            if similarity >= 0.75:
                duplicate = True
                break

        if duplicate:
            continue

        seen_tokens.append(tokens)
        results.append({
            "page": int(passage.page),
            "text": text[:1200],
            "score": round(float(score), 3),
        })

        if len(results) >= limit:
            break

    return results


def _plain_english_explanation(question: str, sentences: list[str]) -> str:
    """Explain the topic without inventing policy-specific facts."""
    q = question.lower()

    if "waiting" in q:
        return (
            "A waiting period is a time during which certain conditions or "
            "treatments may not be covered. The exact duration and the "
            "conditions it applies to must be checked in the policy wording below."
        )

    if "exclusion" in q or "not covered" in q:
        return (
            "An exclusion is a condition or expense that the policy may not "
            "cover. Check the cited wording for exceptions and conditions."
        )

    if "claim" in q or "reimbursement" in q or "cashless" in q:
        return (
            "This concerns how to make or receive payment for a claim. "
            "Check the cited wording for required documents, deadlines, "
            "and claim conditions."
        )

    if "renewal" in q or "renew" in q:
        return (
            "Renewal means continuing the policy for another period. "
            "Check the cited wording for the applicable conditions and deadlines."
        )

    if "limit" in q or "sum insured" in q or "co-payment" in q:
        return (
            "These clauses explain limits on the insurer's payment or costs "
            "you may need to pay yourself. Check the cited wording for the "
            "exact amounts and conditions."
        )

    return (
        "The passages below may help answer your question. Check the cited "
        "wording and any related conditions or exceptions."
    )



def answer_question(question: str, retrieved: list[tuple[object, float]]) -> dict:
    """Answer using retrieved policy text, with a cautious fallback."""
    if not question or not question.strip():
        return {
            "answer": "Please enter a question about the uploaded policy.",
            "evidence": [],
            "method": "no question",
        }

    if not retrieved:
        return {
            "answer": (
                "I couldn't find relevant text in the uploaded policy. "
                "Try different wording or inspect the policy manually."
            ),
            "evidence": [],
            "method": "no evidence",
        }

    retrieved = sorted(retrieved, key=lambda item: float(item[1]), reverse=True)
    best_score = float(retrieved[0][1])

    if best_score < 0.025:
        return {
            "answer": (
                "I couldn't find strong evidence in the uploaded policy "
                "to answer confidently. Try rephrasing your question."
            ),
            "evidence": _evidence(retrieved),
            "method": "low-confidence abstention",
        }

    selected = _relevant_sentences(question, retrieved, limit=3)
    
    if selected:
        evidence = [
            {
                "page": page,
                "text": sentence,
                "score": round(score, 3),
            }
            for _, sentence, page, score in selected
        ]

        # Group the answer into readable, cited points.
        answer_lines = [
            f"- {item['text']} *(page {item['page']})*"
            for item in evidence
        ]

        
        plain_english = _plain_english_explanation(
            question,
            [item["text"] for item in evidence],
        )

        answer = (
            "**Explanation in simple English:**\n\n"
            + plain_english
            + "\n\n**Relevant wording found in the uploaded policy:**\n\n"
            + "\n".join(answer_lines)
            + "\n\n**Please note:** These are extracts, not a definitive "
            "interpretation or claim-approval decision. Check the complete "
            "clauses, policy certificate, applicable section, and exceptions."
        )

        

        return {
            "answer": answer,
            "evidence": _evidence(retrieved),
            "method": "deduplicated relevant-sentence retrieval",
        }

    # Final fallback: show the best available passages without inventing an answer.
    return {
        "answer": (
            "I found potentially relevant passages, but couldn't identify "
            "a sufficiently clear answer. Please inspect the evidence below."
        ),
        "evidence": _evidence(retrieved),
        "method": "evidence-only fallback",
    }
