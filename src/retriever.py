
"""Passage retrieval for PolicyLens AI."""
from dataclasses import dataclass
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Passage:
    text: str
    page: int
    section: str = ""


def chunk_pages(pages: list[dict], chunk_words: int = 180,
                overlap: int = 35) -> list[Passage]:
    chunks = []
    for page in pages:
        words = re.findall(r"\S+", page.get("text", ""))
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_words)
            chunks.append(Passage(" ".join(words[start:end]), int(page["page"])))
            if end == len(words):
                break
            start = max(start + 1, end - overlap)
    return chunks



EXPANSIONS = {
    "waiting": [
        "waiting period",
        "waiting periods",
        "pre-existing disease",
        "pre existing disease",
        "specific disease waiting period",
        "initial waiting period",
        "specified illness",
        "maternity waiting period",
    ],
    "exclusion": [
        "exclusions",
        "excluded",
        "not covered",
        "permanent exclusions",
        "specific exclusions",
    ],
    "claim": [
        "claims",
        "claim procedure",
        "claim documents",
        "documents required for claim",
        "claim intimation",
        "reimbursement claim",
        "cashless claim",
        "claim settlement",
    ],
    "limit": [
        "limits",
        "sub-limit",
        "deductible",
        "co-payment",
        "sum insured",
        "maximum payable amount",
    ],
    "renewal": [
        "policy renewal",
        "renewal conditions",
        "renewal premium",
        "renewal date",
        "grace period",
        "renewal of the policy",
    ],
    "premium": [
        "premium payable",
        "premium amount",
        "premium payment",
        "premium due date",
        "mode of premium payment",
    ],
}



class Retriever:
    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=50000,
            sublinear_tf=True,
        )
        self.matrix = (
            self.vectorizer.fit_transform([p.text for p in passages])
            if passages else None
        )

    def search(self, query: str, top_k: int = 5):
        if not query.strip() or self.matrix is None:
            return []

        query_lower = query.lower()
        terms = [query]
        for key, alternatives in EXPANSIONS.items():
            if key in query_lower:
                terms.extend(alternatives)

        expanded_query = " ".join(dict.fromkeys(terms))
        q = self.vectorizer.transform([expanded_query])
        scores = cosine_similarity(q, self.matrix)[0]

        # Boost passages containing exact topic terms.

        # Boost passages that match the question's specific topic.
        topic_rules = {
            "waiting": [
                "waiting period",
                "pre-existing disease",
                "pre existing disease",
                "specific disease",
                "specified illness",
            ],
            "exclusion": [
                "exclusion",
                "excluded",
                "not covered",
            ],
            "claim": [
                "claim documents",
                "documents required for claim",
                "claim procedure",
                "claim intimation",
                "reimbursement",
                "cashless",
            ],
            "renewal": [
                "policy renewal",
                "renewal of the policy",
                "renewal conditions",
                "renewed",
                "grace period",
            ],
            "premium": [
                "premium payable",
                "premium amount",
                "premium payment",
                "premium due date",
            ],
            "limit": [
                "sum insured",
                "sub-limit",
                "deductible",
                "co-payment",
                "maximum limit",
            ],
        }

        active_topic = next(
            (topic for topic in topic_rules if topic in query_lower),
            None,
        )

        if active_topic:
            for i, passage in enumerate(self.passages):
                text = passage.text.lower()
                matches = sum(
                    1 for term in topic_rules[active_topic]
                    if term in text
                )

                if matches:
                    scores[i] += min(matches * 0.08, 0.24)

        ids = scores.argsort()[::-1][:max(1, top_k)]
        return [
            (self.passages[i], float(scores[i]))
            for i in ids if scores[i] > 0
        ]
