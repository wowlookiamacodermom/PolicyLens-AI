from src.clause_classifier import keyword_classify
from src.retriever import chunk_pages, Retriever

def test_keyword_classifier_returns_label():
    label, score = keyword_classify("The policy excludes cosmetic treatment.")
    assert label == "exclusion"
    assert score > 0

def test_chunk_pages_preserves_page_number():
    chunks = chunk_pages([{"page": 7, "text": "waiting period " * 80}], chunk_words=30, overlap=5)
    assert chunks
    assert all(c.page == 7 for c in chunks)

def test_retriever_returns_matching_passage():
    passages = chunk_pages([{"page": 2, "text": "The waiting period for pre-existing diseases is described in this section."}])
    retriever = Retriever(passages)
    results = retriever.search("pre-existing disease waiting period", top_k=1)
    assert results
    assert results[0][0].page == 2
