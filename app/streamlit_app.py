from __future__ import annotations
import sys, json, tempfile
from pathlib import Path
import streamlit as st
import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.retriever import chunk_pages, Retriever
from src.clause_classifier import predict_clause
from src.qa import answer_question
from src.policy_summary import summarize_policy

st.set_page_config(page_title="PolicyLens AI", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.8rem; max-width: 1200px;}
.hero {padding: 1.5rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#10243a,#214e68); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin-bottom:.25rem;}
.hero p {color:#e2edf5; margin-bottom:0;}
.small-muted {color: #64748b; font-size:.9rem;}
div[data-testid="stMetric"] {background: rgba(120,140,160,.08); border:1px solid rgba(120,140,160,.22); padding: .8rem; border-radius: 12px;}
</style>
<div class="hero">
  <h1>🛡️ PolicyLens AI</h1>
  <p>Understand your health insurance policy — with clause classification, evidence retrieval and page-level references.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.write("Educational prototype for Indian health insurance policy documents.")
    st.info("Not a claim-approval engine or a substitute for insurer/professional advice.")
    top_k = st.slider("Evidence passages to retrieve", 1, 8, 5)
    st.caption("Optional transformer models may download weights on first use.")

uploaded = st.file_uploader("Upload a health insurance policy PDF", type=["pdf"])
if uploaded:
    if uploaded.size > 25 * 1024 * 1024:
        st.error("This prototype limits uploads to 25 MB.")
        st.stop()
    with st.spinner("Extracting text and preparing evidence..."):
        content = uploaded.getvalue()
        try:
            with fitz.open(stream=content, filetype="pdf") as doc:
                pages = [{"page": i+1, "text": p.get_text("text").strip()} for i,p in enumerate(doc)]
        except Exception as e:
            st.error(f"Could not read this PDF: {e}")
            st.stop()
        pages = [p for p in pages if p["text"]]
        if not pages:
            st.error("No selectable text was found. This may be a scanned PDF. OCR support is not enabled in this UI build; run OCR first or add OCR support.")
            st.stop()
        passages = chunk_pages(pages)
        retriever = Retriever(passages)
        st.session_state["policy_pages"] = pages
        st.session_state["policy_passages"] = passages
        st.session_state["policy_retriever"] = retriever
        st.session_state["policy_name"] = uploaded.name
    st.success(f"Loaded {uploaded.name}")
    c1,c2,c3 = st.columns(3)
    c1.metric("Pages with text", len(pages))
    c2.metric("Text passages", len(passages))
    c3.metric("Extracted words", sum(len(p["text"].split()) for p in pages))

if "policy_retriever" not in st.session_state:
    st.markdown("### What you can do")
    a,b,c = st.columns(3)
    a.markdown("#### 📋 Summarize\nRead key clauses in simpler language.")
    b.markdown("#### 🧠 Classify\nIdentify exclusions, coverage and waiting periods.")
    c.markdown("#### 🔎 Ask questions\nRetrieve relevant text and show its page.")
    st.caption("Upload a PDF to begin. Try a public policy wording, not a private policy schedule.")
    st.stop()

pages = st.session_state["policy_pages"]
retriever = st.session_state["policy_retriever"]
passages = st.session_state["policy_passages"]
tab_summary, tab_qa, tab_clauses, tab_text = st.tabs(["✨ Overview", "💬 Ask the policy", "🏷️ Clause explorer", "📄 Extracted text"])

summary = summarize_policy(pages)

with tab_summary:
    st.subheader("Policy overview")
    all_text = "\n".join(p["text"] for p in pages)
    st.write(f"**Document:** {st.session_state.get('policy_name','Uploaded policy')}")
    st.write("This overview is generated from text extraction. Review the source wording before relying on it.")

    
    st.markdown("### 📝 Policy summary in simple language")
    st.caption(
        "Key points extracted from your uploaded policy. "
        "Use the page references to verify the original wording."
    )

    for topic, items in summary.items():
        with st.expander(topic, expanded=True):
            if items:
                for item in items:
                    st.markdown(
                        f"- {item['text']} "
                        f"*(Source: page {item['page']})*"
                    )
            else:
                st.info(
                    f"No clear text about {topic.lower()} was found. "
                    "Check the full policy wording."
                )

    with st.expander("Quick document signals", expanded=True):
        terms = {
            "Coverage": ["covered", "coverage", "benefit"],
            "Exclusions": ["exclusion", "excluded", "not covered"],
            "Waiting periods": ["waiting period", "pre-existing disease"],
            "Claims": ["claim", "reimbursement", "intimation"],
            "Limits": ["sub-limit", "deductible", "co-payment", "sum insured"],
        }
        cols = st.columns(len(terms))
        low = all_text.lower()
        for col,(label,words) in zip(cols,terms.items()):
            col.metric(label, sum(low.count(w) for w in words))
    st.markdown("**Important:** term counts are navigation hints, not verified coverage findings.")
    st.text_area("Opening extracted text", all_text[:5000], height=240)

with tab_qa:
    st.subheader("Policy summary at a glance")

    with st.expander("📋 Coverage, waiting periods, exclusions, limits and claims", expanded=True):
        for topic, items in summary.items():
            st.markdown(f"**{topic}**")
            if items:
                for item in items:
                    st.markdown(
                        f"- {item['text']} *(page {item['page']})*"
                    )
            else:
                st.caption(f"No clear text found for {topic.lower()}.")

    st.divider()
    st.subheader("Ask a question about this policy")

    question = st.text_input("Example: What does the policy say about waiting periods?", key="policy_question")
    if st.button("Find answer", type="primary", disabled=not question.strip()):
        with st.spinner("Searching the policy..."):
            results = retriever.search(question, top_k=top_k)
            response = answer_question(question, results)
        st.markdown("### Answer")
        st.write(response["answer"])
        st.caption(f"Method: {response['method']}")
        st.markdown("### Evidence")
        if response["evidence"]:
            for ev in response["evidence"]:
                with st.expander(f"Page {ev['page']} · relevance {ev['score']:.3f}", expanded=True):
                    st.write(ev["text"])
        else:
            st.warning("No evidence passages were returned.")
        st.caption("Verify full clauses, exceptions, schedules and endorsements. Retrieval relevance is not proof of coverage.")

with tab_clauses:
    st.subheader("Clause explorer")
    st.write("This view predicts clause categories. If a trained model has not been built, the app uses a keyword fallback and labels it accordingly.")
    query = st.text_input("Filter extracted text", placeholder="e.g. waiting period, exclusions, claims")
    candidates = []
    for p in pages:
        # Split page text into paragraph-like blocks; fall back to full lines for PDFs with unusual formatting.
        blocks = [x.strip() for x in p["text"].split("\n") if len(x.strip()) >= 45]
        for block in blocks:
            if query and query.lower() not in block.lower():
                continue
            label, conf, method = predict_clause(block)
            candidates.append((p["page"], block, label, conf, method))
    if not candidates:
        st.info("No matching clause-like text found. Try a broader filter.")
    else:
        st.caption(f"Showing up to 100 of {len(candidates)} extracted blocks.")
        for page, text, label, conf, method in candidates[:100]:
            with st.expander(f"Page {page} · {label.replace('_',' ').title()} · {method}"):
                st.write(text)
                st.caption(f"Model confidence: {conf:.2f} (fallback scores are heuristic, not calibrated probabilities).")

with tab_text:
    st.subheader("Extracted text by page")
    selected = st.selectbox("Choose page", [p["page"] for p in pages])
    page = next(p for p in pages if p["page"] == selected)
    st.text_area("Page text", page["text"], height=400)
