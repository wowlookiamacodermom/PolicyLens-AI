from __future__ import annotations

import hashlib
import html
import sys
from pathlib import Path

import streamlit as st
try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF versions
    import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.retriever import chunk_pages, Retriever
from src.clause_classifier import predict_clause
from src.qa import answer_question
from src.policy_summary import summarize_policy

st.set_page_config(page_title="PolicyLens AI", page_icon="📑", layout="centered",
                   initial_sidebar_state="auto")

# --------------------------------------------------------------------------------------
# Clause categories: each one gets a "highlighter" colour AND a written name, so colour is
# never the only signal.  (background, underline, display name, what it means in plain words)
# --------------------------------------------------------------------------------------
CLAUSES = {
    "waiting_period": ("#FFF0A3", "#B8960C", "Waiting period", "Time you must wait before something is covered"),
    "exclusion":      ("#FFD6D2", "#B3392F", "Exclusion", "Things the policy will not pay for"),
    "coverage":       ("#CFEFD8", "#2E7D4F", "Coverage", "Things the policy will pay for"),
    "limit":          ("#FFE3B5", "#B26A00", "Limit or cost share", "Caps, deductibles and co-payments"),
    "claims":         ("#CFE5FF", "#2563A8", "Claims", "How to ask the insurer to pay"),
    "premium":        ("#CDEFEC", "#1B7A73", "Premium", "What you pay and when"),
    "renewal":        ("#F8D9EE", "#A23B84", "Renewal and cancellation", "Renewing, switching or ending the policy"),
    "definition":     ("#E3DCFB", "#5B46B0", "Definition", "What a term in the policy means"),
    "other":          ("#E6E9EF", "#6B7280", "Other", "Headings, addresses and general conditions"),
}
ORDER = ["waiting_period", "exclusion", "coverage", "limit", "claims", "premium", "renewal", "definition", "other"]


def clause_info(key: str):
    return CLAUSES.get(key, ("#E6E9EF", "#6B7280", key.replace("_", " ").title(), ""))


def tag(key: str) -> str:
    bg, line, name, _ = clause_info(key)
    return f'<span class="tag" style="background:{bg};border-bottom-color:{line}">{html.escape(name)}</span>'


def esc(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def md_lite(text: str) -> str:
    """Turn the small amount of markdown the QA module returns (bold, italics, bullets) into safe HTML."""
    import re
    out, in_list = [], False
    for line in text.split("\n"):
        line = html.escape(line.strip())
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line)
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{line[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>"); in_list = False
        if line:
            out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def paragraphs(text: str, min_len: int = 45, max_len: int = 700) -> list[str]:
    """PDF text arrives one visual line at a time. Join lines back into paragraph-sized blocks."""
    import re
    blocks, cur = [], ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        starts_new = bool(re.match(r"^(\(?[0-9ivxIVX]+[.)]|[a-z]\)|[A-Z][A-Z .&-]{8,}$|•|-)\s*", line))
        if cur and (cur.rstrip().endswith((".", ":", ";")) or starts_new or len(cur) > max_len):
            blocks.append(cur)
            cur = ""
        cur = f"{cur} {line}".strip()
    if cur:
        blocks.append(cur)
    return [b for b in blocks if len(b) >= min_len]


def confidence_word(conf: float, method: str) -> str:
    if "keyword" in method:
        return "Rough guess"
    return "High confidence" if conf >= 0.8 else "Medium confidence" if conf >= 0.55 else "Low confidence"


# --------------------------------------------------------------------------------------
# Look and feel
# --------------------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&family=Source+Serif+4:opsz,wght@8..60,400..600&display=swap');

:root {
  --ink: #172033; --muted: #465267; --line: #D5DBE6; --paper: #F4F6F9; --card: #FFFFFF;
  --ui: 'Bricolage Grotesque', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --doc: 'Source Serif 4', Georgia, 'Times New Roman', serif;
}
html, body, .stApp, [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input, .stApp textarea,
.stApp div[data-baseweb="select"] { font-family: var(--ui); }
span[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: "Material Symbols Rounded" !important; }
.stApp { font-size: 1.06rem; line-height: 1.6; }
.block-container { max-width: 880px; padding-top: 2.4rem; padding-bottom: 4rem; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], .stDeployButton { display: none; }

h1, h2, h3, h4 { font-family: var(--ui) !important; color: var(--ink); letter-spacing: -0.02em; font-weight: 700; }
.display { font-size: clamp(2.1rem, 5.2vw, 3.3rem); line-height: 1.05; font-weight: 800; letter-spacing: -0.035em; margin: 0 0 .9rem; max-width: 15ch; }
.lede { font-size: 1.2rem; color: var(--muted); max-width: 52ch; margin: 0 0 1.6rem; }
.brand { font-weight: 700; letter-spacing: -0.01em; font-size: 1.05rem; margin-bottom: 1.6rem; display:flex; align-items:center; gap:.55rem; }
.brand .mark { width: 1.5rem; height: 1.5rem; border-radius: .4rem; background: var(--ink); position: relative; display:inline-block; }
.brand .mark::after { content:""; position:absolute; left:.28rem; right:.28rem; top:.62rem; height:.42rem; background:#FFE066; border-radius:2px; }

.doc { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 1.3rem 1.5rem; margin: 0 0 1.4rem; }
.doc .row { font-family: var(--doc); font-size: 1.12rem; line-height: 1.75; margin: .15rem 0 .55rem; }
.doc .hl { padding: .05em .25em; border-bottom: 2px solid; border-radius: 2px; box-decoration-break: clone; -webkit-box-decoration-break: clone; }
.doc .note { font-size: .9rem; color: var(--muted); margin-top: .9rem; }
.legend { display: flex; flex-wrap: wrap; gap: .45rem .5rem; margin: .2rem 0 1.4rem; }

.tag { display: inline-block; font-size: .85rem; font-weight: 600; padding: .12rem .55rem; border-radius: 4px; border-bottom: 2px solid; color: var(--ink); line-height: 1.5; }

.facts { display: flex; flex-wrap: wrap; gap: 1.8rem; margin: .2rem 0 1.2rem; }
.facts div { font-size: .95rem; color: var(--muted); }
.facts b { display: block; font-size: 1.7rem; color: var(--ink); letter-spacing: -0.02em; line-height: 1.1; }
.docname { font-size: 1.05rem; color: var(--muted); margin: 0 0 .4rem; word-break: break-word; }

.quote { background: var(--card); border: 1px solid var(--line); border-left: 4px solid var(--ink); border-radius: 6px; padding: .8rem 1rem; margin: .5rem 0; }
.quote .src { font-size: .85rem; color: var(--muted); margin-top: .25rem; }
.serif { font-family: var(--doc); font-size: 1.08rem; line-height: 1.7; }
.answer p { margin: 0 0 .7rem; } .answer ul { margin: 0 0 .7rem; padding-left: 1.3rem; } .answer { font-family: var(--doc); font-size: 1.22rem; line-height: 1.7; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 1.1rem 1.3rem; margin: .3rem 0 1rem; }
.meta { font-size: .88rem; color: var(--muted); }
.clause { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: .8rem 1rem; margin: .55rem 0; }
.clause .head { display: flex; flex-wrap: wrap; gap: .4rem .8rem; align-items: center; margin-bottom: .4rem; }

.bar { display: flex; height: 14px; border-radius: 4px; overflow: hidden; border: 1px solid var(--line); margin: .3rem 0 .7rem; }
.bar span { display: block; height: 100%; }

/* controls */
.stButton > button, .stFormSubmitButton > button { font-family: var(--ui); font-weight: 600; border-radius: 8px; min-height: 2.8rem; border: 1.5px solid var(--ink); background: var(--card); color: var(--ink); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"], .stFormSubmitButton > button { background: var(--ink); color: #fff; }
.stButton > button:hover, .stFormSubmitButton > button:hover { background: #2A3A5C; color: #fff; border-color: #2A3A5C; }
:focus-visible { outline: 3px solid #1D4ED8 !important; outline-offset: 2px !important; }
div[data-testid="stFileUploaderDropzone"] { background: var(--card); border: 2px dashed #7C89A3; border-radius: 12px; padding: 1.6rem; }
div[data-testid="stFileUploaderDropzone"] button { border: 1.5px solid var(--ink); color: var(--ink); background: var(--card); font-weight: 600; }
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { border-radius: 8px; }
.stTextInput input { font-size: 1.05rem; }
div[role="radiogroup"] { gap: .4rem; flex-wrap: wrap; }
div[role="radiogroup"] label { background: var(--card); border: 1.5px solid var(--line); border-radius: 999px; padding: .3rem 1rem; cursor: pointer; }
div[role="radiogroup"] label:has(input:checked) { background: var(--ink); border-color: var(--ink); }
div[role="radiogroup"] label:has(input:checked) * { color: #fff !important; }
div[role="radiogroup"] label > div:first-child { display: none; }
details { background: var(--card); border: 1px solid var(--line) !important; border-radius: 8px; }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] ol { padding-left: 1.2rem; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="brand"><span class="mark"></span>PolicyLens AI</div>', unsafe_allow_html=True)
    st.markdown("**How to use it**")
    st.markdown("1. Upload your policy as a PDF.\n2. Read the plain-language summary.\n3. Ask a question and check the page it points to.")
    top_k = st.slider("Passages to show with each answer", 1, 8, 5,
                      help="More passages give more context but take longer to read.")
    st.markdown(
        '<p class="meta">A learning project for Indian health insurance policies. It explains the text you upload. '
        "It cannot tell you whether a claim will be approved. Always check the full policy or ask your insurer.</p>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# Upload + extraction (done once per file, not on every click)
# --------------------------------------------------------------------------------------
if "policy_retriever" not in st.session_state:
    st.markdown('<div class="brand"><span class="mark"></span>PolicyLens AI</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="display">Read your health policy without the jargon.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="lede">Upload the policy PDF. PolicyLens finds the rules that matter, explains them in plain English '
        "and shows the page each one came from.</p>",
        unsafe_allow_html=True,
    )

_loaded = "policy_retriever" in st.session_state
uploaded = (st.sidebar if _loaded else st).file_uploader(
    "Open a different policy PDF" if _loaded else "Upload your policy PDF", type=["pdf"],
    help="Use a PDF where you can select the text. Scanned pages are not supported yet.")

if uploaded:
    content = uploaded.getvalue()
    digest = hashlib.md5(content).hexdigest()
    if st.session_state.get("doc_hash") != digest:
        with st.spinner("Reading the document..."):
            try:
                with fitz.open(stream=content, filetype="pdf") as doc:
                    pages = [{"page": i + 1, "text": p.get_text("text").strip()} for i, p in enumerate(doc)]
            except Exception as e:
                st.error(f"This file could not be opened ({e}). Try downloading the PDF again.")
                st.stop()
            pages = [p for p in pages if p["text"]]
            if not pages:
                st.error("No text was found, so this looks like a scanned PDF. Run it through an OCR tool first, then upload it again.")
                st.stop()
            passages = chunk_pages(pages)
            st.session_state.update(
                doc_hash=digest, policy_pages=pages, policy_passages=passages,
                policy_retriever=Retriever(passages), policy_name=uploaded.name, asked="", view="Summary",
            )
            st.session_state.pop("policy_question", None)
        st.rerun()

if "policy_retriever" not in st.session_state:
    st.markdown(
        """
<div class="doc">
<div class="row"><span class="hl" style="background:#FFF0A3;border-color:#B8960C">A waiting period of 36 months applies to pre-existing conditions.</span></div>
<div class="row"><span class="hl" style="background:#FFD6D2;border-color:#B3392F">We will not pay for cosmetic surgery unless it follows an accident.</span></div>
<div class="row"><span class="hl" style="background:#CFE5FF;border-color:#2563A8">Tell us within 48 hours of admission to claim without paying upfront.</span></div>
<div class="note">Example sentences, not from a real policy. PolicyLens marks every paragraph like this:</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="legend">' + "".join(tag(k) for k in ORDER) + "</div>", unsafe_allow_html=True)
    st.markdown(
        '<p class="meta">Tip: try a public policy wording from your insurer\'s website, not a personal policy schedule.</p>',
        unsafe_allow_html=True,
    )
    st.stop()

pages = st.session_state["policy_pages"]
retriever = st.session_state["policy_retriever"]
passages = st.session_state["policy_passages"]
doc_key = st.session_state["doc_hash"]
words = sum(len(p["text"].split()) for p in pages)

st.markdown('<div class="brand"><span class="mark"></span>PolicyLens AI</div>', unsafe_allow_html=True)
st.markdown(f'<p class="docname">{html.escape(st.session_state.get("policy_name", "Uploaded policy"))}</p>', unsafe_allow_html=True)
st.markdown(
    f'<div class="facts"><div><b>{len(pages)}</b>pages with text</div>'
    f'<div><b>{len(passages)}</b>passages searched</div><div><b>{words:,}</b>words read</div></div>',
    unsafe_allow_html=True,
)

VIEWS = ["Summary", "Ask a question", "Clause explorer", "Page text"]
view = st.radio("Section", VIEWS, horizontal=True, label_visibility="collapsed", key="view")
st.write("")

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
if view == "Summary":
    st.subheader("Executive Policy Summary")
    st.markdown(
        '<p class="meta">A concise overview of coverage, rules, and claim procedures in plain English.</p>',
        unsafe_allow_html=True,
    )

    summary_data = summarize_policy(pages)

    # Helper to pick only the top 2 brief highlights per category
    def get_highlights(key, max_items=2):
        items = [i["text"].strip().rstrip(".") for i in summary_data.get(key, []) if i.get("text")]
        return items[:max_items]

    coverage = get_highlights("Coverage")
    exclusions = get_highlights("Exclusions")
    waiting = get_highlights("Waiting periods")
    claims = get_highlights("Claims")
    limits = get_highlights("Limits")

    sentences = []

    if coverage:
        cov_text = esc("; ".join(coverage))
        sentences.append(
            f'This policy covers <span class="hl" style="background:#CFEFD8;border-color:#2E7D4F">{cov_text}</span>.'
        )

    if limits:
        lim_text = esc("; ".join(limits))
        sentences.append(
            f'Key limits include <span class="hl" style="background:#FFE3B5;border-color:#B26A00">{lim_text}</span>.'
        )

    if waiting:
        wait_text = esc("; ".join(waiting))
        sentences.append(
            f'Waiting periods apply to <span class="hl" style="background:#FFF0A3;border-color:#B8960C">{wait_text}</span>.'
        )

    if exclusions:
        excl_text = esc("; ".join(exclusions))
        sentences.append(
            f'Exclusions include <span class="hl" style="background:#FFD6D2;border-color:#B3392F">{excl_text}</span>.'
        )

    if claims:
        claim_text = esc("; ".join(claims))
        sentences.append(
            f'Claims require <span class="hl" style="background:#CFE5FF;border-color:#2563A8">{claim_text}</span>.'
        )

    if sentences:
        compact_summary = " ".join(sentences)
        st.markdown(
            f'<div class="doc"><div class="serif" style="font-size:1.1rem; line-height:1.75;">'
            f'{compact_summary}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No clear policy details could be summarized.")

    st.subheader("How often key topics come up")
    low = "\n".join(p["text"] for p in pages).lower()
    terms = {
        "Coverage": ["covered", "coverage", "benefit"],
        "Exclusions": ["exclusion", "excluded", "not covered"],
        "Waiting periods": ["waiting period", "pre-existing disease"],
        "Claims": ["claim", "reimbursement", "intimation"],
        "Limits": ["sub-limit", "deductible", "co-payment", "sum insured"],
    }
    st.markdown(
        '<div class="facts">'
        + "".join(
            f"<div><b>{sum(low.count(w) for w in ws)}</b>{name}</div>"
            for name, ws in terms.items()
        )
        + "</div>"
        '<p class="meta">These counts only show where to look. They are not findings about what is covered.</p>',
        unsafe_allow_html=True,
    )
# --------------------------------------------------------------------------------------
elif view == "Ask a question":
    st.subheader("Ask about this policy")

    def pick(q: str):
        st.session_state["policy_question"] = q
        st.session_state["asked"] = q

    examples = [
        "What are the waiting periods?",
        "What is not covered?",
        "How do I make a claim?",
        "What is the grace period for renewal?",
        "Is there a co-payment or deductible?",
        "What documents do I need to claim?",
    ]
    st.markdown('<p class="meta">Pick a common question or type your own.</p>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(examples):
        cols[i % 2].button(q, key=f"ex{i}", on_click=pick, args=(q,), use_container_width=True)

    with st.form("ask_form", border=False):
        question = st.text_input("Your question", key="policy_question",
                                 placeholder="For example: What does the policy say about pre-existing diseases?")
        sent = st.form_submit_button("Find answer", type="primary")
    if sent and question.strip():
        st.session_state["asked"] = question.strip()

    asked = st.session_state.get("asked", "")
    if asked:
        with st.spinner("Searching the policy..."):
            results = retriever.search(asked, top_k=top_k)
            response = answer_question(asked, results)
        st.markdown(f"#### {html.escape(asked)}")
        st.markdown(f'<div class="answer">{md_lite(response["answer"])}</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="meta">How this was found: {html.escape(str(response["method"]))}</p>', unsafe_allow_html=True)
        st.markdown("#### Where it comes from")
        if response["evidence"]:
            for ev in response["evidence"]:
                st.markdown(
                    f'<div class="quote"><div class="serif">{esc(ev["text"])}</div>'
                    f'<div class="src">Page {ev["page"]} · match strength {ev["score"]:.2f}</div></div>',
                    unsafe_allow_html=True)
        else:
            st.warning("No matching passages were found. Try different words.")
        st.markdown('<p class="meta">A close match is not proof of coverage. Check the full clause, exceptions and your policy schedule.</p>',
                    unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
elif view == "Clause explorer":
    st.subheader("Clause explorer")
    st.markdown(
        '<p class="meta">Every paragraph is tagged with what it is about. The tags come from a trained model, '
        "so a few will be wrong. Use them to jump around, then read the text itself.</p>",
        unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def classify_blocks(_key: str, blocks: tuple):
        return [(pg, text, *predict_clause(text)) for pg, text in blocks]

    blocks = tuple((p["page"], b) for p in pages for b in paragraphs(p["text"]))
    with st.spinner("Tagging paragraphs. This takes a moment the first time..."):
        rows = classify_blocks(doc_key, blocks)

    counts = {k: 0 for k in ORDER}
    for _, _, label, _, _ in rows:
        counts[label] = counts.get(label, 0) + 1
    total = max(sum(counts.values()), 1)
    st.markdown(
        '<div class="bar" role="img" aria-label="Share of paragraphs by tag">' + "".join(
            f'<span style="width:{100*n/total:.2f}%;background:{clause_info(k)[1]}"></span>'
            for k, n in counts.items() if n) + "</div>"
        '<div class="legend">' + "".join(f"{tag(k)}<span class='meta'>{n}</span>" for k, n in counts.items() if n) + "</div>",
        unsafe_allow_html=True)

    names = {clause_info(k)[2]: k for k in counts if counts[k]}
    chosen = st.multiselect("Show only", list(names), placeholder="All types")
    query = st.text_input("Search within paragraphs", placeholder="For example: waiting period")
    wanted = {names[n] for n in chosen}
    shown = [r for r in rows
             if (not wanted or r[2] in wanted) and (not query or query.lower() in r[1].lower())]

    limit = st.session_state.setdefault("clause_limit", 25)
    if not shown:
        st.info("Nothing matches. Clear a filter or try a shorter search.")
    for page, text, label, conf, method in shown[:limit]:
        st.markdown(
            f'<div class="clause"><div class="head">{tag(label)}<span class="meta">Page {page}</span>'
            f'<span class="meta">{confidence_word(conf, method)}</span></div>'
            f'<div class="serif">{esc(text)}</div></div>', unsafe_allow_html=True)
    if len(shown) > limit:
        st.markdown(f'<p class="meta">Showing {limit} of {len(shown)} paragraphs.</p>', unsafe_allow_html=True)
        if st.button("Show 25 more"):
            st.session_state["clause_limit"] = limit + 25
            st.rerun()
    if rows and "keyword" in rows[0][4]:
        st.warning("The trained model was not found, so tags are rough keyword guesses. Put the model in models/clause-classifier for better results.")

# --------------------------------------------------------------------------------------
else:
    st.subheader("Read the original text")
    selected = st.selectbox("Page", [p["page"] for p in pages])
    page = next(p for p in pages if p["page"] == selected)
    st.markdown(f'<div class="quote"><div class="serif">{esc(page["text"])}</div></div>', unsafe_allow_html=True)
