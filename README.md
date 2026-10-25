# PolicyLens AI

**NLP + Machine Learning + Deep Learning for explaining Indian health
insurance policy documents.**

PolicyLens AI extracts policy text, classifies important clauses,
retrieves evidence, and answers questions with page-level citations. It
is an educational prototype, not insurance, medical, or legal advice.

## Features

-   Upload a health insurance policy PDF in the Streamlit UI.
-   Extract text with page numbers using PyMuPDF.
-   Classify clauses into categories such as coverage, exclusions,
    waiting periods, claims, limits, premiums, renewal, and definitions.
-   Train a scikit-learn baseline classifier.
-   Optional transformer fine-tuning with DistilBERT.
-   Retrieve relevant passages using TF-IDF with query/topic expansion;
    optional Sentence Transformers semantic retrieval.
-   Optional extractive QA model (`deepset/roberta-base-squad2`) to
    answer from retrieved passages.
-   Display page-numbered evidence and simple-English explanations;
    users should verify that each passage actually answers their
    question.
-   A source inventory CSV with 45 candidate policy/document records to
    collect from official insurer catalogs.

## Important dataset note

The repository does **not** bundle 45 copyrighted policy PDFs.
`data/policy_source_inventory.csv` is a curated acquisition checklist
with official source pages and candidate product names. Open the
official source page, download the current policy wording and/or
Customer Information Sheet (CIS), and record the exact
product/version/date. Availability and document URLs may change. Check
each insurer's terms and applicable copyright/licence before
redistributing documents. Do not commit personal policy schedules, claim
records, or other sensitive information.

The initial annotation file contains only small **illustrative
examples**, not ground truth from real policies. Replace/extend it with
clauses extracted from documents and manually verified by at least one
reviewer. Keep the policy ID, page number, exact source text, and
annotation provenance.

## Quick start

### Requirements

-   Python 3.10 or 3.11 is recommended.
-   Git
-   Internet access on the first run if optional Hugging Face models
    need to download.

### Windows PowerShell

``` powershell
git clone https://github.com/YOUR_USERNAME/PolicyLens-AI.git
cd PolicyLens-AI

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

Replace `YOUR_USERNAME` with your GitHub username. If PowerShell blocks
virtual-environment activation, you can run the project's Python
directly or follow Microsoft's PowerShell execution-policy guidance.

### Optional machine-learning dependencies

``` powershell
pip install -r requirements-ml.txt
```

Optional transformer features may download model weights the first time
they are used. This can take time and requires internet access. The app
may use its TF-IDF and rule-based fallbacks when optional model
components are unavailable.

## Add policy documents

1.  Use the source inventory: `data/policy_source_inventory.csv`.
2.  Download documents from the insurer's official catalog pages.
3.  Put local PDFs in `data/raw/` (the folder is git-ignored).
4.  Use the app's PDF uploader to try a document, or run the extraction
    utility:

``` bash
python -m src.extract_pdf data/raw/your_policy.pdf --output data/processed/your_policy.json
```

5.  Review extracted text and page numbers before using it for
    annotation.
6.  Add manually reviewed clauses to `data/annotations/clauses.csv`.

## Using the website

1.  Start the app with `streamlit run app/streamlit_app.py`.
2.  Open the local URL shown in the terminal (usually
    `http://localhost:8501`).
3.  Upload a selectable-text health insurance policy PDF in the upload
    area.
4.  Review the extracted text and page numbers. Ask questions in **Ask
    the policy** and check the cited evidence before relying on the
    answer.
5.  Use **Clause explorer** to inspect clause categories and **Extracted
    text** to verify the source wording.

### Example questions to ask

Try these after uploading a policy that contains the relevant sections:

**Waiting periods** - What are the waiting periods under this policy? -
What does the policy say about pre-existing diseases? - Are there
exceptions to the waiting period?

**Claims** - What documents are required to make a claim? - How soon
must claim documents be submitted? - How does reimbursement work under
this policy?

**Coverage and exclusions** - What treatments or conditions are
excluded? - What hospital expenses are covered? - Are there any limits
or sub-limits? - Does the policy mention a co-payment or deductible?

**Renewal and premium** - What is the renewal process? - What is the
grace period for renewal? - What does the policy say about premium
payments?

The answer depends on the specific uploaded document and how well its
text was extracted. If a question concerns a specific benefit, ask about
that benefit by name. Check the cited page and full clause, because a
policy may contain different rules for different sections or benefits.

## Clause annotation schema

`data/annotations/clauses.csv` uses these columns:

-   `text`: exact clause text
-   `label`: one of the supported clause labels
-   `policy_id`: internal document identifier, not a policyholder ID
-   `page`: PDF page number (1-indexed)
-   `reviewed`: `yes` only after human review

Supported labels: `coverage`, `exclusion`, `waiting_period`,
`claim_procedure`, `limit`, `premium`, `renewal_cancellation`,
`definition`, `other`.

Do not use the illustrative seed rows for a final evaluation. Add more
reviewed examples. Split train/test by **policy document**, not by
randomly splitting near-duplicate clauses.

## Train the ML classifier

``` bash
python -m src.train_classifier
```

This trains a TF-IDF + Logistic Regression baseline and saves it to
`models/clause_classifier.joblib`. The Streamlit app loads it
automatically if available.

## Optional: fine-tune a transformer

For an experimental run, install `requirements-ml.txt` and run the
command below. For a meaningful accuracy improvement, first collect
substantially more manually reviewed examples across multiple policy
documents:

``` bash
python -m src.train_transformer --epochs 3
```

This uses DistilBERT sequence classification. Results are not
guaranteed; compare macro-F1 against the baseline and use a
document-level holdout. Do not claim the transformer is better unless
evaluation supports that claim.

## Run tests

``` bash
pytest -q
```

## Project structure

``` text
PolicyLens-AI/
├── app/streamlit_app.py
├── data/
│   ├── policy_source_inventory.csv
│   ├── raw/                  # local PDFs; ignored by Git
│   ├── processed/            # generated text; ignored by Git
│   └── annotations/clauses.csv
├── models/                   # trained models; ignored by Git
├── reports/
├── src/
│   ├── extract_pdf.py
│   ├── clause_classifier.py
│   ├── retriever.py
│   ├── qa.py
│   ├── train_classifier.py
│   └── train_transformer.py
├── tests/
├── requirements.txt
├── requirements-ml.txt
└── README.md
```

## Current prototype status and model evaluation

The project currently demonstrates PDF text extraction, evidence
retrieval, page references, simple-English explanations, and clause
classification. The DistilBERT classifier was trained on a small set of
150 reviewed examples across nine labels. In the latest recorded
document-level evaluation, it achieved approximately **28.95% accuracy**
and **0.3048 macro-F1** on 38 evaluation examples from 11 documents.

These scores are low and indicate that classification is still
experimental. Some labels have very few examples, and the `other`
category is especially limited across source documents. The app also
uses targeted rules for a few common patterns; these rules can improve
those specific cases but do **not** increase or replace the measured
overall model accuracy.

For an academic report, present the scores above as the current measured
result, not as a high-accuracy production classifier. Improve the
dataset with more manually verified examples from diverse policy
documents, keep near-duplicate clauses from the same policy in only one
split, and compare the transformer against the TF-IDF baseline using the
same held-out documents.

## Evaluation plan

-   Classification: macro-F1, per-class precision/recall, confusion
    matrix.
-   Extraction: field-level precision/recall/F1 and exact match for
    durations/amounts.
-   Retrieval: Recall@5 and MRR on manually authored questions with
    known source pages.
-   QA: answer correctness, evidence support, citation accuracy, and
    correct abstention when the document does not answer the question.
-   OCR: manually inspect a sample of pages, especially tables,
    exclusions, and footnotes.

## Safety, privacy, and limitations

-   This app explains text from the uploaded policy; it does not
    determine whether an insurer will approve a claim.
-   Coverage may depend on endorsements, schedule details, waiting
    periods, exclusions, sub-limits, and facts not included in the
    wording.
-   Never upload or commit sensitive personal documents without consent
    and appropriate security controls.
-   Avoid sharing user-uploaded PDFs with third-party APIs unless users
    have explicitly consented.
-   Always verify important decisions against the full policy and the
    insurer/qualified professional.
-   Generated answers can be wrong. Citations and evidence must be
    checked.

## Data sources

Start with the official catalogs listed in
`data/policy_source_inventory.csv`, including: - IRDAI Health Insurance
Products portal: https://irdai.gov.in/health-insurance-products - HDFC
ERGO health policy wordings:
https://www.hdfcergo.com/download/policy-wordings/health - HDFC ERGO
health Customer Information Sheets:
https://www.hdfcergo.com/download/cis/health - Star Health downloads:
https://www.starhealth.in/downloads/ - SBI General downloads:
https://www.sbigeneral.in/downloads

Document availability, versions, and reuse rights vary. Prefer the exact
wording corresponding to the product/version under study; CIS documents
are useful for comparison and validation but are not a substitute for
full policy wording.

## License

Code is provided for educational use. This code license does not grant
rights to third-party policy documents. Add a LICENSE file before public
release and choose a licence appropriate for your project.
