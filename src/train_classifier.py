"""Train a TF-IDF + Logistic Regression baseline from reviewed clauses."""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "annotations" / "clauses.csv"
OUT = ROOT / "models" / "clause_classifier.joblib"

def main():
    df = pd.read_csv(DATA)
    df = df[(df["reviewed"].astype(str).str.lower() == "yes") & df["text"].notna() & df["label"].notna()]
    if df["label"].nunique() < 2 or len(df) < 10:
        raise SystemExit("Add at least 10 human-reviewed clauses across 2+ labels to data/annotations/clauses.csv before training.")
    # This small baseline uses stratified random split. For final research results, split by policy_id to avoid leakage.
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.25, random_state=42, stratify=stratify)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test), zero_division=0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT)
    print(f"Saved {OUT}")
if __name__ == "__main__":
    main()
