
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "annotations" / "clauses.csv"
MODEL_DIR = ROOT / "models" / "distilbert-clause-classifier"


def main():
    if not DATA.exists():
        raise SystemExit(f"Dataset not found: {DATA}")

    if not (MODEL_DIR / "config.json").exists():
        raise SystemExit(f"Trained model not found: {MODEL_DIR}")

    df = pd.read_csv(DATA)
    required = {"text", "label", "reviewed", "source_file"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing CSV columns: {sorted(missing)}")

    df = df[
        df["reviewed"].astype(str).str.strip().str.lower().eq("yes")
        & df["text"].notna()
        & df["label"].notna()
    ].copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"].str.len().ge(20) & df["label"].ne("")]

    # Use the exact label mapping saved with the trained model.
    labels_file = MODEL_DIR / "labels.json"
    if labels_file.exists():
        label2id = json.loads(labels_file.read_text(encoding="utf-8"))
        label2id = {str(k): int(v) for k, v in label2id.items()}
    else:
        config = json.loads(
            (MODEL_DIR / "config.json").read_text(encoding="utf-8")
        )
        label2id = {
            str(label): int(idx)
            for label, idx in config["label2id"].items()
        }

    df = df[df["label"].isin(label2id)].copy()
    df["label_id"] = df["label"].map(label2id).astype(int)

    groups = df["source_file"].fillna("unknown").astype(str)
    if groups.nunique() < 3:
        raise SystemExit("Need at least 3 documents for evaluation.")

    # Reproduce the training script's document-level split.
    labels = set(df["label"])
    train_df = eval_df = None

    splitter = GroupShuffleSplit(
        n_splits=50, test_size=0.2, random_state=42
    )

    for train_idx, eval_idx in splitter.split(df, groups=groups):
        candidate_train = df.iloc[train_idx].copy()
        candidate_eval = df.iloc[eval_idx].copy()

        if (
            set(candidate_train["label"]) == labels
            and set(candidate_eval["label"]) == labels
        ):
            train_df = candidate_train
            eval_df = candidate_eval
            break

    if eval_df is None:
        raise SystemExit(
            "Could not reproduce a document-level split with every label. "
            "Use the original training output if available."
        )

    print(f"Evaluation examples: {len(eval_df)}")
    print(f"Evaluation documents: {eval_df['source_file'].nunique()}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )

    dataset = Dataset.from_pandas(
        eval_df[["text", "label_id"]]
        .rename(columns={"label_id": "labels"}),
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    dataset = dataset.map(
        tokenize, batched=True, remove_columns=["text"]
    )

    args = TrainingArguments(
        output_dir=str(ROOT / "models" / "evaluation-temp"),
        per_device_eval_batch_size=4,
        report_to="none",
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=dataset,
        processing_class=tokenizer,
    )

    output = trainer.predict(dataset)
    predictions = np.argmax(output.predictions, axis=-1)
    actual = output.label_ids

    id2label = {
        int(idx): label for label, idx in label2id.items()
    }
    ordered_ids = sorted(id2label)
    target_names = [id2label[i] for i in ordered_ids]

    print("\nAccuracy:", round(accuracy_score(actual, predictions), 4))
    print(
        "Macro F1:",
        round(f1_score(actual, predictions, average="macro", zero_division=0), 4),
    )

    print("\nClassification report:")
    print(
        classification_report(
            actual,
            predictions,
            labels=ordered_ids,
            target_names=target_names,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(actual, predictions, labels=ordered_ids))
    print("\nLabel order:", target_names)


if __name__ == "__main__":
    main()
