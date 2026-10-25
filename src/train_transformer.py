"""DistilBERT clause classification with document-level validation and resumable checkpoints.

Run from project root: python src/train_transformer.py --epochs 3
Requires human-reviewed rows in data/annotations/clauses.csv.
"""
from pathlib import Path
import argparse
import inspect
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
DATA = ROOT / 'data' / 'annotations' / 'clauses.csv'
OUT = ROOT / 'models' / 'distilbert-clause-classifier'
CHECKPOINTS = ROOT / 'models' / 'transformer-training'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--smoke-test', action='store_true', help='Run two training steps only; no model export')
    parser.add_argument('--resume', action='store_true', help='Resume latest existing checkpoint')
    args = parser.parse_args()

    df = pd.read_csv(DATA)
    required = {'text', 'label', 'reviewed', 'source_file'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'Missing CSV columns: {sorted(missing)}')
    df = df[df['reviewed'].astype(str).str.strip().str.lower().eq('yes') & df['text'].notna() & df['label'].notna()].copy()
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df = df[df['text'].str.len().ge(20) & df['label'].ne('')]
    counts = df['label'].value_counts()
    print('Reviewed label counts:\n', counts.to_string())
    if len(df) < 20 or len(counts) < 2:
        raise SystemExit('Need >=20 human-reviewed clauses and >=2 categories.')

    if not args.smoke_test and (counts.min() < 5 or len(df) < 100):
        raise SystemExit('Not enough balanced reviewed examples for meaningful training/evaluation. Aim for >=100 reviewed rows and >=5 per class first. Use --smoke-test to check GPU training only.')

    labels = sorted(counts.index)
    label2id = {label: i for i, label in enumerate(labels)}
    df['label_id'] = df['label'].map(label2id)

    if args.smoke_test:
        train_df = df.copy()
        eval_df = None
    else:
        groups = df['source_file'].fillna('unknown').astype(str)
        if groups.nunique() < 3:
            raise SystemExit('Need clauses from at least 3 separate documents for document-level evaluation.')
        splitter = GroupShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
        train_df = None
        eval_df = None

        for train_idx, eval_idx in splitter.split(df, groups=groups):
            candidate_train = df.iloc[train_idx].copy()
            candidate_eval = df.iloc[eval_idx].copy()

            if (
                set(candidate_train["label"]) == set(labels)
                and set(candidate_eval["label"]) == set(labels)
            ):
                train_df = candidate_train
                eval_df = candidate_eval
                break

        if train_df is None or eval_df is None:
            raise SystemExit(
                "Could not find a document-level split containing every label "
                "in both training and evaluation. More reviewed examples across "
                "documents are needed for a reliable evaluation."
            )

        print("Found a document-level split containing every label.")
        print(f"Train rows: {len(train_df)}; evaluation rows: {len(eval_df)}")
        print("Training labels:", sorted(train_df["label"].unique()))
        print("Evaluation labels:", sorted(eval_df["label"].unique()))  

        train_labels = set(train_df['label'])
        eval_labels = set(eval_df['label'])

        print("Training labels:", sorted(train_labels))
        print("Evaluation labels:", sorted(eval_labels))

        missing_train = set(labels) - train_labels
        missing_eval = set(labels) - eval_labels

        if missing_train or missing_eval:
            print("Warning: Some labels are missing from one split.")
            print("Missing from training:", sorted(missing_train))
            print("Missing from evaluation:", sorted(missing_eval))
            print("Continuing with this prototype split.")

    print(f'Train rows: {len(train_df)}; evaluation rows: {len(eval_df)}')

    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def make_dataset(frame):
        ds = Dataset.from_pandas(frame[['text', 'label_id']].rename(columns={'label_id': 'labels'}), preserve_index=False)
        return ds.map(lambda batch: tokenizer(batch['text'], truncation=True, max_length=256), batched=True, remove_columns=['text'])

    train_ds = make_dataset(train_df)
    eval_ds = make_dataset(eval_df) if eval_df is not None else None
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels),
        id2label={i: label for label, i in label2id.items()}, label2id=label2id,
    )

    def metrics(eval_pred):
        predictions = eval_pred.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        preds = np.argmax(predictions, axis=-1)
        return {
            'accuracy': accuracy_score(eval_pred.label_ids, preds),
            'macro_f1': f1_score(eval_pred.label_ids, preds, average='macro', zero_division=0),
        }

    kwargs = dict(
        output_dir=str(CHECKPOINTS),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_strategy='steps',
        save_steps=50 if not args.smoke_test else 1,
        save_total_limit=3,
        logging_steps=1 if args.smoke_test else 10,
        report_to='none',
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        max_steps=2 if args.smoke_test else -1,
    )
    if eval_ds is not None:
        # Transformers changed evaluation_strategy to eval_strategy in newer versions.
        key = 'eval_strategy' if 'eval_strategy' in inspect.signature(TrainingArguments).parameters else 'evaluation_strategy'
        kwargs[key] = 'steps'
        kwargs['eval_steps'] = 50
        kwargs['load_best_model_at_end'] = True
        kwargs['metric_for_best_model'] = 'macro_f1'
        kwargs['greater_is_better'] = True

    training_args = TrainingArguments(**kwargs)
    trainer_kwargs = dict(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds,
                          compute_metrics=metrics if eval_ds is not None else None)
    trainer_sig = inspect.signature(Trainer).parameters
    trainer_kwargs['processing_class' if 'processing_class' in trainer_sig else 'tokenizer'] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    print('Device:', 'CUDA / ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    if args.resume:
        checkpoints = sorted(CHECKPOINTS.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
        if not checkpoints:
            raise SystemExit('No checkpoint found to resume. Run without --resume first.')
        print('Resuming:', checkpoints[-1])
        trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
    else:
        trainer.train()

    if args.smoke_test:
        print('GPU smoke test finished; model not exported.')
        return
    
    
    metrics = trainer.evaluate()
    print("\nFinal evaluation metrics:", metrics)

    if eval_ds is not None:
        predictions_output = trainer.predict(eval_ds)
        predictions = predictions_output.predictions

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predicted_ids = np.argmax(predictions, axis=-1)
        true_ids = predictions_output.label_ids

        label_ids = list(range(len(labels)))
        target_names = [label for label, _ in sorted(label2id.items(), key=lambda item: item[1])]

        print("\nClassification report:")
        print(
            classification_report(
                true_ids,
                predicted_ids,
                labels=label_ids,
                target_names=target_names,
                zero_division=0,
            )
        )

        print("Confusion matrix (rows=true, columns=predicted):")
        print(confusion_matrix(true_ids, predicted_ids, labels=label_ids))

        OUT.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(OUT))
        tokenizer.save_pretrained(str(OUT))
        (OUT / 'labels.json').write_text(json.dumps(label2id, indent=2), encoding='utf-8')
        print('Saved model to', OUT)

if __name__ == '__main__':
    main()
