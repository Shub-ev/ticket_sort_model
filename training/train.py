import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_ticket_classifier"

MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 256


class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index], dtype=torch.long
        )
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
    }


def train():
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    val_df = pd.read_csv(SPLIT_DIR / "validation.csv")

    # Fit label encoding using training labels only.
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["Assignment Group"].astype(str))

    for name, df in [("validation", val_df)]:
        unknown = (
            set(df["Assignment Group"].astype(str))
            - set(label_encoder.classes_)
        )
        if unknown:
            raise ValueError(
                f"Unknown assignment groups in {name}: {unknown}"
            )

    train_labels = label_encoder.transform(
        train_df["Assignment Group"].astype(str)
    )
    val_labels = label_encoder.transform(
        val_df["Assignment Group"].astype(str)
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TicketDataset(
        train_df["text"].astype(str).tolist(),
        train_labels,
        tokenizer,
    )
    val_dataset = TicketDataset(
        val_df["text"].astype(str).tolist(),
        val_labels,
        tokenizer,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
    )

    # Make prediction IDs correspond to actual assignment group names.
    model.config.id2label = {
        i: label for i, label in enumerate(label_encoder.classes_)
    }
    model.config.label2id = {
        label: i for i, label in enumerate(label_encoder.classes_)
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "training_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting DistilBERT training...")
    print("Training tickets:", len(train_dataset))
    print("Validation tickets:", len(val_dataset))
    print("Assignment groups:", len(label_encoder.classes_))
    print("GPU available:", torch.cuda.is_available())

    trainer.train()

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    label_mapping = {
        str(i): label
        for i, label in enumerate(label_encoder.classes_)
    }
    with open(MODEL_DIR / "label_mapping.json", "w", encoding="utf-8") as file:
        json.dump(label_mapping, file, indent=2, ensure_ascii=False)

    print("\nTraining completed.")
    print("Saved model to:", MODEL_DIR)
    print("Validation metrics:", trainer.evaluate())


train()