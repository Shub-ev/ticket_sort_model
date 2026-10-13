
import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "distilbert_ticket_classifier"

MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 256


def prepare_training_data():
    # Load the three existing data splits.
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    validation_df = pd.read_csv(SPLIT_DIR / "validation.csv")
    test_df = pd.read_csv(SPLIT_DIR / "test.csv")

    # Fit the label encoder on training labels only.
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["Assignment Group"].astype(str))

    # Every split must contain only labels known during training.
    known_labels = set(label_encoder.classes_)

    for split_name, df in [
        ("training", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        labels = set(df["Assignment Group"].astype(str))
        unknown_labels = labels - known_labels

        if unknown_labels:
            raise ValueError(
                f"Unknown labels in {split_name}: {unknown_labels}"
            )

    # Encode the target variable into integer class IDs.
    train_df["labels"] = label_encoder.transform(
        train_df["Assignment Group"].astype(str)
    )
    validation_df["labels"] = label_encoder.transform(
        validation_df["Assignment Group"].astype(str)
    )
    test_df["labels"] = label_encoder.transform(
        test_df["Assignment Group"].astype(str)
    )

    # Load the pretrained DistilBERT tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    # Tokenize each split.
    train_tokens = tokenize_batch(train_df["text"].tolist())
    validation_tokens = tokenize_batch(validation_df["text"].tolist())
    test_tokens = tokenize_batch(test_df["text"].tolist())

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the label mapping and tokenizer for later training/inference.
    tokenizer.save_pretrained(ARTIFACT_DIR)

    label_mapping = {
        str(index): label
        for index, label in enumerate(label_encoder.classes_)
    }

    with open(ARTIFACT_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    # Report tokenization results.
    print("Tokenization completed successfully.")
    print("Assignment groups:", len(label_mapping))
    print("Training tickets:", len(train_df))
    print("Validation tickets:", len(validation_df))
    print("Test tickets:", len(test_df))

    print("\nLabel mapping:")
    print(json.dumps(label_mapping, indent=2, ensure_ascii=False))

    print("\nFirst ticket token IDs:")
    print(train_tokens["input_ids"][0][:20])

    print("\nFirst ticket label:", train_df["labels"].iloc[0])
    print("Tokenized training examples:", len(train_tokens["input_ids"]))



if __name__ == "__main__":
    prepare_training_data()