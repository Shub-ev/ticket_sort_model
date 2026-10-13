import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_ticket_classifier"
TEST_SET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ticket_test_set.json"
)
SPLIT_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv"
BATCH_SIZE = 16


def create_test_set_json():
    if TEST_SET_PATH.is_file():
        return
    if not SPLIT_TEST_PATH.is_file():
        raise FileNotFoundError(
            f"Test split not found: {SPLIT_TEST_PATH}. "
            "Run data/split_data.py first."
        )

    test_df = pd.read_csv(str(SPLIT_TEST_PATH))
    required_columns = {"text", "Assignment Group"}
    missing_columns = required_columns - set(test_df.columns)
    if missing_columns:
        raise ValueError(
            f"{SPLIT_TEST_PATH} is missing columns: {sorted(missing_columns)}"
        )

    tickets = [
        {
            "text": str(row["text"]),
            "expected_assignment_group": str(row["Assignment Group"]),
        }
        for _, row in test_df.iterrows()
    ]
    TEST_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TEST_SET_PATH.open("w", encoding="utf-8") as file:
        json.dump(tickets, file, indent=2, ensure_ascii=False)
    print(f"Created JSON test set from held-out split: {TEST_SET_PATH}")


def load_test_set():
    if not TEST_SET_PATH.is_file():
        raise FileNotFoundError(
            f"Test set not found: {TEST_SET_PATH}."
        )

    with TEST_SET_PATH.open(encoding="utf-8") as file:
        tickets = json.load(file)

    if not isinstance(tickets, list) or not tickets:
        raise ValueError("The JSON test set must be a non-empty list.")

    required_fields = {"text", "expected_assignment_group"}
    for index, ticket in enumerate(tickets):
        if (
            not isinstance(ticket, dict)
            or not required_fields.issubset(ticket)
        ):
            raise ValueError(
                f"Ticket {index} must have text and "
                "expected_assignment_group fields."
            )
        if not isinstance(ticket["text"], str) or not ticket["text"].strip():
            raise ValueError(f"Ticket {index} has empty or invalid text.")
        if not isinstance(ticket["expected_assignment_group"], str):
            raise ValueError(
                f"Ticket {index} has an invalid expected assignment group."
            )

    return tickets


def evaluate_model():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_DIR}. Run training/train.py first."
        )

    create_test_set_json()
    tickets = load_test_set()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    max_length = 384
    metadata_path = MODEL_DIR / "training_metadata.json"
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as file:
            max_length = json.load(file).get("max_length", max_length)

    label_names = [
        model.config.id2label[index]
        for index in range(model.config.num_labels)
    ]
    actual = [ticket["expected_assignment_group"] for ticket in tickets]
    unknown_labels = set(actual) - set(label_names)
    if unknown_labels:
        raise ValueError(
            "Test set contains labels unknown to this model: "
            f"{sorted(unknown_labels)}"
        )

    predictions = []
    confidences = []
    with torch.inference_mode():
        for start in range(0, len(tickets), BATCH_SIZE):
            batch = tickets[start:start + BATCH_SIZE]
            inputs = tokenizer(
                [ticket["text"] for ticket in batch],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)
            probabilities = model(**inputs).logits.softmax(dim=-1)
            batch_confidences, predicted_ids = probabilities.max(dim=-1)
            predicted_ids = predicted_ids.cpu().tolist()
            predictions.extend(
                model.config.id2label[index] for index in predicted_ids
            )
            confidences.extend(batch_confidences.cpu().tolist())

    print(f"Device: {device.type}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Test tickets: {len(tickets)}")
    print(f"Accuracy: {accuracy_score(actual, predictions):.4f}")
    print(
        "Macro F1: "
        f"{f1_score(actual, predictions, average='macro', zero_division=0):.4f}"
    )
    print("\nPer-assignment-group results:")
    print(
        classification_report(
            actual,
            predictions,
            labels=label_names,
            target_names=label_names,
            zero_division=0,
        )
    )

    print("\nTicket classifications:")
    for index, (ticket, prediction, confidence) in enumerate(
        zip(tickets, predictions, confidences),
        start=1,
    ):
        expected = ticket["expected_assignment_group"]
        result = "correct" if prediction == expected else "incorrect"
        print(
            f"{index}. Classified group: {prediction} | "
            f"Confidence: {confidence:.1%} | "
            f"Expected group: {expected} | {result}"
        )


if __name__ == "__main__":
    evaluate_model()
