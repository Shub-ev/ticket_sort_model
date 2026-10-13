import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert_ticket_classifier"
MODEL_NAME = "distilbert/distilbert-base-uncased"


class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
        )
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class ClassWeightedTrainer(Trainer):
    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError("Every assignment group must appear in training data.")

    # Square-root weighting helps minority groups without the instability of
    # fully inverse-frequency weighting for groups with very few examples.
    weights = np.sqrt(len(labels) / (num_classes * counts))
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
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


def load_splits():
    splits = {}
    for name in ("train", "validation", "test"):
        path = SPLIT_DIR / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name} split at {path}. Run data/split_data.py first."
            )

        df = pd.read_csv(path)
        required_columns = {"text", "Assignment Group"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing_columns)}"
            )
        df = df[["text", "Assignment Group"]].copy()
        df["text"] = df["text"].fillna("").astype(str).str.strip()
        df["Assignment Group"] = (
            df["Assignment Group"].fillna("").astype(str).str.strip()
        )
        df = df[
            df["text"].ne("") & df["Assignment Group"].ne("")
        ].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"The {name} split contains no usable tickets.")
        splits[name] = df

    for first, second in (("train", "validation"), ("train", "test"),
                          ("validation", "test")):
        overlap = set(splits[first]["text"]) & set(splits[second]["text"])
        if overlap:
            raise ValueError(
                f"Found {len(overlap)} ticket texts in both {first} and "
                f"{second}. Regenerate splits with data/split_data.py."
            )

    return splits


def train(args):
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable, so training cannot use the NVIDIA GPU. "
            "Install the CUDA-enabled PyTorch build using the setup command "
            "in README.md, then verify it with "
            "\"python -c \\\"import torch; print(torch.cuda.is_available())\\\"\". "
            "Use --allow-cpu only if you intentionally want CPU training."
        )

    use_cuda = torch.cuda.is_available() and not args.allow_cpu
    set_seed(args.seed)
    splits = load_splits()

    label_encoder = LabelEncoder()
    label_encoder.fit(splits["train"]["Assignment Group"])
    labels_by_split = {}
    for name, df in splits.items():
        labels = set(df["Assignment Group"])
        unknown = labels - set(label_encoder.classes_)
        missing = set(label_encoder.classes_) - labels
        if unknown or missing:
            raise ValueError(
                f"Invalid {name} labels; unknown={sorted(unknown)}, "
                f"missing={sorted(missing)}. Each split must include every "
                "training assignment group."
            )
        labels_by_split[name] = label_encoder.transform(
            df["Assignment Group"]
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    datasets = {
        name: TicketDataset(
            df["text"].tolist(),
            labels_by_split[name],
            tokenizer,
            args.max_length,
        )
        for name, df in splits.items()
    }

    label_names = label_encoder.classes_.tolist()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={label: index for index, label in enumerate(label_names)},
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_dir = PROJECT_ROOT / "models" / "training_runs" / run_id
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        use_cpu=args.allow_cpu,
        fp16=use_cuda and not use_bf16,
        bf16=use_bf16,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
    )

    class_weights = compute_class_weights(
        labels_by_split["train"],
        num_classes=len(label_names),
    )
    trainer = ClassWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if use_cuda else None,
        ),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.001,
            )
        ],
    )

    print(f"Base model: {args.model_name}")
    print(f"Training/validation/test examples: "
          f"{len(datasets['train']):,}/"
          f"{len(datasets['validation']):,}/"
          f"{len(datasets['test']):,}")
    print(f"Assignment groups: {len(label_names)}")
    if use_cuda:
        print(f"Device: CUDA ({torch.cuda.get_device_name()})")
        print(f"CUDA runtime: {torch.version.cuda}; mixed precision: "
              f"{'BF16' if use_bf16 else 'FP16'}")
    else:
        print("Device: CPU (explicitly requested)")
    print("Training with validation-based early stopping...")

    trainer.train()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    test_output = trainer.predict(
        datasets["test"],
        metric_key_prefix="test",
    )
    test_predictions = np.argmax(test_output.predictions, axis=-1)
    per_class_report = classification_report(
        labels_by_split["test"],
        test_predictions,
        labels=np.arange(len(label_names)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    test_metrics = {
        key: float(value)
        for key, value in test_output.metrics.items()
    }
    evaluation = {
        "test_metrics": test_metrics,
        "per_class": per_class_report,
        "test_examples": len(datasets["test"]),
        "selection_metric": "validation macro F1",
        "best_validation_macro_f1": float(trainer.state.best_metric),
    }
    with (MODEL_DIR / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2, ensure_ascii=False)

    with (MODEL_DIR / "training_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "base_model": args.model_name,
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
                "max_length": args.max_length,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": (
                    args.gradient_accumulation_steps
                ),
                "requested_epochs": args.epochs,
                "class_names": label_names,
                "class_weights": class_weights.tolist(),
                "class_counts": {
                    name: {
                        label: int(count)
                        for label, count in (
                            df["Assignment Group"].value_counts().items()
                        )
                    }
                    for name, df in splits.items()
                },
                "split_sizes": {
                    name: len(df) for name, df in splits.items()
                },
                "checkpoint_dir": str(checkpoint_dir),
                "best_checkpoint": trainer.state.best_model_checkpoint,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    with (MODEL_DIR / "label_mapping.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {str(index): label for index, label in enumerate(label_names)},
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved best validation checkpoint to: {MODEL_DIR}")
    print("Held-out test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("\nPer-assignment-group test results:")
    print(
        pd.DataFrame(per_class_report)
        .reindex(label_names)
        .dropna(how="all")
        .to_string()
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT to route tickets to assignment groups."
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training on CPU when CUDA is unavailable (slow).",
    )
    args = parser.parse_args()

    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
