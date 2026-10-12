
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_incidents.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

RANDOM_STATE = 42


def split_data():
    df = pd.read_csv(INPUT_PATH)

    df = df.dropna(subset=["text", "Assignment Group"]).copy()

    # Remove empty text and labels.
    df["text"] = df["text"].astype(str).str.strip()
    df["Assignment Group"] = (
        df["Assignment Group"].astype(str).str.strip()
    )

    df = df[
        df["text"].ne("")
        & df["Assignment Group"].ne("")
    ].copy()

    # Each class needs enough examples for a stratified split.
    group_counts = df["Assignment Group"].value_counts()

    rare_groups = group_counts[group_counts < 3]
    if not rare_groups.empty:
        raise ValueError(
            "Some assignment groups have fewer than 3 tickets. "
            "Review these groups before splitting:\n"
            f"{rare_groups.to_string()}"
        )

    # Reserve 10% for the final test set.
    train_val, test = train_test_split(
        df,
        test_size=0.10,
        random_state=RANDOM_STATE,
        stratify=df["Assignment Group"],
    )

    # Split the remaining 90% into 80% train and 10% validation.
    train, validation = train_test_split(
        train_val,
        test_size=1 / 9,
        random_state=RANDOM_STATE,
        stratify=train_val["Assignment Group"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print(f"Total tickets:     {len(df):,}")
    print(f"Training tickets:  {len(train):,}")
    print(f"Validation tickets:{len(validation):,}")
    print(f"Test tickets:      {len(test):,}")

    print("\nTraining distribution:")
    print(train["Assignment Group"].value_counts().to_string())

    print("\nSaved datasets to:", OUTPUT_DIR)

split_data()