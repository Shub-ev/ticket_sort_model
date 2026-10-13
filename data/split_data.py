from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_incidents.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

RANDOM_STATE = 42


def prepare_dataset(df):
    required_columns = {"text", "Assignment Group"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df[["text", "Assignment Group"]].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["Assignment Group"] = (
        df["Assignment Group"].fillna("").astype(str).str.strip()
    )
    df = df[df["text"].ne("") & df["Assignment Group"].ne("")].copy()

    conflicting = (
        df.groupby("text")["Assignment Group"].nunique().loc[lambda counts: counts > 1]
    )
    if not conflicting.empty:
        raise ValueError(
            f"{len(conflicting)} identical ticket texts have conflicting labels. "
            "Resolve these labels before splitting."
        )

    # Keep one copy of each exact ticket so duplicates cannot leak across splits
    # or overwhelm training through repeated examples.
    return df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)


def create_splits(df):
    class_counts = df["Assignment Group"].value_counts()
    rare_classes = class_counts[class_counts < 3]
    if not rare_classes.empty:
        raise ValueError(
            "Each assignment group needs at least 3 distinct tickets for "
            "train/validation/test splits:\n"
            f"{rare_classes.to_string()}"
        )

    train_val, test = train_test_split(
        df,
        test_size=0.10,
        random_state=RANDOM_STATE,
        stratify=df["Assignment Group"],
    )
    train, validation = train_test_split(
        train_val,
        test_size=1 / 9,
        random_state=RANDOM_STATE,
        stratify=train_val["Assignment Group"],
    )

    splits = {"train": train, "validation": validation, "test": test}
    expected_classes = set(class_counts.index)
    for name, split in splits.items():
        missing_classes = expected_classes - set(split["Assignment Group"])
        if missing_classes:
            raise ValueError(
                f"The {name} split is missing assignment groups: "
                f"{sorted(missing_classes)}"
            )

    return splits


def split_data():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found: {INPUT_PATH}")

    df = prepare_dataset(pd.read_csv(INPUT_PATH))
    splits = create_splits(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    print(f"Distinct usable tickets: {len(df):,}")
    for name, split in splits.items():
        print(f"{name.capitalize()} tickets: {len(split):,}")
    print("\nAssignment groups (distinct tickets):")
    print(df["Assignment Group"].value_counts().to_string())
    print(f"\nSaved leakage-free splits to: {OUTPUT_DIR}")


if __name__ == "__main__":
    split_data()
