from pathlib import Path
import pandas as pd
import re
import html
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "incidents.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_incidents.csv"
)

REQUIRED_COLUMNS = [
    "Short description",
    "Description",
    "Assignment Group",
]

print(INPUT_PATH)



def clean_html(text: str) -> str:
    if not text:
        return ""

    # Decode HTML entities such as &nbsp; and &amp;
    # Converts HTML entities into their corresponding characters.
    text = html.unescape(text)

    # Remove HTML tags and extract readable text
    text = BeautifulSoup(text, "html.parser").get_text(
        separator=" ",
        strip=True,
    )

    # Normalize whitespace using Python's regular expression module (re)
    # .strip() removes white spaces from start and end
    text = re.sub(r"\s+", " ", text).strip()

    return text



def validate_dataset():
    # check if Input path exists (file and directory)
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"CSV file not found at {INPUT_PATH}")


    # read CSV file using pandas into pandas DataFrame
    # A DataFrame is a table-like data structure containing rows and columns.
    # utf-8-sig handles UTF-8 files that may contain byte-order-mark (BOM)
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")


    # df.columns get all column names
    # .astype(str) convert column names to string
    # .str.strip(str) removes white spaces from start and end of str
    df.columns = df.columns.astype(str).str.strip()


    # checks if all required columns are present
    # '-' preforms set difference, returning required names that are not present in the actual csv
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


    # Keep only required columns
    # The .copy() makes it safer to modify the selected DataFrame without unintentionally modifying a view of the original
    df = df[REQUIRED_COLUMNS].copy()


    # Clean both "Short description" and "Description" columns
    for column in ["Short description", "Description"]:
        df[column] = (df[column].fillna("").astype(str).map(clean_html))

    df["Assignment Group"] = (
        df["Assignment Group"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    print(f"Original tickets: {len(df):,}")
    print("Missing assignment groups:",
          df["Assignment Group"].eq("").sum())

    empty_text = (
            df["Short description"].eq("")
            & df["Description"].eq("")
    )
    print("Tickets with no description text:", empty_text.sum())

    # Remove records without usable text or a target label.
    df = df[
        df["Assignment Group"].ne("")
        & (
                df["Short description"].ne("")
                | df["Description"].ne("")
        )
        ].copy()

    # Combine the two input variables.
    df["text"] = (
            "[SHORT DESCRIPTION] "
            + df["Short description"]
            + " [DESCRIPTION] "
            + df["Description"]
    )

    print(f"\nUsable tickets: {len(df):,}")
    print(f"Unique assignment groups: {df['Assignment Group'].nunique()}")

    print("\nTickets per assignment group:")
    print(df["Assignment Group"].value_counts().to_string())

    print(
        "\nExact duplicate text-label pairs:",
        df.duplicated(subset=["text", "Assignment Group"]).sum(),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df[["text", "Assignment Group"]].to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"\nSaved cleaned dataset: {OUTPUT_PATH}")


validate_dataset()