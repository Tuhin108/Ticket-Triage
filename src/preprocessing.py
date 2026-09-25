"""
preprocessing.py  -  Task 1: Data Preparation

Functions to load the ticket CSV, inspect it, clean it (duplicates /
missing values) and clean the ticket description text. Can also be run
directly to print a full data-preparation report to the console.

Run:
    python src/preprocessing.py
"""

import re
import pandas as pd


def load_data(path="data/tickets.csv"):
    return pd.read_csv(path)


def clean_text(text):
    """Lowercase, strip punctuation/extra spaces from a ticket description.
    Used for the TF-IDF ('classic') model."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()   # collapse extra whitespace
    return text


def light_clean(text):
    """Whitespace-only cleanup that preserves case and punctuation. Sentence
    embedding models are trained on natural language, so they work better on
    text close to its original form than on the aggressively stripped
    version used for TF-IDF."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def prepare_dataset(df):
    """Full cleaning pipeline: report missing/duplicates, then fix them."""
    report = {}

    report["rows_before"] = len(df)
    report["missing_before"] = df.isnull().sum().to_dict()
    report["duplicates_before"] = int(df.duplicated().sum())

    # Drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Drop rows with no description or no category (can't train/label without them)
    df = df.dropna(subset=["ticket_description", "category"]).reset_index(drop=True)

    # Fill remaining missing categorical fields with "Unknown" rather than dropping rows
    for col in ["priority", "status"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Clean the text column into new columns used for modeling
    df["clean_description"] = df["ticket_description"].apply(clean_text)
    df["light_clean_description"] = df["ticket_description"].apply(light_clean)

    report["rows_after"] = len(df)
    report["missing_after"] = df.isnull().sum().to_dict()
    report["duplicates_after"] = int(df.duplicated().sum())

    return df, report


if __name__ == "__main__":
    df = load_data()

    print("=" * 60)
    print(" TASK 1: DATA PREPARATION")
    print("=" * 60)

    print("\n--- Raw dataset preview ---")
    print(df.head())

    print("\n--- Basic info ---")
    print(df.info())

    print("\n--- Missing values (before cleaning) ---")
    print(df.isnull().sum())

    print("\n--- Duplicate records (before cleaning) ---")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    cleaned_df, report = prepare_dataset(df)

    print("\n--- Cleaning summary ---")
    print(f"Rows before cleaning: {report['rows_before']}")
    print(f"Rows after cleaning:  {report['rows_after']}")
    print(f"Duplicates removed:   {report['duplicates_before']}")

    print("\n--- Missing values (after cleaning) ---")
    print(pd.Series(report["missing_after"]))

    print("\n--- Sample cleaned text ---")
    print(cleaned_df[["ticket_description", "clean_description"]].head())

    cleaned_df.to_csv("data/tickets_cleaned.csv", index=False)
    print("\nSaved cleaned dataset to data/tickets_cleaned.csv")
