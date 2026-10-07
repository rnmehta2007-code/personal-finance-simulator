"""
===============================================================================
Step 1: Data Loading & Initial Inspection
===============================================================================
Description:
    This script loads the raw personal finance dataset from a relative CSV path
    and performs basic structural checks (shape, column names, data types,
    sample rows, missing values, duplicates, and summary statistics).

Usage:
    Run directly: python 1_data_loading.py
    Or import load_data() in downstream scripts.
===============================================================================
"""

from pathlib import Path
import pandas as pd

# Define relative path to the dataset
DATASET_PATH = Path(__file__).parent / "personal_finance_dataset.csv"


def load_data(filepath: str | Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load the personal finance dataset from CSV file path into a pandas DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")
    df = pd.read_csv(path)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print basic inspection metrics for the loaded dataset.
    """
    print("=" * 80)
    print("DATASET INITIAL INSPECTION SUMMARY")
    print("=" * 80)

    # Step 1.1: Dataset Shape
    print(f"\n[1] Dataset Shape (Rows, Columns): {df.shape}")

    # Step 1.2: Column Names & Data Types
    print("\n[2] Column Names & Data Types:")
    print(df.dtypes)

    # Step 1.3: Head (First 5 Rows)
    print("\n[3] First 5 Rows (Head):")
    print(df.head())

    # Step 1.4: Missing Values Count
    print("\n[4] Missing Values Count per Column:")
    missing = df.isna().sum()
    print(missing)

    # Step 1.5: Duplicate Rows Count
    print(f"\n[5] Duplicate Rows Count: {df.duplicated().sum()}")

    # Step 1.6: Summary Statistics (Descriptive Stats)
    print("\n[6] Summary Statistics (Descriptive Stats for Numerical Columns):")
    print(df.describe().T)
    print("=" * 80)


if __name__ == "__main__":
    print("Executing Step 1: Data Loading...")
    df_raw = load_data()
    inspect_data(df_raw)
    print("Data loading completed successfully!")
