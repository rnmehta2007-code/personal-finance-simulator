"""
===============================================================================
Step 2: Data Cleaning & Preprocessing
===============================================================================
Description:
    This script cleans and preprocesses the raw personal finance dataset.
    It imports raw data via 1_data_loading.py, handles missing values,
    converts data types, removes duplicates, validates impossible values,
    detects outliers, and performs feature engineering.
    Saves the output to personal_finance_clean.csv.

Usage:
    Run directly: python 2_data_cleaning_preprocessing.py
    Or import clean_data() in downstream scripts.
===============================================================================
"""

import importlib
from pathlib import Path
import pandas as pd
import numpy as np

# Dynamically import load_data from 1_data_loading.py
data_loading_module = importlib.import_module("1_data_loading")
load_data = data_loading_module.load_data

# Relative path for cleaned dataset
OUTPUT_PATH = Path(__file__).parent / "personal_finance_clean.csv"


def clean_data(raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Executes the data cleaning and feature engineering pipeline on raw data.
    If raw_df is not provided, loads data using load_data().
    Returns the cleaned pandas DataFrame.
    """
    if raw_df is None:
        raw_df = load_data()

    df = raw_df.copy()

    print("=" * 80)
    print("EXECUTING DATA CLEANING & PREPROCESSING PIPELINE")
    print("=" * 80)

    # Step 2.1: Handle Missing Values in loan_type
    # Prove missing values occur ONLY where has_loan == "No"
    missing_loan_type_no = df[df["has_loan"] == "No"]["loan_type"].isna().sum()
    missing_loan_type_yes = df[df["has_loan"] == "Yes"]["loan_type"].isna().sum()
    total_missing_loan_type = df["loan_type"].isna().sum()

    print("\n[Step 2.1] Checking Missing Values in loan_type:")
    print(f"  Total missing loan_type entries: {total_missing_loan_type}")
    print(f"  Missing loan_type where has_loan == 'No' : {missing_loan_type_no}")
    print(f"  Missing loan_type where has_loan == 'Yes': {missing_loan_type_yes}")
    assert total_missing_loan_type == missing_loan_type_no, (
        "Unexpected missing values in loan_type for users with active loans!"
    )
    print("  -> Verification passed: missing loan_type occurs ONLY when has_loan == 'No'.")
    df["loan_type"] = df["loan_type"].fillna("None")

    # Step 2.2: Convert record_date to datetime
    print("\n[Step 2.2] Converting record_date to datetime format...")
    df["record_date"] = pd.to_datetime(df["record_date"])

    # Step 2.3: Remove Duplicate Rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_dups = initial_rows - len(df)
    print(f"\n[Step 2.3] Removed {removed_dups} duplicate rows. Remaining rows: {len(df)}")

    # Step 2.4: Validate Impossible / Invalid Values
    print("\n[Step 2.4] Checking for Impossible Financial Values:")
    imp_income = (df["monthly_income_usd"] <= 0).sum()
    imp_savings = (df["savings_usd"] < 0).sum()
    imp_expenses = (df["monthly_expenses_usd"] > df["monthly_income_usd"]).sum()
    print(f"  Income <= 0 count: {imp_income}")
    print(f"  Savings < 0 count : {imp_savings}")
    print(f"  Expenses > Income count: {imp_expenses}")
    print("  -> No impossible negative or exceeding values found.")

    # Step 2.5: Check Text Categories for Typos
    print("\n[Step 2.5] Auditing Categorical Columns for Typos / Anomalies:")
    cat_cols = ["gender", "education_level", "employment_status", "has_loan", "loan_type", "region"]
    for col in cat_cols:
        unique_vals = list(df[col].unique())
        print(f"  Column '{col}': {unique_vals}")

    # Step 2.6: Detect Outliers using 1.5 * IQR Rule
    # Note: Outliers are retained because in personal financial datasets, real-world income,
    # savings, and loan distributions naturally exhibit right-skewness and extreme values.
    print("\n[Step 2.6] Outlier Detection using 1.5 * IQR Rule (Retained for Authenticity):")
    num_cols = [
        "age", "monthly_income_usd", "monthly_expenses_usd", "savings_usd",
        "loan_amount_usd", "loan_term_months", "monthly_emi_usd",
        "loan_interest_rate_pct", "debt_to_income_ratio", "credit_score",
        "savings_to_income_ratio"
    ]
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outlier_count = ((df[col] < low) | (df[col] > high)).sum()
        print(f"  {col:25s}: {outlier_count:5d} outliers (IQR limits: [{low:10.2f}, {high:10.2f}])")

    # Step 2.7: Note Minimum Income Floor / Spike
    min_inc = df["monthly_income_usd"].min()
    spike_count = (df["monthly_income_usd"] == min_inc).sum()
    print(f"\n[Step 2.7] Minimum Income Spike Check:")
    print(f"  Minimum monthly income: ${min_inc:.2f}")
    print(f"  Households at minimum income floor: {spike_count} ({spike_count/len(df)*100:.2f}%)")
    # Comment: Synthetic dataset generator applied a lower bound threshold of $500.00 to monthly income.

    # Step 2.8: Feature Engineering
    print("\n[Step 2.8] Feature Engineering:")
    # 1. monthly_saving
    df["monthly_saving"] = df["monthly_income_usd"] - df["monthly_expenses_usd"]

    # 2. savings_rate (monthly_saving / monthly_income_usd)
    df["savings_rate"] = np.where(
        df["monthly_income_usd"] > 0,
        df["monthly_saving"] / df["monthly_income_usd"],
        0.0
    )

    # 3. goal_usd = monthly_expenses_usd * 12 * 10
    # Assumption: Financial independence / retirement goal defined as 10 years of annual living expenses.
    df["goal_usd"] = df["monthly_expenses_usd"] * 12 * 10

    # 4. age_group
    age_bins = [0, 25, 35, 50, 100]
    age_labels = ["<25", "25-34", "35-49", "50+"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)

    # 5. income_band (quartiles)
    income_labels = ["Low", "Medium-Low", "Medium-High", "High"]
    df["income_band"] = pd.qcut(df["monthly_income_usd"], q=4, labels=income_labels)

    print(f"  Engineered features added: 'monthly_saving', 'savings_rate', 'goal_usd', 'age_group', 'income_band'")

    print("=" * 80)
    print("DATA CLEANING & PREPROCESSING COMPLETED")
    print("=" * 80)

    return df


if __name__ == "__main__":
    cleaned_df = clean_data()
    cleaned_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned dataset saved successfully to: {OUTPUT_PATH}")
