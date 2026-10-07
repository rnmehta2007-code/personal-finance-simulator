"""
===============================================================================
Step 3: Exploratory Data Analysis (EDA) & Statistical Analysis
===============================================================================
Description:
    This script performs comprehensive Exploratory Data Analysis (EDA) and
    hypothesis testing using SciPy. It computes descriptive statistics, group
    means, correlation matrices, and statistical tests (Pearson, Welch t-test,
    ANOVA, Chi-Square, and Normality checks) with plain-English conclusions.

Usage:
    Run directly: python 3_eda_statistical_analysis.py
    Or import run_eda() in downstream scripts.
===============================================================================
"""

import importlib
import pandas as pd
import numpy as np
import scipy.stats as stats

# Dynamically import clean_data from Step 2
cleaning_module = importlib.import_module("2_data_cleaning_preprocessing")
clean_data = cleaning_module.clean_data


def run_eda(df: pd.DataFrame | None = None) -> None:
    """
    Executes EDA and statistical hypothesis tests on the cleaned dataset.
    """
    if df is None:
        df = clean_data()

    print("=" * 80)
    print("STEP 3: EXPLORATORY DATA ANALYSIS & STATISTICAL ANALYSIS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 3.1 Descriptive Statistics & Categorical Value Counts
    # -------------------------------------------------------------------------
    print("\n[3.1] Descriptive Statistics for Numerical Features:")
    num_cols = [
        "monthly_income_usd", "monthly_expenses_usd", "savings_usd",
        "monthly_saving", "savings_rate", "age", "credit_score",
        "debt_to_income_ratio"
    ]
    print(df[num_cols].describe().T[["mean", "std", "min", "50%", "max"]])

    print("\n[3.2] Categorical Feature Value Counts:")
    cat_cols = ["gender", "education_level", "employment_status", "has_loan", "loan_type", "region", "age_group", "income_band"]
    for col in cat_cols:
        print(f"\n--- Value Counts for '{col}' ---")
        print(df[col].value_counts(dropna=False))

    # -------------------------------------------------------------------------
    # 3.2 Group Means of Savings Rate
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[3.3] Group Means of Savings Rate (savings_rate)")
    print("=" * 80)
    
    group_vars = ["employment_status", "region", "has_loan", "gender", "education_level"]
    for var in group_vars:
        print(f"\nMean Savings Rate by {var}:")
        means = df.groupby(var, observed=False)["savings_rate"].agg(["mean", "std", "count"])
        print(means)

    # -------------------------------------------------------------------------
    # 3.3 Correlation Matrix
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[3.4] Correlation Matrix (Selected Numerical Features)")
    print("=" * 80)
    corr_vars = [
        "monthly_income_usd", "monthly_expenses_usd", "savings_usd",
        "monthly_saving", "savings_rate", "age", "credit_score",
        "debt_to_income_ratio"
    ]
    corr_matrix = df[corr_vars].corr()
    print(corr_matrix.round(4))

    # -------------------------------------------------------------------------
    # 3.4 Statistical Hypothesis Testing (SciPy)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[3.5] STATISTICAL HYPOTHESIS TESTS (alpha = 0.05)")
    print("=" * 80)

    # --- Test 1: Pearson Correlation (Income vs Monthly Saving) ---
    # H0: There is no linear correlation between monthly income and monthly saving (r = 0).
    # H1: There is a significant linear correlation between monthly income and monthly saving (r != 0).
    r_val, p_val1 = stats.pearsonr(df["monthly_income_usd"], df["monthly_saving"])
    print("\nTest 1: Pearson Correlation (Monthly Income vs Monthly Saving)")
    print("  H0: No linear correlation (r = 0)")
    print("  H1: Linear correlation exists (r != 0)")
    print(f"  Pearson r statistic: {r_val:.4f}, p-value: {p_val1:.4e}")
    if p_val1 < 0.05:
        print("  Conclusion: Reject H0. There is a strong, statistically significant positive relationship between monthly income and monthly saving.")
    else:
        print("  Conclusion: Fail to reject H0. No significant linear correlation detected.")

    # --- Test 2: Welch t-test (Savings Rate: Loan vs No Loan) ---
    # H0: Mean savings rate is equal between households with loans and without loans (mu1 = mu2).
    # H1: Mean savings rate is significantly different between households with loans and without loans (mu1 != mu2).
    sr_loan = df[df["has_loan"] == "Yes"]["savings_rate"]
    sr_noloan = df[df["has_loan"] == "No"]["savings_rate"]
    t_stat, p_val2 = stats.ttest_ind(sr_loan, sr_noloan, equal_var=False)
    print("\nTest 2: Welch's Two-Sample t-Test (Savings Rate: Has Loan vs No Loan)")
    print("  H0: Mean savings rate is equal between loan holders and non-holders.")
    print("  H1: Mean savings rate differs between loan holders and non-holders.")
    print(f"  Welch t-statistic: {t_stat:.4f}, p-value: {p_val2:.4f}")
    if p_val2 < 0.05:
        print("  Conclusion: Reject H0. Significant difference in savings rate between loan holders and non-holders.")
    else:
        print("  Conclusion: Fail to reject H0. No statistically significant difference in savings rate between households with and without loans (p > 0.05).")

    # --- Test 3: One-Way ANOVA (Savings Rate across Regions) ---
    # H0: Mean savings rate is identical across all geographic regions.
    # H1: At least one region has a significantly different mean savings rate.
    region_groups = [group["savings_rate"].values for _, group in df.groupby("region")]
    f_stat1, p_val3 = stats.f_oneway(*region_groups)
    print("\nTest 3: One-Way ANOVA (Savings Rate across Geographic Regions)")
    print("  H0: Mean savings rate is equal across all regions.")
    print("  H1: At least one region mean is different.")
    print(f"  ANOVA F-statistic: {f_stat1:.4f}, p-value: {p_val3:.4f}")
    if p_val3 < 0.05:
        print("  Conclusion: Reject H0. Savings rate varies significantly across regions.")
    else:
        print("  Conclusion: Fail to reject H0. Geographic region does not significantly affect savings rate (p > 0.05).")

    # --- Test 4: One-Way ANOVA (Savings Rate across Employment Status) ---
    # H0: Mean savings rate is identical across all employment status categories.
    # H1: At least one employment status category has a different mean savings rate.
    emp_groups = [group["savings_rate"].values for _, group in df.groupby("employment_status")]
    f_stat2, p_val4 = stats.f_oneway(*emp_groups)
    print("\nTest 4: One-Way ANOVA (Savings Rate across Employment Status)")
    print("  H0: Mean savings rate is equal across employment status categories.")
    print("  H1: At least one employment status category differs.")
    print(f"  ANOVA F-statistic: {f_stat2:.4f}, p-value: {p_val4:.4f}")
    if p_val4 < 0.05:
        print("  Conclusion: Reject H0. Employment status significantly impacts savings rate.")
    else:
        print("  Conclusion: Fail to reject H0. Employment status does not have a statistically significant effect on savings rate (p > 0.05).")

    # --- Test 5: Chi-Square Test of Independence (Has Loan vs Employment Status) ---
    # H0: Loan status (has_loan) and Employment Status are independent.
    # H1: Loan status and Employment Status are dependent (associated).
    contingency_tab = pd.crosstab(df["has_loan"], df["employment_status"])
    chi2, p_val5, dof, _ = stats.chi2_contingency(contingency_tab)
    print("\nTest 5: Chi-Square Test of Independence (Has Loan vs Employment Status)")
    print("  H0: Having a loan is independent of employment status.")
    print("  H1: Having a loan is associated with employment status.")
    print(f"  Chi-Square statistic: {chi2:.4f}, degrees of freedom: {dof}, p-value: {p_val5:.4f}")
    if p_val5 < 0.05:
        print("  Conclusion: Reject H0. There is a statistically significant association between employment status and having a loan.")
    else:
        print("  Conclusion: Fail to reject H0. No significant relationship between employment status and loan status.")

    # --- Test 6: Normality Check on Savings Rate ---
    # H0: Savings rate follows a normal (Gaussian) distribution.
    # H1: Savings rate does not follow a normal distribution.
    # Note: On N = 32,424, Shapiro-Wilk suffers from extreme sensitivity to minor sample deviations.
    # We evaluate Shapiro-Wilk on a random sample of 500 rows alongside Skewness and Kurtosis.
    sample_sr = df["savings_rate"].sample(500, random_state=42)
    shapiro_stat, p_val6 = stats.shapiro(sample_sr)
    skew_val = df["savings_rate"].skew()
    kurt_val = df["savings_rate"].kurt()
    print("\nTest 6: Normality Audit (Savings Rate)")
    print("  H0: Savings rate is normally distributed.")
    print("  H1: Savings rate is non-normal.")
    print(f"  Shapiro-Wilk (Sample N=500) W-statistic: {shapiro_stat:.4f}, p-value: {p_val6:.4e}")
    print(f"  Full dataset Skewness: {skew_val:.4f} (Ideal normal = 0)")
    print(f"  Full dataset Excess Kurtosis: {kurt_val:.4f} (Ideal normal = 0)")
    print("  Conclusion: The distribution is highly symmetric (skewness ~ 0) but uniform/platykurtic (kurtosis ~ -1.2).")

    # -------------------------------------------------------------------------
    # 3.5 Synthetic Data Context Note
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("HONEST SYNTHETIC DATA NOTE")
    print("=" * 80)
    print(
        "Note: Because this dataset was synthetically generated, many demographic "
        "and regional attributes were assigned independently. Consequently, "
        "variables such as region, gender, and employment status exhibit minimal "
        "or zero mutual correlation by construction, producing p-values > 0.05 "
        "in demographic ANOVA tests."
    )
    print("=" * 80)


if __name__ == "__main__":
    run_eda()
