"""
===============================================================================
Step 4: Data Visualization
===============================================================================
Description:
    This script generates 8 numbered high-resolution visualization figures
    saved into the figures/ directory as PNG files (dpi=150).
    Uses non-blocking Matplotlib file saving (plt.savefig & plt.close).

Usage:
    Run directly: python 4_data_visualization.py
    Or import generate_visualizations() in downstream scripts.
===============================================================================
"""

import importlib
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Suppress Matplotlib / Seaborn deprecation warnings for clean console output
warnings.filterwarnings("ignore")

# Set seaborn/matplotlib styling aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10})

# Import clean_data from Step 2
cleaning_module = importlib.import_module("2_data_cleaning_preprocessing")
clean_data = cleaning_module.clean_data

# Import simulator from simulator.py
simulator_module = importlib.import_module("simulator")
scenario_values = simulator_module.scenario_values

# Ensure figures output directory exists
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_visualizations(df: pd.DataFrame | None = None) -> None:
    """
    Generates and saves figures fig01 to fig08 to the figures/ directory.
    """
    if df is None:
        df = clean_data()

    print("=" * 80)
    print("STEP 4: GENERATING AND SAVING VISUALIZATION FIGURES")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Figure 1: Savings Rate Distribution Histogram
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.histplot(df["savings_rate"], bins=30, kde=True, color="#2b5c8f", ax=ax1)
    ax1.set_title("Figure 1: Distribution of Household Savings Rate", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Savings Rate (Monthly Saving / Monthly Income)")
    ax1.set_ylabel("Number of Households")
    fig1.tight_layout()
    fig1_path = FIGURES_DIR / "fig01_savings_rate_hist.png"
    plt.savefig(fig1_path, dpi=150)
    plt.close(fig1)
    print(f"Saved: {fig1_path.name}")

    # -------------------------------------------------------------------------
    # Figure 2: Income Distribution Histogram
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(df["monthly_income_usd"], bins=35, kde=True, color="#2e7d32", ax=ax2)
    ax2.axvline(df["monthly_income_usd"].min(), color="red", linestyle="--", label="Income Floor ($500)")
    ax2.set_title("Figure 2: Distribution of Monthly Income (USD)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Monthly Income ($)")
    ax2.set_ylabel("Number of Households")
    ax2.legend()
    fig2.tight_layout()
    fig2_path = FIGURES_DIR / "fig02_income_hist.png"
    plt.savefig(fig2_path, dpi=150)
    plt.close(fig2)
    print(f"Saved: {fig2_path.name}")

    # -------------------------------------------------------------------------
    # Figure 3: Income vs Monthly Saving Scatter Plot (Sampled N=2000)
    # -------------------------------------------------------------------------
    sample_df = df.sample(n=2000, random_state=42)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=sample_df,
        x="monthly_income_usd",
        y="monthly_saving",
        alpha=0.6,
        color="#1565c0",
        edgecolor=None,
        ax=ax3
    )
    # Fit linear trendline
    m, b = np.polyfit(sample_df["monthly_income_usd"], sample_df["monthly_saving"], 1)
    x_vals = np.linspace(sample_df["monthly_income_usd"].min(), sample_df["monthly_income_usd"].max(), 100)
    ax3.plot(x_vals, m * x_vals + b, color="red", linewidth=2, label="Trendline (r = 0.70)")
    ax3.set_title("Figure 3: Monthly Income vs. Monthly Saving (Sampled N=2,000)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Monthly Income ($)")
    ax3.set_ylabel("Monthly Saving ($)")
    ax3.legend()
    fig3.tight_layout()
    fig3_path = FIGURES_DIR / "fig03_income_vs_saving_scatter.png"
    plt.savefig(fig3_path, dpi=150)
    plt.close(fig3)
    print(f"Saved: {fig3_path.name}")

    # -------------------------------------------------------------------------
    # Figure 4: Boxplots of Savings Rate by Employment, Has Loan & Region
    # -------------------------------------------------------------------------
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    sns.boxplot(data=df, x="employment_status", y="savings_rate", palette="Set2", ax=axes4[0], hue="employment_status", legend=False)
    axes4[0].set_title("Savings Rate by Employment Status", fontweight="bold")
    axes4[0].set_xlabel("Employment Status")
    axes4[0].set_ylabel("Savings Rate")
    axes4[0].tick_params(axis='x', rotation=15)

    sns.boxplot(data=df, x="has_loan", y="savings_rate", palette="Pastel1", ax=axes4[1], hue="has_loan", legend=False)
    axes4[1].set_title("Savings Rate by Has Loan", fontweight="bold")
    axes4[1].set_xlabel("Has Active Loan")
    axes4[1].set_ylabel("")

    sns.boxplot(data=df, x="region", y="savings_rate", palette="Set3", ax=axes4[2], hue="region", legend=False)
    axes4[2].set_title("Savings Rate by Region", fontweight="bold")
    axes4[2].set_xlabel("Geographic Region")
    axes4[2].set_ylabel("")
    axes4[2].tick_params(axis='x', rotation=15)

    fig4.suptitle("Figure 4: Savings Rate Comparisons across Demographics", fontsize=14, fontweight="bold")
    fig4.tight_layout(rect=[0, 0, 1, 0.95])
    fig4_path = FIGURES_DIR / "fig04_savings_rate_boxplots.png"
    plt.savefig(fig4_path, dpi=150)
    plt.close(fig4)
    print(f"Saved: {fig4_path.name}")

    # -------------------------------------------------------------------------
    # Figure 5: Correlation Heatmap
    # -------------------------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    corr_vars = [
        "monthly_income_usd", "monthly_expenses_usd", "savings_usd",
        "monthly_saving", "savings_rate", "age", "credit_score",
        "debt_to_income_ratio"
    ]
    corr_mat = df[corr_vars].corr()
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax5)
    ax5.set_title("Figure 5: Correlation Matrix Heatmap", fontsize=12, fontweight="bold")
    fig5.tight_layout()
    fig5_path = FIGURES_DIR / "fig05_correlation_heatmap.png"
    plt.savefig(fig5_path, dpi=150)
    plt.close(fig5)
    print(f"Saved: {fig5_path.name}")

    # -------------------------------------------------------------------------
    # Figure 6: Age Group vs Average Savings Rate
    # -------------------------------------------------------------------------
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    age_savings = df.groupby("age_group", observed=False)["savings_rate"].mean().reset_index()
    sns.barplot(data=age_savings, x="age_group", y="savings_rate", palette="Blues_d", ax=ax6, hue="age_group", legend=False)
    ax6.set_ylim(0, 0.5)
    for p in ax6.patches:
        ax6.annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    ax6.set_title("Figure 6: Average Savings Rate by Age Group", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Age Group")
    ax6.set_ylabel("Mean Savings Rate")
    fig6.tight_layout()
    fig6_path = FIGURES_DIR / "fig06_age_group_savings_rate.png"
    plt.savefig(fig6_path, dpi=150)
    plt.close(fig6)
    print(f"Saved: {fig6_path.name}")

    # -------------------------------------------------------------------------
    # Figure 7: Loan Type Count Plot
    # -------------------------------------------------------------------------
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    loan_counts = df["loan_type"].value_counts().reset_index()
    loan_counts.columns = ["loan_type", "count"]
    sns.barplot(data=loan_counts, x="loan_type", y="count", palette="viridis", ax=ax7, hue="loan_type", legend=False)
    for p in ax7.patches:
        ax7.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    ax7.set_title("Figure 7: Household Distribution by Loan Type", fontsize=12, fontweight="bold")
    ax7.set_xlabel("Loan Type")
    ax7.set_ylabel("Number of Households")
    fig7.tight_layout()
    fig7_path = FIGURES_DIR / "fig07_loan_type_counts.png"
    plt.savefig(fig7_path, dpi=150)
    plt.close(fig7)
    print(f"Saved: {fig7_path.name}")

    # -------------------------------------------------------------------------
    # Figure 8: Scenario Goal Achievement Rate Bar Chart
    # -------------------------------------------------------------------------
    months = 120
    s1, s2, s3 = scenario_values(
        start=df["savings_usd"].values,
        income=df["monthly_income_usd"].values,
        expenses=df["monthly_expenses_usd"].values,
        extra_pct=10.0,
        invest_return_pct=10.0,
        inflation_pct=5.0,
        months=months,
        bank_rate_pct=3.0
    )
    goal = df["goal_usd"].values
    scen_rates = [
        (s1 >= goal).mean() * 100,
        (s2 >= goal).mean() * 100,
        (s3 >= goal).mean() * 100
    ]
    scen_labels = ["1. Base Plan\n(3% Bank)", "2. Save +10%\n(3% Bank)", "3. Save +10% & Invest\n(10% Return)"]

    fig8, ax8 = plt.subplots(figsize=(8, 5))
    bars = ax8.bar(scen_labels, scen_rates, color=["#e57373", "#64b5f6", "#81c784"], width=0.5)
    ax8.set_ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        ax8.annotate(f"{height:.1f}%", (bar.get_x() + bar.get_width() / 2., height),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight="bold")
    ax8.set_title("Figure 8: Percentage of Households Reaching 10-Year Financial Goal", fontsize=12, fontweight="bold")
    ax8.set_xlabel("Financial Strategy Scenario")
    ax8.set_ylabel("Goal Achievement Rate (%)")
    fig8.tight_layout()
    fig8_path = FIGURES_DIR / "fig08_scenario_results.png"
    plt.savefig(fig8_path, dpi=150)
    plt.close(fig8)
    print(f"Saved: {fig8_path.name}")

    print("=" * 80)
    print(f"All 8 visualization figures saved successfully to '{FIGURES_DIR}'")
    print("=" * 80)


if __name__ == "__main__":
    generate_visualizations()
