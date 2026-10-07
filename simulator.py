"""
===============================================================================
Personal Finance Compound Interest Simulator Module
===============================================================================
Description:
    Core mathematical engine for compound growth, inflation adjustment,
    and multi-scenario financial wealth projections.

Functions:
    1. future_value(start, monthly, annual_rate, months)
    2. real_value(nominal, inflation, months)
    3. scenario_values(start, income, expenses, extra_pct, invest_return_pct,
                       inflation_pct, months, bank_rate_pct=3.0)

Usage:
    Import in simulation/analysis scripts or Streamlit app.
===============================================================================
"""

import numpy as np


def future_value(start: float | np.ndarray, monthly: float | np.ndarray, annual_rate: float, months: float | np.ndarray):
    """
    Computes the nominal future value of an initial deposit plus regular monthly savings.

    Parameters:
        start (float or np.ndarray): Initial principal/savings balance.
        monthly (float or np.ndarray): Regular monthly savings contribution amount.
        annual_rate (float): Annual nominal interest/return rate (as a decimal, e.g., 0.10 for 10%).
        months (float or np.ndarray): Total number of investment months.

    Returns:
        float or np.ndarray: Accumulated nominal future value.
    """
    r = annual_rate / 12.0
    months = np.asarray(months, dtype=float)
    if r == 0:
        return start + monthly * months
    growth = (1.0 + r) ** months
    return start * growth + monthly * (growth - 1.0) / r


def real_value(nominal: float | np.ndarray, inflation: float, months: float | np.ndarray):
    """
    Converts nominal future value to real inflation-adjusted purchasing power in today's terms.

    Parameters:
        nominal (float or np.ndarray): Nominal monetary balance.
        inflation (float): Annual inflation rate (as a decimal, e.g., 0.05 for 5%).
        months (float or np.ndarray): Total duration in months.

    Returns:
        float or np.ndarray: Inflation-adjusted real monetary value.
    """
    return nominal / ((1.0 + inflation) ** (np.asarray(months, dtype=float) / 12.0))


def scenario_values(start: float | np.ndarray, income: float | np.ndarray, expenses: float | np.ndarray,
                    extra_pct: float, invest_return_pct: float, inflation_pct: float,
                    months: float | np.ndarray, bank_rate_pct: float = 3.0):
    """
    Calculates inflation-adjusted real values across 3 financial scenarios:
        - Scenario 1: Base plan saving current monthly surplus at bank savings rate (default 3%).
        - Scenario 2: Save extra percentage of monthly income at bank savings rate.
        - Scenario 3: Save extra percentage and invest surplus at investment return rate.

    Returns:
        tuple: (s1_real, s2_real, s3_real)
    """
    saving = income - expenses
    extra = income * (extra_pct / 100.0)
    
    s1 = future_value(start, saving, bank_rate_pct / 100.0, months)
    s2 = future_value(start, saving + extra, bank_rate_pct / 100.0, months)
    s3 = future_value(start, saving + extra, invest_return_pct / 100.0, months)
    
    infl = inflation_pct / 100.0
    return tuple(real_value(s, infl, months) for s in (s1, s2, s3))


if __name__ == "__main__":
    print("Running simulator.py sanity check...")

    # Sanity Test: future_value(0, 10000, 0, 12) MUST equal 120000
    test_fv = future_value(0, 10000, 0, 12)
    print(f"Sanity Test: future_value(0, 10000, 0, 12) = {test_fv}")
    assert test_fv == 120000.0, f"Sanity test failed! Expected 120000.0, got {test_fv}"
    
    # Second sanity check with interest
    test_fv_interest = future_value(1000, 100, 0.12, 12)
    print(f"Sanity Test with 12% interest: future_value(1000, 100, 0.12, 12) = {test_fv_interest:.2f}")
    
    print("All simulator.py sanity checks passed successfully!")