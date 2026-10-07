import numpy as np

def future_value(start, monthly, annual_rate, months):
    r = annual_rate / 12
    months = np.asarray(months, dtype=float)
    if r == 0:
        return start + monthly * months
    growth = (1 + r) ** months
    return start * growth + monthly * (growth - 1) / r

def real_value(nominal, inflation, months):
    return nominal / (1 + inflation) ** (np.asarray(months, dtype=float) / 12)

def scenario_values(start, income, expenses, extra_pct, invest_return_pct,
                    inflation_pct, months, bank_rate_pct=3.0):
    saving = income - expenses
    extra = income * extra_pct / 100
    s1 = future_value(start, saving, bank_rate_pct / 100, months)
    s2 = future_value(start, saving + extra, bank_rate_pct / 100, months)
    s3 = future_value(start, saving + extra, invest_return_pct / 100, months)
    infl = inflation_pct / 100
    return tuple(real_value(s, infl, months) for s in (s1, s2, s3))