import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from simulator import scenario_values

st.set_page_config(page_title="Personal Finance Simulator", layout="wide")
st.title("💰 Personal Finance Simulator")

def inr(x):
    """Show money in Indian style: lakh (L) and crore (Cr)."""
    x = float(x)
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:.2f} L"
    return f"₹{x:,.0f}"

st.sidebar.header("Your details")
income = st.sidebar.number_input("Monthly income (₹)", 1000, 10000000, 60000, step=1000)
expenses = st.sidebar.number_input("Monthly expenses (₹)", 1000, 10000000, 35000, step=1000)
savings = st.sidebar.number_input("Current savings (₹)", 0, 1000000000, 100000, step=10000)
goal = st.sidebar.number_input("Savings goal (₹)", 10000, 1000000000, 2500000, step=100000)
years = st.sidebar.slider("Years", 1, 40, 10)

st.sidebar.header("Assumptions")
extra = st.sidebar.slider("Extra savings (% of income)", 0, 30, 10)
ret = st.sidebar.slider("Investment return (% per year)", 0, 20, 10)
infl = st.sidebar.slider("Inflation (% per year)", 0, 15, 5)

if expenses >= income:
    st.warning("Your expenses are at or above your income, so the base plan saves nothing.")

months = np.arange(0, years * 12 + 1)
s1, s2, s3 = scenario_values(savings, income, expenses, extra, ret, infl, months)

fig = go.Figure()
for name, s in [("1. Current plan", s1), ("2. Save more", s2), ("3. Save more + invest", s3)]:
    fig.add_trace(go.Scatter(x=months / 12, y=s, name=name, mode="lines"))
fig.add_hline(y=goal, line_dash="dash", annotation_text="Goal")
fig.update_layout(xaxis_title="Years", yaxis_title="Balance in today's money (₹)", height=500)
st.plotly_chart(fig, width="stretch")

cols = st.columns(3)
for c, (name, s) in zip(cols, [("Current plan", s1), ("Save more", s2), ("Save more + invest", s3)]):
    reached = s >= goal
    when = f"Goal reached in {np.argmax(reached) / 12:.1f} years" if reached.any() else "Goal not reached"
    c.metric(name, inr(s[-1]), when)

@st.cache_data
def load_benchmark():
    return pd.read_csv("personal_finance_clean.csv")

try:
    d = load_benchmark()
    yours = (income - expenses) / income * 100
    st.info(f"Your savings rate: {yours:.1f}%  |  Dataset average: {d['savings_rate'].mean() * 100:.1f}%  "
            f"(from {len(d):,} synthetic households)")
except FileNotFoundError:
    pass

st.caption("Balances are inflation-adjusted. The current plan and 'save more' use a 3% savings account.")