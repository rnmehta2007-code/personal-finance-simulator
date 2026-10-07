# Personal Finance Simulator

A structured Python and Data Science project analyzing 32,424 household financial profiles and projecting 10-year wealth trajectories under 3 financial growth scenarios.

---

## Project Overview
This project models personal financial independence goals using compound interest formulas and inflation adjustments. It evaluates household income, living expenses, existing savings, and debt obligations to project financial milestone completion rates.

### Core Features:
- **Modular Pipeline**: Standalone Python scripts for data loading, cleaning, EDA, statistical testing, visualization, and simulation.
- **SciPy Hypothesis Testing**: Statistical inference (Pearson correlation, Welch t-test, One-way ANOVA, Chi-square independence, Normality checks).
- **Compound Growth Simulation**: Multi-scenario projections evaluating base savings vs. increased savings rate and investment returns.
- **Interactive Web App**: Streamlit web dashboard with interactive sliders and Plotly growth charts.
- **Single Notebook**: Combined Jupyter notebook `Personal_Finance_Simulator.ipynb` containing all code, markdown notes, and pre-executed outputs.

---

## Dataset Description
- **File**: `personal_finance_dataset.csv`
- **Rows**: 32,424 household records
- **Columns**: 20 attributes (demographics, income, expenses, savings, loan details, credit scores, regions, and dates)
- **Currency**: USD ($)

---

## Project Structure
```
personal_finance_simulator/
├── 1_data_loading.py               # Step 1: Data loading & inspection
├── 2_data_cleaning_preprocessing.py # Step 2: Data cleaning & feature engineering
├── 3_eda_statistical_analysis.py   # Step 3: EDA & SciPy statistical hypothesis tests
├── 4_data_visualization.py        # Step 4: Numbered PNG charts generation (fig01 to fig08)
├── 5_results_analysis.py           # Step 5: 10-Year simulation & sensitivity analysis
├── simulator.py                    # Mathematical compound interest engine
├── app.py                          # Streamlit web application
├── Personal_Finance_Simulator.ipynb# Executed Jupyter notebook (Steps 1 to 5)
├── personal_finance_dataset.csv    # Raw dataset (32,424 rows)
├── personal_finance_clean.csv      # Processed dataset with engineered features
├── results_summary.txt             # Generated text summary report of results
├── requirements.txt                # Required Python packages
├── README.md                       # Project documentation
└── figures/                        # Directory containing generated PNG figures
```

---

## How to Run

### 1. Execute Pipeline Scripts (Steps 1 to 5)
Run each script sequentially from the project root:

```bash
python 1_data_loading.py
python 2_data_cleaning_preprocessing.py
python 3_eda_statistical_analysis.py
python 4_data_visualization.py
python 5_results_analysis.py
```

### 2. Run the Interactive Streamlit Web App
Launch the interactive web application in your browser:

```bash
streamlit run app.py
```

### 3. Open the Combined Notebook
Open `Personal_Finance_Simulator.ipynb` in Jupyter Notebook or VS Code to review all executed code cells, markdown explanations, and inline figures.
