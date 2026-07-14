# Brent Oil Change Point Analysis

**10 Academy: Artificial Intelligence Mastery — Week 10 Challenge**
Change Point Analysis and Statistical Modeling of Time Series Data

Client context: Birhan Energies, an energy-sector consultancy, needs to
understand how major political and economic events (wars, sanctions, OPEC
decisions, financial crises) have shifted Brent crude oil prices, in order to
support investors, policymakers, and energy companies with data-driven
insight.

## Project Objective

Quantify how major geopolitical and economic events have shifted the
statistical behavior of Brent crude oil prices (1987–2022) using Bayesian
change point detection, and communicate the results through a written report
and an interactive dashboard.

## Repository Structure

```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml          # CI: runs pytest on every push/PR
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   ├── BrentOilPrices.csv         # raw daily price data (Date, Price)
│   └── brent_oil_events.csv       # compiled key events dataset
├── docs/
│   ├── analysis_workflow.md       # Task 1: plan, change point model
│   │                                explanation, expected outputs
│   └── assumptions_and_limitations.md
├── notebooks/
│   ├── 01_eda_time_series_properties.py    # EDA script (trend/
│   │                                          stationarity/volatility)
│   └── 01_eda_time_series_properties.ipynb # same analysis as a notebook
├── outputs/                       # generated plots (created on run)
├── src/
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_data.py               # starter unit tests
└── scripts/
    └── __init__.py
```

## Data

**Source**: Historical Brent oil prices, daily, May 20 1987 – Nov 14 2022
(9,011 observations, no missing values). Columns: `Date` (day-month-year,
e.g. `20-May-87`), `Price` (USD per barrel).

Place `BrentOilPrices.csv` in `data/` before running any notebook or script.

**Compiled events dataset** (`data/brent_oil_events.csv`): 16 major
geopolitical/economic events (wars, OPEC decisions, sanctions, financial
crises) with dates, categories, descriptions, and expected price direction —
used to cross-reference against detected change points.

## Setup

```bash
# clone the repo
git clone https://github.com/<your-username>/brent-oil-change-point-analysis.git
cd brent-oil-change-point-analysis

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Running the Analysis

**EDA (trend, stationarity, volatility) — Task 1:**

```bash
# from the project root
python notebooks/01_eda_time_series_properties.py
```

Or open `notebooks/01_eda_time_series_properties.ipynb` in Jupyter/VS Code
and run all cells. Generated plots are saved to `outputs/`.

**Running tests:**

```bash
pytest tests/ -v
```

## Methodology Summary

1. **Data Ingestion & Cleaning** — load daily prices, parse dates, sort
   chronologically.
2. **EDA** — trend inspection, log-return transform, Augmented Dickey-Fuller
   stationarity testing, rolling volatility analysis.
3. **Event Compilation** — structured dataset of major events likely to have
   shifted prices.
4. **Bayesian Change Point Modeling** (PyMC) — discrete-uniform prior over the
   switch point `tau`, separate before/after mean parameters, Normal
   likelihood, MCMC sampling.
5. **Model Diagnostics** — convergence checks (`r_hat`, trace plots).
6. **Insight Generation** — match detected change points to compiled events
   and quantify the magnitude of each shift.
7. **Dashboard** — Flask backend + React frontend for interactive
   stakeholder exploration of results.

Full details: see `docs/analysis_workflow.md`.

## Key EDA Findings (Task 1)

- **Trend**: raw price series is clearly non-stationary — flat through the
  1990s, a 2000s bull run, the 2008 spike/crash, the 2014–16 collapse, the
  2020 COVID crash, and the 2021–22 surge.
- **Stationarity**: raw price ADF p-value = 0.29 (non-stationary); log
  returns ADF p-value < 0.0001 (stationary) — confirming log returns are the
  correct input for the change point model.
- **Volatility**: clearly clustered rather than constant, with the largest
  spike in the full series occurring in 2020.

## Assumptions and Limitations

See `docs/assumptions_and_limitations.md` for the full discussion, including
an important note: **this analysis identifies statistical correlation in
time between detected change points and known events — it does not prove
causation.** Any event association presented is a hypothesis based on
temporal proximity, not a verified causal claim.

## Team

Tutors: Kerod, Feven, Mahbubah
Slack: `#all-week10`

## License

Educational project for the 10 Academy AI Mastery program.
