"""
Task 2: Bayesian Change Point Modeling and Insight Generation
================================================================
Brent Oil Change Point Analysis — 10 Academy Week 10

Run from the project root:
    python notebooks/02_bayesian_change_point_analysis.py

Or paste into a Jupyter notebook cell-by-cell (split on the "# %%" markers).

Outputs:
    outputs/raw_price_series.png
    outputs/log_returns.png
    outputs/tau_posterior.png
    outputs/before_after_means.png
    outputs/trace_plot.png
    data/model_output.json   <- consumed by the Flask dashboard backend (Task 3)
"""

# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

OUTPUT_DIR = "outputs"
DATA_DIR = "data"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% -------------------------------------------------------------
# 1. Data Preparation and EDA
# -------------------------------------------------------------
df = pd.read_csv(f"{DATA_DIR}/BrentOilPrices.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%y")
df = df.sort_values("Date").reset_index(drop=True)

print(df.head())
print(f"Rows: {len(df)}  |  Range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# Raw price plot
plt.figure(figsize=(14, 5))
plt.plot(df["Date"], df["Price"], linewidth=0.8, color="#1f4e79")
plt.title("Brent Oil Price (Daily, USD/barrel)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/raw_price_series.png", dpi=150)
plt.close()

# Log returns for stationarity
df["log_price"] = np.log(df["Price"])
df["log_return"] = df["log_price"].diff()
df = df.dropna().reset_index(drop=True)

plt.figure(figsize=(14, 5))
plt.plot(df["Date"], df["log_return"], linewidth=0.5, color="#a83232")
plt.title("Brent Oil Log Returns (volatility clustering)")
plt.xlabel("Date")
plt.ylabel("log(P_t) - log(P_t-1)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/log_returns.png", dpi=150)
plt.close()

# %% -------------------------------------------------------------
# 2. Bayesian Change Point Model (PyMC)
# -------------------------------------------------------------
# NOTE: Full 9,000+ point daily series is heavy for a naive single
# change point model geared toward one structural break. For a
# course-project-scale analysis, either:
#   (a) run the single change point model on the full log_return
#       series (it will find the single MOST dominant regime shift), or
#   (b) restrict the window to a specific period of interest
#       (e.g. around 2014-2016 oil price collapse) to find a more
#       locally meaningful change point.
# Below shows (a); swap in a filtered df for (b).

y = df["log_return"].values
n = len(y)

with pm.Model() as model:
    # Switch point: discrete uniform prior over every index in the series
    tau = pm.DiscreteUniform("tau", lower=0, upper=n - 1)

    # Before / after parameters for the mean log-return
    mu_1 = pm.Normal("mu_1", mu=0, sigma=0.05)
    mu_2 = pm.Normal("mu_2", mu=0, sigma=0.05)

    # Shared (or separate, if you want to test volatility shifts too) sigma
    sigma = pm.HalfNormal("sigma", sigma=0.05)

    idx = np.arange(n)
    mu = pm.math.switch(tau >= idx, mu_1, mu_2)

    likelihood = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        target_accept=0.9,
        return_inferencedata=True,
    )

# %% -------------------------------------------------------------
# 3. Interpret the Model Output
# -------------------------------------------------------------
summary = pm.summary(trace, var_names=["tau", "mu_1", "mu_2", "sigma"])
print(summary)
# Check r_hat ~1.0 for all parameters before trusting the result

az.plot_trace(trace, var_names=["tau", "mu_1", "mu_2", "sigma"])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/trace_plot.png", dpi=150)
plt.close()

# Posterior distribution of tau -> map back to a calendar date
tau_samples = trace.posterior["tau"].values.flatten()
tau_mode = int(np.round(np.median(tau_samples)))
change_date = df["Date"].iloc[tau_mode]

plt.figure(figsize=(10, 4))
plt.hist(tau_samples, bins=50, color="#2e7d32")
plt.axvline(tau_mode, color="black", linestyle="--", label=f"median tau -> {change_date.date()}")
plt.title("Posterior Distribution of Change Point (tau)")
plt.xlabel("Index in series")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tau_posterior.png", dpi=150)
plt.close()

# Quantify the shift in mean log-return before/after
mu_1_samples = trace.posterior["mu_1"].values.flatten()
mu_2_samples = trace.posterior["mu_2"].values.flatten()

plt.figure(figsize=(10, 4))
plt.hist(mu_1_samples, bins=50, alpha=0.6, label="mu_1 (before)", color="#1565c0")
plt.hist(mu_2_samples, bins=50, alpha=0.6, label="mu_2 (after)", color="#c62828")
plt.title("Posterior Distributions: Mean Log-Return Before vs After")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/before_after_means.png", dpi=150)
plt.close()

pct_change_in_mean_return = float(
    (np.mean(mu_2_samples) - np.mean(mu_1_samples)) / abs(np.mean(mu_1_samples)) * 100
    if np.mean(mu_1_samples) != 0 else np.nan
)

prob_increase = float(np.mean(mu_2_samples > mu_1_samples))

print(f"Detected change point: {change_date.date()} (index {tau_mode})")
print(f"Mean log-return before: {np.mean(mu_1_samples):.5f}")
print(f"Mean log-return after:  {np.mean(mu_2_samples):.5f}")
print(f"P(mu_2 > mu_1): {prob_increase:.3f}")

# %% -------------------------------------------------------------
# 4. Associate the Change Point with Researched Events
# -------------------------------------------------------------
events = pd.read_csv(f"{DATA_DIR}/brent_oil_events.csv")
events["Date"] = pd.to_datetime(events["Date"])
events["days_from_change_point"] = (events["Date"] - change_date).dt.days

# Nearest events within a 90-day window (adjust as needed)
nearby_events = events.loc[events["days_from_change_point"].abs() <= 90].copy()
nearby_events = nearby_events.sort_values("days_from_change_point", key=abs)
print("Nearby candidate events:")
print(nearby_events[["Date", "days_from_change_point"]].to_string(index=False))

# %% -------------------------------------------------------------
# 5. Export results for the dashboard (Task 3 backend reads this)
# -------------------------------------------------------------
avg_price_before = float(df.loc[df.index < tau_mode, "Price"].mean())
avg_price_after = float(df.loc[df.index >= tau_mode, "Price"].mean())
price_pct_change = float((avg_price_after - avg_price_before) / avg_price_before * 100)

model_output = {
    "change_point_date": str(change_date.date()),
    "change_point_index": tau_mode,
    "mean_log_return_before": float(np.mean(mu_1_samples)),
    "mean_log_return_after": float(np.mean(mu_2_samples)),
    "prob_increase_after": prob_increase,
    "avg_price_before_usd": round(avg_price_before, 2),
    "avg_price_after_usd": round(avg_price_after, 2),
    "price_pct_change": round(price_pct_change, 2),
    "r_hat_max": float(summary["r_hat"].max()),
    "nearby_events": nearby_events[["Date", "days_from_change_point"]].assign(
        Date=lambda d: d["Date"].astype(str)
    ).to_dict(orient="records"),
}

with open(f"{DATA_DIR}/model_output.json", "w") as f:
    json.dump(model_output, f, indent=2)

print("\nSaved model_output.json for the dashboard backend.")
print(json.dumps(model_output, indent=2))
