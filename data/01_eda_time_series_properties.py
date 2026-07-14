# %% [markdown]
# # Task 1 - Understanding the Model and Data
# Trend, stationarity, and volatility analysis of Brent oil prices.
#
# This notebook uses reusable, tested modules from `src/` for data loading,
# date parsing, stationarity testing, and plotting, rather than inline logic.

# %%
import sys
sys.path.append("..")  # allow importing from src/ when run inside notebooks/

from src.data_loader import load_price_data
from src.date_utils import parse_date_column
from src.stationarity import compute_log_returns, run_adf_test
from src.eda_plots import plot_price_trend, plot_log_returns, plot_rolling_volatility

# %% [markdown]
# ## 1. Load and prepare data
# `load_price_data` validates the file exists and has the expected columns;
# `parse_date_column` handles the mixed date formats present in the source
# file and sorts chronologically.

# %%
try:
    df = load_price_data("../data/BrentOilPrices.csv")
    df = parse_date_column(df, column="Date", sort=True)
except (FileNotFoundError, ValueError, KeyError) as e:
    print(f"Failed to load/parse price data: {e}")
    raise

print(df.head())
print(df.info())
print("Missing values:\n", df.isna().sum())

# %% [markdown]
# ## 2. Trend analysis - raw price plot

# %%
plot_price_trend(
    df,
    output_path="../outputs/raw_price_trend.png",
    title="Brent Oil Price (1987-2022)",
)

# %% [markdown]
# ## 3. Log returns - variance-stabilizing transform

# %%
try:
    log_returns = compute_log_returns(df["Price"])
except ValueError as e:
    print(f"Failed to compute log returns: {e}")
    raise

df = df.iloc[1:].copy()  # align with log_returns (first obs has no return)
df["log_return"] = log_returns.values

plot_log_returns(df, output_path="../outputs/log_returns.png")

# %% [markdown]
# ## 4. Stationarity testing (Augmented Dickey-Fuller)

# %%
adf_price = run_adf_test(df["Price"], label="Raw Price")
adf_return = run_adf_test(df["log_return"], label="Log Returns")

print(adf_price)
print()
print(adf_return)

# %% [markdown]
# ## 5. Volatility patterns - rolling std as a proxy

# %%
plot_rolling_volatility(
    df, window=30, output_path="../outputs/rolling_volatility.png"
)

# %% [markdown]
# ## 6. Summary printout for the report

# %%
print("\n--- Summary for report ---")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Raw price ADF p-value: {adf_price.p_value:.4f} -> "
      f"{'stationary' if adf_price.is_stationary else 'non-stationary'}")
print(f"Log return ADF p-value: {adf_return.p_value:.4f} -> "
      f"{'stationary' if adf_return.is_stationary else 'non-stationary'}")
