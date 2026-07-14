"""
stationarity.py

Utilities for computing log returns and running stationarity diagnostics
(Augmented Dickey-Fuller test) on a Brent oil price series.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass
class ADFResult:
    """Container for a readable Augmented Dickey-Fuller test result."""
    label: str
    statistic: float
    p_value: float
    critical_values: Dict[str, float]

    @property
    def is_stationary(self) -> bool:
        """True if the null hypothesis of a unit root is rejected at alpha=0.05."""
        return self.p_value < 0.05

    def is_stationary_at(self, alpha: float) -> bool:
        """Return True if the null hypothesis is rejected at the given alpha."""
        return self.p_value < alpha

    def __str__(self) -> str:
        conclusion = "Stationary" if self.is_stationary else "Non-stationary"
        return (
            f"ADF Test - {self.label}\n"
            f"  Statistic: {self.statistic:.4f}\n"
            f"  p-value:   {self.p_value:.4f}\n"
            f"  Critical values: {self.critical_values}\n"
            f"  Conclusion: {conclusion}"
        )


def compute_log_returns(price: pd.Series) -> pd.Series:
    """
    Compute log returns: log(P_t) - log(P_{t-1}).

    Parameters
    ----------
    price : pd.Series
        Series of price values. Must contain only positive numbers.

    Returns
    -------
    pd.Series
        Log returns, with the first (NaN) observation dropped.

    Raises
    ------
    ValueError
        If `price` contains non-positive values (log is undefined) or is
        empty.
    """
    if price.empty:
        raise ValueError("Price series is empty; cannot compute log returns.")

    if (price <= 0).any():
        raise ValueError(
            "Price series contains non-positive values; log returns are "
            "undefined for prices <= 0."
        )

    log_price = np.log(price)
    return log_price.diff().dropna()


def run_adf_test(series: pd.Series, label: str = "series") -> ADFResult:
    """
    Run the Augmented Dickey-Fuller stationarity test on a series.

    Parameters
    ----------
    series : pd.Series
        The time series to test (e.g. raw prices or log returns).
    label : str, default 'series'
        A human-readable label used in the returned result's string form.

    Returns
    -------
    ADFResult
        Structured result including the test statistic, p-value, critical
        values, and a `.is_stationary` convenience property.

    Raises
    ------
    ValueError
        If `series` is empty or has fewer than 2 non-null observations
        (the ADF test cannot run on such input).
    """
    clean = series.dropna()

    if len(clean) < 2:
        raise ValueError(
            f"Series '{label}' has fewer than 2 non-null observations "
            f"({len(clean)} found); the ADF test requires more data."
        )

    stat, p_value, _, _, critical_values, _ = adfuller(clean)

    return ADFResult(
        label=label,
        statistic=stat,
        p_value=p_value,
        critical_values=critical_values,
    )
