"""
eda_plots.py

Reusable plotting functions for Brent oil price EDA: trend, log returns,
and rolling volatility. Each function saves a PNG and returns the Figure
so it can also be displayed inline (e.g. in a notebook).
"""

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def _validate_columns(df: pd.DataFrame, columns: list) -> None:
    """Raise a clear error if any required column is missing from df."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s) {missing} for plotting. "
            f"Available columns: {list(df.columns)}."
        )


def plot_price_trend(
    df: pd.DataFrame,
    date_col: str = "Date",
    price_col: str = "Price",
    output_path: Union[str, Path] = "outputs/raw_price_trend.png",
    title: str = "Brent Oil Price",
) -> Figure:
    """
    Plot the raw price series over time and save it to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing `date_col` and `price_col`.
    date_col : str, default 'Date'
    price_col : str, default 'Price'
    output_path : str or Path
        Where to save the PNG. Parent directories are created if needed.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    KeyError
        If required columns are missing from `df`.
    ValueError
        If `df` is empty.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty dataframe.")
    _validate_columns(df, [date_col, price_col])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[date_col], df[price_col], linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("USD per barrel")
    fig.tight_layout()
    fig.savefig(out, dpi=150)

    return fig


def plot_log_returns(
    df: pd.DataFrame,
    date_col: str = "Date",
    return_col: str = "log_return",
    output_path: Union[str, Path] = "outputs/log_returns.png",
    title: str = "Brent Oil Log Returns (volatility clustering check)",
) -> Figure:
    """
    Plot log returns over time to visualize volatility clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing `date_col` and `return_col`.
    date_col : str, default 'Date'
    return_col : str, default 'log_return'
    output_path : str or Path
    title : str

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    KeyError
        If required columns are missing from `df`.
    ValueError
        If `df` is empty.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty dataframe.")
    _validate_columns(df, [date_col, return_col])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[date_col], df[return_col], linewidth=0.5, color="darkorange")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    fig.tight_layout()
    fig.savefig(out, dpi=150)

    return fig


def plot_rolling_volatility(
    df: pd.DataFrame,
    date_col: str = "Date",
    return_col: str = "log_return",
    window: int = 30,
    output_path: Union[str, Path] = "outputs/rolling_volatility.png",
    title: str = "Rolling Volatility of Log Returns",
) -> Figure:
    """
    Compute and plot rolling standard deviation of log returns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing `date_col` and `return_col`.
    date_col : str, default 'Date'
    return_col : str, default 'log_return'
    window : int, default 30
        Rolling window size in trading days. Must be a positive integer.
    output_path : str or Path
    title : str

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    KeyError
        If required columns are missing from `df`.
    ValueError
        If `df` is empty or `window` is not a positive integer.
    """
    if df.empty:
        raise ValueError("Cannot plot an empty dataframe.")
    _validate_columns(df, [date_col, return_col])
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"'window' must be a positive integer, got {window!r}.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rolling_vol = df[return_col].rolling(window=window).std()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[date_col], rolling_vol, linewidth=0.8, color="crimson")
    ax.set_title(f"{title} ({window}-Day Window)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Std Dev")
    fig.tight_layout()
    fig.savefig(out, dpi=150)

    return fig
