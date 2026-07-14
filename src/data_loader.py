"""
data_loader.py

Utilities for loading and validating the raw Brent oil price dataset.
"""

from pathlib import Path
from typing import Union

import pandas as pd


REQUIRED_COLUMNS = {"Date", "Price"}


def load_price_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load the raw Brent oil price CSV and perform basic structural validation.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file containing at least 'Date' and 'Price' columns.

    Returns
    -------
    pd.DataFrame
        The raw, unparsed dataframe (Date is still a string at this stage;
        use `date_utils.parse_date_column` to convert it).

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file is empty, unreadable as CSV, or missing required columns.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"Price data file not found at '{path}'. "
            f"Check the path and make sure the file has been placed in the "
            f"expected data/ directory."
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The file at '{path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Could not parse '{path}' as CSV. Confirm it is a valid, "
            f"comma-separated file."
        ) from exc

    if df.empty:
        raise ValueError(f"'{path}' was read successfully but contains no rows.")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"'{path}' is missing required column(s): {sorted(missing_columns)}. "
            f"Found columns: {list(df.columns)}."
        )

    return df


def load_events_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load the compiled key-events CSV and validate its expected structure.

    Parameters
    ----------
    filepath : str or Path
        Path to the events CSV file.

    Returns
    -------
    pd.DataFrame
        The events dataframe.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or missing required columns, or has fewer than
        10 events (the challenge's minimum requirement).
    """
    path = Path(filepath)
    required = {
        "event_id", "date", "event_name", "category",
        "description", "expected_price_direction",
    }

    if not path.exists():
        raise FileNotFoundError(f"Events data file not found at '{path}'.")

    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read '{path}' as a valid CSV.") from exc

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"'{path}' is missing required column(s): {sorted(missing)}."
        )

    if len(df) < 10:
        raise ValueError(
            f"Expected at least 10 events, found {len(df)} in '{path}'."
        )

    return df
