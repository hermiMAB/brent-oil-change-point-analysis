"""
date_utils.py

Utilities for parsing the mixed date formats found in the Brent oil price
dataset (e.g. '20-May-87' and 'Oct 27, 2022'), and validating the result.
"""

from typing import Union

import pandas as pd


# Known date formats observed in the raw source file, tried in order.
KNOWN_DATE_FORMATS = ("%d-%b-%y", "%b %d, %Y")


def parse_single_date(date_str: Union[str, float]) -> pd.Timestamp:
    """
    Parse a single date string that may be in one of several known formats.

    Parameters
    ----------
    date_str : str
        A date string such as '20-May-87' or '"Oct 27, 2022"'.

    Returns
    -------
    pd.Timestamp
        The parsed timestamp.

    Raises
    ------
    ValueError
        If the value cannot be parsed as a date by any known format or by
        pandas' general-purpose inference.
    """
    cleaned = str(date_str).strip().strip('"')

    for fmt in KNOWN_DATE_FORMATS:
        try:
            return pd.to_datetime(cleaned, format=fmt)
        except ValueError:
            continue

    try:
        return pd.to_datetime(cleaned)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not parse date value '{date_str}'. Expected formats: "
            f"{KNOWN_DATE_FORMATS} or an ISO-8601-compatible string."
        ) from exc


def parse_date_column(
    df: pd.DataFrame, column: str = "Date", sort: bool = True
) -> pd.DataFrame:
    """
    Parse a dataframe's date column (handling mixed formats) and optionally
    sort the dataframe chronologically.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the date column to parse. Not modified in place.
    column : str, default 'Date'
        Name of the column containing date strings.
    sort : bool, default True
        Whether to sort the returned dataframe by the parsed date, ascending.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with `column` converted to datetime64, and the index
        reset if sorted.

    Raises
    ------
    KeyError
        If `column` is not present in `df`.
    ValueError
        If one or more values in `column` cannot be parsed as dates.
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in dataframe. "
            f"Available columns: {list(df.columns)}."
        )

    result = df.copy()

    try:
        result[column] = result[column].apply(parse_single_date)
    except ValueError as exc:
        raise ValueError(
            f"Failed to parse one or more values in column '{column}'. {exc}"
        ) from exc

    if sort:
        result = result.sort_values(column).reset_index(drop=True)

    return result
