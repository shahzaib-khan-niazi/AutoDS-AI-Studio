"""Structure detection utility functions.

Deterministic helpers for analyzing DataFrame structure signals.
"""

from typing import Optional
import numpy as np
import pandas as pd

from utils.dataframe import safe_numeric_columns, safe_datetime_columns


def sample_dataframe(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Sample rows from a DataFrame for analysis.

    Args:
        df: Source DataFrame.
        n: Maximum number of rows to sample.

    Returns:
        Sampled DataFrame (or full DataFrame if smaller than n).
    """
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=42).copy()


def column_type_summary(df: pd.DataFrame) -> dict[str, int]:
    """Summarize the count of columns by dtype category.

    Returns:
        Dict with keys like 'numeric', 'categorical', 'datetime', 'boolean'.
    """
    summary: dict[str, int] = {
        "numeric": 0,
        "categorical": 0,
        "datetime": 0,
        "boolean": 0,
    }
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            summary["numeric"] += 1
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            summary["datetime"] += 1
        elif pd.api.types.is_bool_dtype(dtype):
            summary["boolean"] += 1
        else:
            summary["categorical"] += 1
    return summary


def unique_ratio(df: pd.DataFrame, column: str) -> float:
    """Ratio of unique values to total non-null values.

    Args:
        df: Source DataFrame.
        column: Column name.

    Returns:
        Float from 0.0 to 1.0.
    """
    non_null = df[column].dropna()
    if len(non_null) == 0:
        return 0.0
    return float(non_null.nunique() / len(non_null))


def numeric_ratio(df: pd.DataFrame, column: str) -> float:
    """Ratio of values in an object column that can be parsed as numeric.

    Args:
        df: Source DataFrame.
        column: Column name.

    Returns:
        Float from 0.0 to 1.0.
    """
    non_null = df[column].dropna()
    if len(non_null) == 0:
        return 0.0

    numeric_count = 0
    for val in non_null.head(200):
        try:
            float(val)
            numeric_count += 1
        except (ValueError, TypeError):
            pass

    sample_size = min(len(non_null), 200)
    return numeric_count / sample_size if sample_size > 0 else 0.0


def datetime_ratio(df: pd.DataFrame, column: str) -> float:
    """Ratio of values in an object column that look like dates.

    Args:
        df: Source DataFrame.
        column: Column name.

    Returns:
        Float from 0.0 to 1.0.
    """
    non_null = df[column].dropna()
    if len(non_null) == 0:
        return 0.0

    date_count = 0
    for val in non_null.head(100):
        try:
            pd.to_datetime(val)
            date_count += 1
        except (ValueError, TypeError, OverflowError):
            pass

    sample_size = min(len(non_null), 100)
    return date_count / sample_size if sample_size > 0 else 0.0


def repeated_value_pattern(df: pd.DataFrame, column: str, threshold: float = 0.5) -> bool:
    """Check if a column has a high ratio of repeated values.

    Args:
        df: Source DataFrame.
        column: Column name.
        threshold: Minimum ratio of most-common-value frequency to count.

    Returns:
        True if the column shows a repeated-value pattern.
    """
    non_null = df[column].dropna()
    if len(non_null) == 0:
        return False
    top_freq = non_null.value_counts().iloc[0] if len(non_null.value_counts()) > 0 else 0
    return (top_freq / len(non_null)) >= threshold


def header_likelihood(df: pd.DataFrame) -> float:
    """Estimate the likelihood that the first row is actually a header/metadata row.

    Heuristics:
    - First row values are all strings
    - First row values differ significantly in type from subsequent rows
    - First row values look like labels

    Returns:
        Float from 0.0 to 1.0 indicating likelihood.
    """
    if len(df) < 2:
        return 0.0

    first_row = df.iloc[0]
    second_row = df.iloc[1]
    score = 0.0
    signals = 0

    for col in df.columns:
        first_val = first_row[col]
        second_val = second_row[col]
        signals += 1

        if first_val is None or (isinstance(first_val, float) and np.isnan(first_val)):
            continue

        # First row is string, second row is numeric
        first_is_str = isinstance(first_val, str)
        second_is_num = False
        if second_val is not None:
            try:
                if not isinstance(second_val, str):
                    float(second_val)
                    second_is_num = True
                else:
                    float(second_val)
                    second_is_num = True
            except (ValueError, TypeError):
                pass

        if first_is_str and second_is_num:
            score += 1

        # First row looks like a label (short string, not a number)
        if first_is_str and len(str(first_val)) < 30:
            try:
                float(first_val)
            except (ValueError, TypeError):
                score += 0.3

    return min(1.0, score / max(signals, 1))


def column_name_similarity(columns: list[str]) -> float:
    """Measure how similar column names are to each other.

    High similarity suggests columns represent categories rather than
    independent variables (e.g., 'Jan', 'Feb', 'Mar' or 'Region_A', 'Region_B').

    Returns:
        Float from 0.0 to 1.0.
    """
    if len(columns) < 3:
        return 0.0

    col_strs = [str(c).lower().strip() for c in columns]

    # Check for common prefix/suffix patterns
    prefix_groups: dict[str, int] = {}
    for c in col_strs:
        # Extract prefix (first word or up to underscore)
        parts = c.replace("-", "_").split("_")
        if len(parts) > 1:
            prefix = parts[0]
            prefix_groups[prefix] = prefix_groups.get(prefix, 0) + 1

    # If many columns share a prefix, they're likely categories
    if prefix_groups:
        max_group = max(prefix_groups.values())
        if max_group >= 3 and max_group / len(columns) > 0.4:
            return max_group / len(columns)

    # Check for similar lengths (category names tend to be similar length)
    lengths = [len(c) for c in col_strs]
    if len(lengths) > 2:
        avg_len = sum(lengths) / len(lengths)
        if avg_len > 0:
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
            # Low variance = similar lengths
            if variance < 4 and avg_len < 20:
                return 0.3

    return 0.0


def uniform_dtype_ratio(df: pd.DataFrame) -> float:
    """Ratio of columns that share the same dtype as the majority dtype.

    High uniformity suggests columns represent the same kind of data
    (e.g., all numeric values in a pivot table).

    Returns:
        Float from 0.0 to 1.0.
    """
    if len(df.columns) == 0:
        return 0.0

    dtype_counts: dict[str, int] = {}
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

    max_count = max(dtype_counts.values()) if dtype_counts else 0
    return max_count / len(df.columns)
