"""Safe DataFrame utility functions.

These functions create copies for display and provide dtype-safe
operations. They NEVER modify the original DataFrame.
"""

from typing import Optional
import numpy as np
import pandas as pd
from core.logging import logger


def safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Create a safe deep copy of a DataFrame.

    Args:
        df: Source DataFrame.

    Returns:
        A new DataFrame copy.
    """
    return df.copy()


def make_arrow_safe_preview(
    df: pd.DataFrame, max_rows: Optional[int] = None
) -> pd.DataFrame:
    """Create a display-safe copy of a DataFrame for Streamlit/PyArrow.

    Object columns with mixed Python types (str, datetime, int) cause
    ArrowTypeError. This function converts such columns to string
    representations in a COPY — the original DataFrame is untouched.

    Args:
        df: Source DataFrame (not modified).
        max_rows: Optional row limit for the preview.

    Returns:
        A new DataFrame safe for st.dataframe().
    """
    preview = df.head(max_rows) if max_rows else df.copy()
    preview = preview.copy()

    for col in preview.columns:
        dtype = preview[col].dtype
        if dtype == object:
            # Object columns may contain mixed types — convert to string
            try:
                preview[col] = preview[col].apply(
                    lambda x: str(x) if x is not None and not _is_na(x) else None
                )
            except Exception:
                logger.debug("Could not convert column '{}' for preview", col)
                preview[col] = preview[col].astype(str)

    return preview


def _is_na(value: object) -> bool:
    """Check if a value is NA/NaN safely."""
    try:
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        return pd.isna(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def safe_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Get column names that are safely numeric (int or float dtypes).

    Args:
        df: Source DataFrame.

    Returns:
        List of column names with numeric dtypes.
    """
    result: list[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            result.append(str(col))
    return result


def safe_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Get column names that are datetime dtypes.

    Args:
        df: Source DataFrame.

    Returns:
        List of column names with datetime dtypes.
    """
    result: list[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result.append(str(col))
    return result


def safe_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Get column names that are categorical or object dtypes (non-numeric, non-datetime).

    Args:
        df: Source DataFrame.

    Returns:
        List of column names with object/category dtypes.
    """
    result: list[str] = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == object or isinstance(dtype, pd.CategoricalDtype):
            result.append(str(col))
    return result


def get_column_dtype(df: pd.DataFrame, column: str) -> str:
    """Get the dtype of a column as a string.

    Args:
        df: Source DataFrame.
        column: Column name.

    Returns:
        String representation of the dtype.
    """
    return str(df[column].dtype)


def is_numeric_column(df: pd.DataFrame, column: str) -> bool:
    """Check if a column has a numeric dtype."""
    return bool(pd.api.types.is_numeric_dtype(df[column]))


def is_datetime_column(df: pd.DataFrame, column: str) -> bool:
    """Check if a column has a datetime dtype."""
    return bool(pd.api.types.is_datetime64_any_dtype(df[column]))


def is_categorical_column(df: pd.DataFrame, column: str) -> bool:
    """Check if a column has object or category dtype."""
    dtype = df[column].dtype
    return dtype == object or isinstance(dtype, pd.CategoricalDtype)

