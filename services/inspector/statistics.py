"""Statistics calculation for dataset inspection.

Handles numeric, categorical, and datetime columns separately.
Does NOT use df.describe(datetime_is_numeric=True).
"""

import numpy as np
import pandas as pd

from core.logging import logger
from models.inspection import NumericStats, CategoricalStats, DatetimeStats
from utils.dataframe import safe_numeric_columns, safe_datetime_columns, safe_categorical_columns


def compute_numeric_stats(df: pd.DataFrame) -> list[NumericStats]:
    """Compute descriptive statistics for numeric columns.

    Args:
        df: Source DataFrame.

    Returns:
        List of NumericStats, one per numeric column.
    """
    results: list[NumericStats] = []
    numeric_cols = safe_numeric_columns(df)

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            results.append(NumericStats(column=col, count=0))
            continue

        try:
            count = int(len(series))
            mean_val = float(series.mean())
            median_val = float(series.median())
            std_val = float(series.std()) if count > 1 else 0.0
            min_val = float(series.min())
            max_val = float(series.max())

            # Skewness — requires at least 3 values
            skew_val = 0.0
            if count >= 3:
                try:
                    skew_val = float(series.skew())
                    if np.isnan(skew_val) or np.isinf(skew_val):
                        skew_val = 0.0
                except (ValueError, TypeError):
                    skew_val = 0.0

            # Handle NaN/inf in computed stats
            for val_name, val in [("mean", mean_val), ("std", std_val)]:
                if np.isnan(val) or np.isinf(val):
                    if val_name == "mean":
                        mean_val = 0.0
                    elif val_name == "std":
                        std_val = 0.0

            results.append(NumericStats(
                column=col,
                count=count,
                mean=mean_val,
                median=median_val,
                std=std_val,
                min_value=min_val,
                max_value=max_val,
                skewness=skew_val,
            ))
        except Exception as e:
            logger.warning("Error computing stats for column '{}': {}", col, str(e))
            results.append(NumericStats(column=col, count=int(len(series))))

    return results


def compute_categorical_stats(df: pd.DataFrame) -> list[CategoricalStats]:
    """Compute statistics for categorical/object columns.

    Args:
        df: Source DataFrame.

    Returns:
        List of CategoricalStats, one per categorical column.
    """
    results: list[CategoricalStats] = []
    cat_cols = safe_categorical_columns(df)

    for col in cat_cols:
        series = df[col].dropna()
        count = int(len(series))
        unique = int(series.nunique())

        top_value = ""
        top_freq = 0
        if count > 0:
            try:
                value_counts = series.value_counts()
                if len(value_counts) > 0:
                    top_value = str(value_counts.index[0])
                    top_freq = int(value_counts.iloc[0])
            except Exception:
                pass

        results.append(CategoricalStats(
            column=col,
            count=count,
            unique=unique,
            top=top_value,
            top_frequency=top_freq,
        ))

    return results


def compute_datetime_stats(df: pd.DataFrame) -> list[DatetimeStats]:
    """Compute statistics for datetime columns.

    Args:
        df: Source DataFrame.

    Returns:
        List of DatetimeStats, one per datetime column.
    """
    results: list[DatetimeStats] = []
    dt_cols = safe_datetime_columns(df)

    for col in dt_cols:
        series = df[col].dropna()
        count = int(len(series))

        min_date = ""
        max_date = ""
        if count > 0:
            try:
                min_date = str(series.min())
                max_date = str(series.max())
            except Exception:
                pass

        results.append(DatetimeStats(
            column=col,
            count=count,
            min_date=min_date,
            max_date=max_date,
        ))

    return results
