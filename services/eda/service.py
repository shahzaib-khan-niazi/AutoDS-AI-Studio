"""Exploratory Data Analysis (EDA) Service.

Computes correlation matrices, distribution summaries, and visual statistics.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd

from core.logging import logger
from utils.dataframe import safe_numeric_columns, safe_categorical_columns


class EDAService:
    """Service for exploratory data analysis."""

    @classmethod
    def get_correlation_matrix(cls, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute Pearson correlation matrix for numeric columns.

        Args:
            df: Source DataFrame.

        Returns:
            Correlation DataFrame or None if fewer than 2 numeric columns.
        """
        num_cols = safe_numeric_columns(df)
        if len(num_cols) < 2:
            return None

        clean_num = df[num_cols].dropna()
        if len(clean_num) < 3:
            return None

        try:
            return clean_num.corr(method="pearson").round(3)
        except Exception as e:
            logger.warning("Correlation computation failed: {}", str(e))
            return None

    @classmethod
    def get_column_distribution(cls, df: pd.DataFrame, column: str) -> dict[str, Any]:
        """Get distribution summary for a column (numeric histogram or categorical frequency)."""
        if column not in df.columns:
            return {}

        series = df[column].dropna()
        if len(series) == 0:
            return {}

        if pd.api.types.is_numeric_dtype(series):
            return {
                "type": "numeric",
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": series.mean(),
                "median": series.median(),
                "values": series.tolist()[:1000],  # Sample for plot
            }
        else:
            val_counts = series.value_counts().head(20)
            return {
                "type": "categorical",
                "categories": [str(c) for c in val_counts.index],
                "counts": [int(c) for c in val_counts.values],
            }
