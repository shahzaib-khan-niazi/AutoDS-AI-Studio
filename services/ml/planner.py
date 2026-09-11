"""ML Planner — detects task type (classification vs regression) and suitable targets."""

from typing import Optional
import numpy as np
import pandas as pd

from models.ml import MLTaskType
from utils.dataframe import is_numeric_column, safe_numeric_columns


class MLPlanner:
    """Helper for planning ML tasks on clean datasets."""

    @classmethod
    def detect_task_type(cls, df: pd.DataFrame, target_column: str) -> MLTaskType:
        """Automatically detect if the target column represents classification or regression.

        Args:
            df: Clean DataFrame.
            target_column: Name of the target feature.

        Returns:
            MLTaskType.
        """
        if target_column not in df.columns:
            return MLTaskType.REGRESSION

        series = df[target_column].dropna()
        n_unique = series.nunique()

        # Binary
        if n_unique == 2:
            return MLTaskType.BINARY_CLASSIFICATION

        # Non-numeric is always classification
        if not is_numeric_column(df, target_column):
            return MLTaskType.MULTICLASS_CLASSIFICATION

        # If numeric with few unique discrete values (< 10 and < 5% of row count)
        if n_unique <= 10 and n_unique / max(len(series), 1) < 0.05:
            return MLTaskType.MULTICLASS_CLASSIFICATION

        return MLTaskType.REGRESSION

    @classmethod
    def suggest_target_columns(cls, df: pd.DataFrame) -> list[str]:
        """Suggest candidate target columns from the dataset."""
        candidates: list[str] = []
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            # Avoid high-cardinality ID columns
            if series.nunique() == len(series) and len(series) > 20:
                continue
            candidates.append(col)

        # Put columns with target-like names at the front
        priority_keywords = ["target", "label", "price", "churn", "salary", "revenue", "class", "outcome", "survived", "status", "score"]
        candidates.sort(
            key=lambda c: any(kw in c.lower() for kw in priority_keywords),
            reverse=True,
        )
        return candidates
