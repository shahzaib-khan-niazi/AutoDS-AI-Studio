"""Unit tests for DatasetProfiler (services/profiler/service.py).

Tests deterministic profile calculation, quality scoring, column detection,
and NaN/Inf handling.
"""

import json
import numpy as np
import pandas as pd
import pytest

from services.profiler.service import DatasetProfiler
from core.schemas.dataset_profile import DatasetProfile


class TestDatasetProfiler:
    """Unit tests for DatasetProfiler class."""

    def test_empty_dataframe_profile(self) -> None:
        """Profiler should handle 0-row and 0-column DataFrames gracefully without crashing."""
        df_empty = pd.DataFrame()
        profile = DatasetProfiler.profile(df_empty, filename="empty.csv")

        assert isinstance(profile, DatasetProfile)
        assert profile.row_count == 0
        assert profile.column_count == 0
        assert profile.quality_score == 0.0
        assert profile.quality_status == "Critical"
        assert len(profile.detected_issues) == 1
        assert profile.detected_issues[0].issue_type == "empty_dataset"

    def test_normal_dataframe_profile(self) -> None:
        """Profiler should produce accurate metrics for a standard DataFrame."""
        df = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            "city": ["NY", "LA", "NY", "SF", "LA"],
        })
        profile = DatasetProfiler.profile(df, filename="test.csv")

        assert profile.filename == "test.csv"
        assert profile.row_count == 5
        assert profile.column_count == 3
        assert profile.duplicate_row_count == 0
        assert profile.duplicate_row_percentage == 0.0
        assert profile.quality_score >= 80.0
        assert len(profile.columns) == 3

    def test_missing_values_detection(self) -> None:
        """Profiler should accurately count and percentage missing values."""
        df = pd.DataFrame({
            "a": [1, None, 3, None, 5],
            "b": ["x", "y", "z", "w", "v"],
        })
        profile = DatasetProfiler.profile(df)

        assert "a" in profile.missing_columns
        assert "b" not in profile.missing_columns

        col_a = next(c for c in profile.columns if c.name == "a")
        assert col_a.missing_count == 2
        assert col_a.missing_percentage == 40.0
        assert col_a.non_null_count == 3

    def test_duplicate_rows_detection(self) -> None:
        """Profiler should detect duplicate rows accurately."""
        df = pd.DataFrame({
            "id": [1, 1, 2, 2],
            "val": ["A", "A", "B", "B"],
        })
        profile = DatasetProfiler.profile(df)

        assert profile.duplicate_row_count == 2
        assert profile.duplicate_row_percentage == 50.0

    def test_constant_columns_detection(self) -> None:
        """Profiler should flag columns with nunique <= 1 as constant."""
        df = pd.DataFrame({
            "const_col": ["fixed", "fixed", "fixed"],
            "var_col": [1, 2, 3],
        })
        profile = DatasetProfiler.profile(df)

        assert "const_col" in profile.constant_columns
        assert "var_col" not in profile.constant_columns

        col_const = next(c for c in profile.columns if c.name == "const_col")
        assert col_const.is_constant is True

    def test_numeric_statistics(self) -> None:
        """Profiler should calculate min, max, mean, median, std, q1, q3."""
        df = pd.DataFrame({
            "scores": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        profile = DatasetProfiler.profile(df)
        col = profile.columns[0]

        assert col.is_numeric is True
        assert col.min_val == 10.0
        assert col.max_val == 50.0
        assert col.mean_val == 30.0
        assert col.median_val == 30.0
        assert col.q1_val == 20.0
        assert col.q3_val == 40.0

    def test_categorical_statistics(self) -> None:
        """Profiler should calculate top value counts for categorical columns."""
        df = pd.DataFrame({
            "color": ["red", "red", "red", "blue", "green"],
        })
        profile = DatasetProfiler.profile(df)
        col = profile.columns[0]

        assert col.is_categorical is True
        assert col.top_values is not None
        assert col.top_values["red"] == 3
        assert col.top_values["blue"] == 1

    def test_datetime_statistics(self) -> None:
        """Profiler should parse datetime columns and extract min/max dates."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"]),
        })
        profile = DatasetProfiler.profile(df)
        col = profile.columns[0]

        assert col.is_datetime is True
        assert "2024-01-01" in str(col.min_date)
        assert "2024-12-31" in str(col.max_date)

    def test_possible_id_detection(self) -> None:
        """Profiler should flag ID-like column names or highly unique identifier columns."""
        df = pd.DataFrame({
            "user_id": [101, 102, 103, 104, 105],
            "uuid_code": ["u1", "u2", "u3", "u4", "u5"],
            "age": [20, 30, 40, 50, 60],
        })
        profile = DatasetProfiler.profile(df)

        assert "user_id" in profile.possible_id_columns
        assert "uuid_code" in profile.possible_id_columns
        assert "age" not in profile.possible_id_columns

    def test_high_cardinality_detection(self) -> None:
        """Profiler should flag string/object columns with very high unique values."""
        df = pd.DataFrame({
            "unique_text": [f"text_{i}" for i in range(100)],
        })
        profile = DatasetProfiler.profile(df)

        assert "unique_text" in profile.high_cardinality_columns
        col = profile.columns[0]
        assert col.is_high_cardinality is True

    def test_quality_score_range(self) -> None:
        """Quality score should always fall between 0.0 and 100.0."""
        # Perfect dataset
        df_perfect = pd.DataFrame({"a": range(10), "b": range(10, 20)})
        prof_perfect = DatasetProfiler.profile(df_perfect)
        assert prof_perfect.quality_score >= 90.0

        # Highly problematic dataset
        df_bad = pd.DataFrame({
            "all_null": [None] * 50,
            "const": [1] * 50,
            "dup": ["x"] * 50,
        })
        prof_bad = DatasetProfiler.profile(df_bad)
        assert 0.0 <= prof_bad.quality_score <= 100.0
        assert prof_bad.quality_score < prof_perfect.quality_score

    def test_json_serialization_and_nan_inf_handling(self) -> None:
        """DatasetProfile should serialize cleanly to JSON without NaN or Inf errors."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, np.inf, -np.inf, 5.0],
        })
        profile = DatasetProfiler.profile(df)

        # Must convert to dict and serialize to JSON without raising ValueError
        json_str = json.dumps(profile.model_dump(), ensure_ascii=False)
        assert "NaN" not in json_str
        assert "Infinity" not in json_str

    def test_mixed_types_and_suspicious_columns(self) -> None:
        """Profiler should detect mixed Python types and 100% missing columns."""
        df = pd.DataFrame({
            "mixed_col": [1, "two", 3.0, "four", 5],
            "all_missing": [None, None, None, None, None],
            "normal_col": ["a", "b", "c", "d", "e"],
        })
        profile = DatasetProfiler.profile(df)

        assert "mixed_col" in profile.mixed_type_columns
        assert "all_missing" in profile.suspicious_columns
        assert "normal_col" not in profile.mixed_type_columns
        assert "normal_col" not in profile.suspicious_columns
        assert profile.columns[0].standard_deviation is None
        assert profile.columns[2].cardinality == 5
