"""Tests for dataset inspector service."""

import datetime
import numpy as np
import pandas as pd
import pytest

from services.inspector.service import InspectorService
from services.inspector.statistics import (
    compute_numeric_stats,
    compute_categorical_stats,
    compute_datetime_stats,
)
from services.inspector.quality import assess_quality


def test_inspector_with_mixed_dtypes() -> None:
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "category": ["A", "B", "A", "C", "B"],
        "price": [10.5, 20.0, 15.2, np.nan, 35.0],
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]),
    })

    result = InspectorService.inspect(df)

    assert result.row_count == 5
    assert result.column_count == 4
    assert len(result.numeric_stats) == 2  # id, price
    assert len(result.categorical_stats) == 1  # category
    assert len(result.datetime_stats) == 1  # timestamp
    assert result.quality.quality_score > 0.8


def test_inspector_quality_issues() -> None:
    # Dataset with duplicates, missing values, empty column, empty row
    df = pd.DataFrame({
        "a": [1, 1, 2, np.nan, 3],
        "b": ["x", "x", "y", np.nan, "z"],
        "empty_col": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "const_col": ["same", "same", "same", np.nan, "same"],
        "Unnamed: 4": [10, 10, 20, np.nan, 30],
    })

    report = assess_quality(df)

    assert report.duplicate_rows >= 1
    assert report.empty_rows >= 1
    assert "empty_col" in report.empty_columns
    assert "const_col" in report.constant_columns
    assert any("Unnamed: 4" in s for s in report.suspicious_column_names)
    assert report.quality_score < 1.0


def test_numeric_statistics_safe() -> None:
    df = pd.DataFrame({
        "all_nan": [np.nan, np.nan, np.nan],
        "single_val": [42.0, np.nan, np.nan],
        "valid_num": [10.0, 20.0, 30.0],
    })

    stats = compute_numeric_stats(df)
    assert len(stats) == 3

    valid_stat = next(s for s in stats if s.column == "valid_num")
    assert valid_stat.count == 3
    assert valid_stat.mean == 20.0
    assert valid_stat.min_value == 10.0
    assert valid_stat.max_value == 30.0

