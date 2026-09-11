"""Unit tests for DatasetProfileValidator (services/profiler/validator.py).

Verifies structural and range validation of DatasetProfile objects.
"""

import pytest

from core.schemas.dataset_profile import DatasetProfile, ColumnProfile
from services.profiler.validator import DatasetProfileValidator


class TestDatasetProfileValidator:
    """Unit tests for DatasetProfileValidator class."""

    def test_valid_profile_passes_validation(self) -> None:
        """A well-formed DatasetProfile should pass validation without errors."""
        profile = DatasetProfile(
            filename="valid.csv",
            row_count=10,
            column_count=2,
            duplicate_row_count=0,
            duplicate_row_percentage=0.0,
            memory_usage_mb=0.1,
            quality_score=95.0,
            quality_status="Excellent",
            columns=[
                ColumnProfile(
                    name="id",
                    dtype="Integer",
                    pandas_dtype="int64",
                    non_null_count=10,
                    missing_count=0,
                    missing_percentage=0.0,
                    unique_count=10,
                    unique_percentage=100.0,
                    is_numeric=True,
                ),
                ColumnProfile(
                    name="name",
                    dtype="String/Text",
                    pandas_dtype="object",
                    non_null_count=10,
                    missing_count=0,
                    missing_percentage=0.0,
                    unique_count=10,
                    unique_percentage=100.0,
                    is_categorical=True,
                ),
            ],
        )

        is_valid, errors = DatasetProfileValidator.validate_profile(profile)
        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_row_and_column_counts(self) -> None:
        """Validator should detect negative row count and column count mismatches."""
        profile = DatasetProfile(
            filename="bad.csv",
            row_count=-5,
            column_count=2,
            duplicate_row_count=0,
            duplicate_row_percentage=0.0,
            memory_usage_mb=0.1,
            quality_score=50.0,
            columns=[
                ColumnProfile(
                    name="c1",
                    dtype="Integer",
                    pandas_dtype="int64",
                    non_null_count=5,
                    missing_count=0,
                    missing_percentage=0.0,
                    unique_count=5,
                    unique_percentage=100.0,
                )
            ],  # Only 1 column, but column_count=2
        )

        is_valid, errors = DatasetProfileValidator.validate_profile(profile)
        assert is_valid is False
        assert any("row_count" in err for err in errors)
        assert any("Column count mismatch" in err for err in errors)

    def test_invalid_percentages_and_missing_counts(self) -> None:
        """Validator should flag missing_count > row_count or percentage out of range."""
        profile = DatasetProfile(
            filename="bad_pct.csv",
            row_count=10,
            column_count=1,
            duplicate_row_count=0,
            duplicate_row_percentage=150.0,  # Invalid
            memory_usage_mb=0.1,
            quality_score=50.0,
            columns=[
                ColumnProfile(
                    name="col1",
                    dtype="Integer",
                    pandas_dtype="int64",
                    non_null_count=5,
                    missing_count=15,  # Exceeds row_count of 10
                    missing_percentage=150.0,
                    unique_count=5,
                    unique_percentage=50.0,
                )
            ],
        )

        is_valid, errors = DatasetProfileValidator.validate_profile(profile)
        assert is_valid is False
        assert any("duplicate_row_percentage" in err for err in errors)
        assert any("exceeds total row_count" in err for err in errors)

    def test_empty_column_name_detected(self) -> None:
        """Validator should flag empty or whitespace column names."""
        profile = DatasetProfile(
            filename="empty_col_name.csv",
            row_count=5,
            column_count=1,
            duplicate_row_count=0,
            duplicate_row_percentage=0.0,
            memory_usage_mb=0.1,
            quality_score=80.0,
            columns=[
                ColumnProfile(
                    name="   ",
                    dtype="Integer",
                    pandas_dtype="int64",
                    non_null_count=5,
                    missing_count=0,
                    missing_percentage=0.0,
                    unique_count=5,
                    unique_percentage=100.0,
                )
            ],
        )

        is_valid, errors = DatasetProfileValidator.validate_profile(profile)
        assert is_valid is False
        assert any("Column name is empty" in err for err in errors)

    def test_nan_or_inf_in_stats_fails_validation(self) -> None:
        """Validator should detect NaN/Inf values in column numeric statistics."""
        profile = DatasetProfile(
            filename="inf_stat.csv",
            row_count=5,
            column_count=1,
            duplicate_row_count=0,
            duplicate_row_percentage=0.0,
            memory_usage_mb=0.1,
            quality_score=80.0,
            columns=[
                ColumnProfile(
                    name="val",
                    dtype="Float",
                    pandas_dtype="float64",
                    non_null_count=5,
                    missing_count=0,
                    missing_percentage=0.0,
                    unique_count=5,
                    unique_percentage=100.0,
                    mean_val=float("inf"),
                )
            ],
        )

        is_valid, errors = DatasetProfileValidator.validate_profile(profile)
        assert is_valid is False
        assert any("NaN or Infinity" in err for err in errors)
