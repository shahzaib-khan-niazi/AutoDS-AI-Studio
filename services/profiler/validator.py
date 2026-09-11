"""Validator for DatasetProfile before sending to LLM.

Ensures that the DatasetProfile is well-formed, contains no NaN/Inf floats,
and adheres to all structural and range invariants.
"""

import math
from typing import Any, List, Tuple
from core.schemas.dataset_profile import DatasetProfile, ColumnProfile
from core.logging import logger


class DatasetProfileValidator:
    """Validates deterministic DatasetProfile instances."""

    @classmethod
    def validate_profile(cls, profile: DatasetProfile) -> Tuple[bool, List[str]]:
        """Validate a DatasetProfile object.

        Args:
            profile: The DatasetProfile to validate.

        Returns:
            Tuple of (is_valid: bool, error_messages: List[str]).
        """
        errors: List[str] = []

        # 1. Dataset-level counts
        if profile.row_count < 0:
            errors.append(f"Invalid row_count: {profile.row_count} (must be >= 0)")

        if profile.column_count < 0:
            errors.append(f"Invalid column_count: {profile.column_count} (must be >= 0)")

        if len(profile.columns) != profile.column_count:
            errors.append(
                f"Column count mismatch: profile reports {profile.column_count} columns, "
                f"but contains {len(profile.columns)} ColumnProfile entries."
            )

        if profile.duplicate_row_count < 0:
            errors.append(f"Invalid duplicate_row_count: {profile.duplicate_row_count}")

        if not (0.0 <= profile.duplicate_row_percentage <= 100.0):
            errors.append(
                f"Invalid duplicate_row_percentage: {profile.duplicate_row_percentage:.2f}% (must be between 0 and 100)"
            )

        if not (0.0 <= profile.quality_score <= 100.0):
            errors.append(
                f"Invalid quality_score: {profile.quality_score} (must be between 0 and 100)"
            )

        if math.isnan(profile.quality_score) or math.isinf(profile.quality_score):
            errors.append("quality_score is NaN or Infinity")

        # 2. Validate individual columns
        for idx, col in enumerate(profile.columns):
            cls._validate_column(col, profile.row_count, idx, errors)

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(
                "DatasetProfile validation failed with {} errors: {}",
                len(errors),
                "; ".join(errors[:5]),
            )

        return is_valid, errors

    @classmethod
    def _validate_column(
        cls, col: ColumnProfile, row_count: int, idx: int, errors: List[str]
    ) -> None:
        """Validate a single ColumnProfile entry."""
        col_ref = f"Column[{idx}] ('{col.name}')"

        if not col.name or not col.name.strip():
            errors.append(f"{col_ref}: Column name is empty or whitespace.")

        if col.missing_count < 0:
            errors.append(f"{col_ref}: Invalid missing_count ({col.missing_count})")

        if col.missing_count > row_count:
            errors.append(
                f"{col_ref}: missing_count ({col.missing_count}) exceeds total row_count ({row_count})"
            )

        if not (0.0 <= col.missing_percentage <= 100.0):
            errors.append(
                f"{col_ref}: missing_percentage ({col.missing_percentage:.2f}%) out of range [0, 100]"
            )

        if col.unique_count < 0:
            errors.append(f"{col_ref}: Invalid unique_count ({col.unique_count})")

        if col.unique_count > row_count:
            errors.append(
                f"{col_ref}: unique_count ({col.unique_count}) exceeds total row_count ({row_count})"
            )

        if not (0.0 <= col.unique_percentage <= 100.0):
            errors.append(
                f"{col_ref}: unique_percentage ({col.unique_percentage:.2f}%) out of range [0, 100]"
            )

        # Check numeric stats for NaN/Inf
        num_fields = [
            ("min_val", col.min_val),
            ("max_val", col.max_val),
            ("mean_val", col.mean_val),
            ("median_val", col.median_val),
            ("std_val", col.std_val),
            ("q1_val", col.q1_val),
            ("q3_val", col.q3_val),
        ]
        for field_name, val in num_fields:
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                errors.append(f"{col_ref}: Numeric stat '{field_name}' is NaN or Infinity")
