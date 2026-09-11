"""Post-repair validator.

Every repair must pass validation before being committed.
Checks row/column sanity, all-null columns, catastrophic loss.
"""

import pandas as pd

from core.logging import logger
from core.constants import MAX_ROW_LOSS_PERCENTAGE, MAX_COLUMN_LOSS_PERCENTAGE
from models.repair import ValidationResult


def validate_repair(
    original_df: pd.DataFrame,
    repaired_df: pd.DataFrame,
    operation: str,
) -> ValidationResult:
    """Validate a repaired DataFrame against the original.

    Args:
        original_df: DataFrame before repair.
        repaired_df: DataFrame after repair.
        operation: Name of the repair operation performed.

    Returns:
        ValidationResult indicating whether the repair is safe to commit.
    """
    warnings: list[str] = []
    errors: list[str] = []

    # Check DataFrame exists and is valid
    if not isinstance(repaired_df, pd.DataFrame):
        return ValidationResult(
            valid=False,
            confidence=0.0,
            errors=["Repair did not produce a valid DataFrame"],
        )

    if repaired_df.empty and not original_df.empty:
        return ValidationResult(
            valid=False,
            confidence=0.0,
            errors=["Repair produced an empty DataFrame"],
        )

    original_rows = len(original_df)
    repaired_rows = len(repaired_df)
    original_cols = len(original_df.columns)
    repaired_cols = len(repaired_df.columns)

    # Check for catastrophic row loss (except for structural ops like unpivot, explode)
    structural_ops = {
        "unpivot",
        "flatten_headers",
        "explode_multi_value_cells",
        "remove_repeated_headers",
        "remove_metadata_rows",
    }
    if operation not in structural_ops and original_rows > 0:
        row_loss = (original_rows - repaired_rows) / original_rows
        if row_loss > MAX_ROW_LOSS_PERCENTAGE:
            errors.append(
                f"Catastrophic row loss: {row_loss:.0%} "
                f"({original_rows} -> {repaired_rows})"
            )
        elif row_loss > 0.2:
            warnings.append(
                f"Significant row reduction: {row_loss:.0%} "
                f"({original_rows} → {repaired_rows})"
            )

    # Check for catastrophic column loss (except for structural ops)
    if operation not in structural_ops and original_cols > 0:
        col_loss = (original_cols - repaired_cols) / original_cols
        if col_loss > MAX_COLUMN_LOSS_PERCENTAGE:
            errors.append(
                f"Catastrophic column loss: {col_loss:.0%} "
                f"({original_cols} → {repaired_cols})"
            )

    # Check for unexpected all-null columns or severe null leakage during type conversions
    non_null_loss_ops = {"convert_types", "auto_dtypes", "normalize_dates", "standardize_values"}
    for col in repaired_df.columns:
        col_str = str(col)
        orig_cols = [str(c) for c in original_df.columns]
        if col_str in orig_cols:
            orig_nulls = int(original_df[col_str].isna().sum())
            new_nulls = int(repaired_df[col].isna().sum())
            if repaired_df[col].isna().all() and repaired_cols > 0:
                if not original_df[col_str].isna().all():
                    errors.append(f"Column '{col_str}' became entirely null after repair")
            elif operation in non_null_loss_ops and new_nulls > orig_nulls:
                lost_values = new_nulls - orig_nulls
                loss_ratio = lost_values / max(len(original_df) - orig_nulls, 1)
                if loss_ratio > 0.15:
                    errors.append(
                        f"Data loss rejected: '{col_str}' lost {lost_values} non-null values ({loss_ratio:.1%}) during {operation}"
                    )
                else:
                    warnings.append(
                        f"Column '{col_str}' lost {lost_values} value(s) ({loss_ratio:.1%}) during {operation}"
                    )

    # Check column names are valid strings
    for col in repaired_df.columns:
        if col is None:
            errors.append("Repair produced a None column name")
        elif str(col).strip() == "" and operation != "flatten_headers":
            warnings.append(f"Empty column name detected: {repr(col)}")

    # Compute confidence
    confidence = 1.0
    confidence -= len(warnings) * 0.05
    confidence -= len(errors) * 0.3
    confidence = max(0.0, min(1.0, confidence))

    valid = len(errors) == 0

    if not valid:
        logger.warning("Repair validation FAILED for '{}': {}", operation, errors)
    elif warnings:
        logger.info("Repair validation passed with warnings for '{}': {}", operation, warnings)
    else:
        logger.info("Repair validation passed for '{}'", operation)

    return ValidationResult(
        valid=valid,
        confidence=confidence,
        warnings=warnings,
        errors=errors,
    )
