"""Type conversion repair module.

Safe type conversion with validation — never converts blindly.
"""

import pandas as pd
from models.repair import RepairRecord
from core.logging import logger
from datetime import datetime


def convert_to_numeric(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, RepairRecord]:
    """Attempt to convert specified columns to numeric dtype.

    Uses pd.to_numeric with errors='coerce' — non-numeric values become NaN.

    Args:
        df: Source DataFrame (not modified).
        columns: Column names to convert.

    Returns:
        Tuple of (DataFrame with converted columns, RepairRecord).
    """
    result = df.copy()
    converted: list[str] = []
    failed: list[str] = []

    for col in columns:
        if col not in result.columns:
            failed.append(col)
            continue
        try:
            original_non_null = result[col].notna().sum()
            result[col] = pd.to_numeric(result[col], errors="coerce")
            new_non_null = result[col].notna().sum()

            # Warn if too many values became NaN
            lost = original_non_null - new_non_null
            if lost > 0:
                logger.warning(
                    "Column '{}': {} values became NaN during numeric conversion",
                    col, lost,
                )

            converted.append(col)
        except Exception as e:
            logger.warning("Failed to convert '{}' to numeric: {}", col, str(e))
            failed.append(col)

    record = RepairRecord(
        operation="convert_types",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=len(failed) == 0,
        warnings=[f"Failed to convert: {c}" for c in failed],
        details={"converted": converted, "failed": failed, "target_type": "numeric"},
    )

    return result, record


from services.repair.dates import normalize_date_column, DEFAULT_DISPLAY_FORMAT


def convert_to_datetime(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, RepairRecord]:
    """Attempt to convert specified columns to datetime dtype using the shared date standardization engine.

    Args:
        df: Source DataFrame (not modified).
        columns: Column names to convert.

    Returns:
        Tuple of (DataFrame with converted columns and DD-MM-YY display format, RepairRecord).
    """
    result = df.copy()
    converted: list[str] = []
    failed: list[str] = []

    for col in columns:
        if col not in result.columns:
            failed.append(col)
            continue
        try:
            result, _rec = normalize_date_column(result, col, display_format=DEFAULT_DISPLAY_FORMAT)
            converted.append(col)
        except Exception as e:
            logger.warning("Failed to convert '{}' to datetime: {}", col, str(e))
            failed.append(col)

    record = RepairRecord(
        operation="convert_types",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=len(failed) == 0,
        warnings=[f"Failed to convert: {c}" for c in failed],
        details={"converted": converted, "failed": failed, "target_type": "datetime"},
    )

    return result, record
