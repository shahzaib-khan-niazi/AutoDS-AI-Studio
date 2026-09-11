"""Whitespace cleaning repair module.

Strips leading/trailing whitespace and collapses multiple internal spaces
to a single space in all object/string columns.
"""

import re
from datetime import datetime
import pandas as pd
from models.repair import RepairRecord
from core.logging import logger


def strip_whitespace(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, RepairRecord]:
    """Strip whitespace from string values across the DataFrame.

    For every object/string column:
    - Removes leading and trailing whitespace
    - Collapses multiple internal spaces to a single space

    Args:
        df: Source DataFrame (not modified).
        columns: Specific columns to clean. If None, cleans all object columns.

    Returns:
        Tuple of (cleaned DataFrame, RepairRecord).
    """
    result = df.copy()

    if columns:
        target_cols = [c for c in columns if c in result.columns and result[c].dtype == "object"]
    else:
        target_cols = [c for c in result.columns if result[c].dtype == "object"]

    total_cells_modified = 0
    column_details: dict[str, int] = {}

    for col in target_cols:
        original_values = result[col].copy()

        # Only process non-null values
        mask = result[col].notna()
        if mask.sum() == 0:
            continue

        str_series = result[col][mask].astype(str)

        # Strip edges
        stripped = str_series.str.strip()
        # Collapse multiple internal spaces to single space
        collapsed = stripped.str.replace(r"\s+", " ", regex=True)

        result.loc[mask, col] = collapsed

        # Count how many cells actually changed
        changed_count = (original_values[mask].astype(str) != collapsed).sum()
        if changed_count > 0:
            total_cells_modified += changed_count
            column_details[col] = changed_count

    logger.info(
        "Whitespace cleanup: {} cells modified across {} columns",
        total_cells_modified,
        len(column_details),
    )

    record = RepairRecord(
        operation="strip_whitespace",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        details={
            "total_cells_modified": total_cells_modified,
            "columns_cleaned": column_details,
        },
    )

    return result, record
