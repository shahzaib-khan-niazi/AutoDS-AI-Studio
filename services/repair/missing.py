"""Missing value repair module.

Supports dropping empty rows/columns, dropping high-null columns,
and intelligent deterministic imputation strategies (median, mode, mean, ffill, bfill).
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
from models.repair import IssueTaxonomy, RepairRecord
from core.logging import logger
from datetime import datetime
from utils.dataframe import safe_numeric_columns, safe_datetime_columns, safe_categorical_columns
from services.repair.auto_dtypes import _is_identifier_column


def drop_empty_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, RepairRecord]:
    """Drop rows that are completely empty (all NaN).

    Args:
        df: Source DataFrame (not modified).

    Returns:
        Tuple of (DataFrame without empty rows, RepairRecord).
    """
    rows_before = len(df)
    result = df.dropna(how="all").reset_index(drop=True)
    rows_after = len(result)
    dropped = rows_before - rows_after

    logger.info("Dropped {} completely empty rows", dropped)

    record = RepairRecord(
        operation="drop_empty_rows",
        timestamp=datetime.now(),
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        column="all_columns",
        issue_type=IssueTaxonomy.MISSING.value,
        original_value=f"{dropped} empty row(s)",
        new_value="removed",
        method="deterministic",
        confidence=1.0,
        reason=f"Dropped {dropped} completely empty row(s)",
        status="applied",
        risk_level="safe",
        details={"empty_rows_dropped": dropped},
    )

    return result, record


def drop_empty_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, RepairRecord]:
    """Drop columns that are completely empty (all NaN).

    Args:
        df: Source DataFrame (not modified).

    Returns:
        Tuple of (DataFrame without empty columns, RepairRecord).
    """
    cols_before = len(df.columns)
    empty_cols = [col for col in df.columns if df[col].isna().all()]

    result = df.copy()
    if empty_cols:
        result = result.drop(columns=empty_cols)

    cols_after = len(result.columns)

    logger.info("Dropped {} completely empty columns: {}", len(empty_cols), empty_cols)

    record = RepairRecord(
        operation="drop_empty_columns",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=cols_before,
        columns_after=cols_after,
        success=True,
        column=", ".join(empty_cols) if empty_cols else None,
        issue_type=IssueTaxonomy.MISSING.value,
        original_value=f"{len(empty_cols)} empty column(s)",
        new_value="removed",
        method="deterministic",
        confidence=1.0,
        reason=f"Dropped {len(empty_cols)} completely empty column(s)",
        status="applied",
        risk_level="safe",
        details={"empty_columns_dropped": empty_cols},
    )

    return result, record


def drop_high_missing_columns(
    df: pd.DataFrame, threshold: float = 0.8
) -> tuple[pd.DataFrame, RepairRecord]:
    """Drop columns that have missing percentage exceeding the threshold.

    Args:
        df: Source DataFrame (not modified).
        threshold: Max allowable missing ratio (default 0.8 = 80% null).

    Returns:
        Tuple of (DataFrame without excessive null columns, RepairRecord).
    """
    cols_before = len(df.columns)
    total_rows = len(df)
    dropped_cols: list[str] = []

    result = df.copy()
    if total_rows > 0:
        for col in df.columns:
            null_pct = df[col].isna().sum() / total_rows
            if null_pct >= threshold and not df[col].isna().all():
                dropped_cols.append(col)

        if dropped_cols:
            result = result.drop(columns=dropped_cols)

    cols_after = len(result.columns)
    logger.info("Dropped {} high-missing columns: {}", len(dropped_cols), dropped_cols)

    record = RepairRecord(
        operation="drop_high_missing_columns",
        timestamp=datetime.now(),
        rows_before=total_rows,
        rows_after=len(result),
        columns_before=cols_before,
        columns_after=cols_after,
        success=True,
        column=", ".join(dropped_cols) if dropped_cols else None,
        issue_type=IssueTaxonomy.MISSING.value,
        original_value=f"{len(dropped_cols)} column(s) with >={threshold:.0%} missing",
        new_value="removed",
        method="deterministic",
        confidence=0.95,
        reason=f"Dropped {len(dropped_cols)} column(s) exceeding {threshold:.0%} missing threshold",
        status="applied",
        risk_level="medium",
        details={"threshold": threshold, "dropped_columns": dropped_cols},
    )

    return result, record


def fill_missing(
    df: pd.DataFrame,
    strategy: str = "auto",
    columns: Optional[list[str]] = None,
    custom_value: Any = None,
) -> tuple[pd.DataFrame, RepairRecord]:
    """Fill missing values using specified or automatic dtype-aware strategy.

    Strategies:
    - 'auto': Numeric -> median, Categorical -> mode or 'Missing', Datetime -> ffill
    - 'median': Numeric median
    - 'mean': Numeric mean
    - 'mode': Most frequent value
    - 'constant': Fixed custom value
    - 'ffill': Forward fill
    - 'bfill': Backward fill

    Args:
        df: Source DataFrame (not modified).
        strategy: Imputation strategy.
        columns: Columns to impute (all missing if None).
        custom_value: Value for 'constant' strategy.

    Returns:
        Tuple of (DataFrame with imputed values, RepairRecord).
    """
    result = df.copy()
    missing_before = int(result.isna().sum().sum())

    if missing_before == 0:
        return result, RepairRecord(
            operation="fill_missing",
            timestamp=datetime.now(),
            rows_before=len(df),
            rows_after=len(df),
            columns_before=len(df.columns),
            columns_after=len(df.columns),
            success=True,
            column="all_columns",
            issue_type=IssueTaxonomy.MISSING.value,
            original_value="0 missing values",
            new_value="unchanged",
            method="deterministic",
            confidence=1.0,
            reason="No missing values found to impute",
            status="applied",
            risk_level="safe",
            details={"strategy": strategy, "cells_filled": 0},
        )

    target_cols = columns if columns is not None else [c for c in result.columns if result[c].isna().any()]
    imputed_details: dict[str, str] = {}

    numeric_cols = set(safe_numeric_columns(result))
    datetime_cols = set(safe_datetime_columns(result))
    categorical_cols = set(safe_categorical_columns(result))

    for col in target_cols:
        if col not in result.columns or not result[col].isna().any():
            continue

        non_null = result[col].dropna()
        if len(non_null) == 0:
            continue

        strat = strategy
        if strat == "auto":
            if _is_identifier_column(result[col]):
                continue
            elif col in numeric_cols:
                strat = "median"
            elif col in datetime_cols:
                strat = "ffill"
            elif col in categorical_cols:
                strat = "mode"
            else:
                strat = "mode"

        if strat == "median":
            if col in numeric_cols or pd.api.types.is_numeric_dtype(result[col]):
                fill_val = non_null.median()
                result[col] = result[col].fillna(fill_val)
                imputed_details[col] = f"median ({fill_val:.2f})"

        elif strat == "mean":
            if col in numeric_cols or pd.api.types.is_numeric_dtype(result[col]):
                fill_val = non_null.mean()
                result[col] = result[col].fillna(fill_val)
                imputed_details[col] = f"mean ({fill_val:.2f})"

        elif strat == "mode":
            mode_vals = non_null.mode()
            if len(mode_vals) > 0:
                fill_val = mode_vals.iloc[0]
                result[col] = result[col].fillna(fill_val)
                imputed_details[col] = f"mode ('{fill_val}')"

        elif strat == "constant":
            if custom_value is not None:
                result[col] = result[col].fillna(custom_value)
                imputed_details[col] = f"constant ({custom_value})"

        elif strat == "ffill":
            result[col] = result[col].ffill()
            imputed_details[col] = "forward fill"

        elif strat == "bfill":
            result[col] = result[col].bfill()
            imputed_details[col] = "backward fill"

    missing_after = int(result.isna().sum().sum())
    cells_filled = missing_before - missing_after

    logger.info("Filled {} missing cells across {} columns using strategy '{}'", cells_filled, len(imputed_details), strategy)

    record = RepairRecord(
        operation="fill_missing",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        column=", ".join(imputed_details.keys()) if imputed_details else None,
        issue_type=IssueTaxonomy.MISSING.value,
        original_value=f"{missing_before} missing cells",
        new_value=f"{missing_after} missing cells",
        method=f"deterministic_imputation_{strategy}",
        confidence=0.92,
        reason=f"Imputed {cells_filled} missing cell(s) using strategy '{strategy}'",
        status="applied",
        risk_level="safe",
        details={
            "strategy": strategy,
            "cells_filled": cells_filled,
            "columns_imputed": imputed_details,
        },
    )

    return result, record
