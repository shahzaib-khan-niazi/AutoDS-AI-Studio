"""Type conversion repair module.

Safe type conversion with validation — never converts blindly.
"""

import pandas as pd
from typing import Any, Optional
from models.repair import RepairRecord
from core.logging import logger
from datetime import datetime


import numpy as np
from services.repair.auto_dtypes import parse_numeric_value
from utils.number_parser import parse_written_number, parse_magnitude_abbreviation


MISSING_SENTINEL_STRINGS = {
    "n/a", "na", "null", "none", "nan", "unavailable", "unknown",
    "missing", "-", "?", "#n/a", "#na", "#value!", "#ref!", "#error",
}


def convert_series_to_numeric_explicit(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    """Convert any Series to numeric dtype (Int64 or Float64) upon explicit user request.

    Order of interpretation per cell:
    1. Already numeric (int/float)
    2. Standard numeric string ("34", "43.5")
    3. Formatted numeric string ("$50,000", "50K", "1.5M", "(1,000)", "25%")
    4. Natural language written number ("ten", "five", "one hundred", "twenty five thousand")
    5. Missing / unparseable / arbitrary text -> pd.NA (nullable Int64/Float64)
    
    Arbitrary text values that cannot be interpreted as numbers become pd.NA with lineage.
    Categorical integer coding is NEVER invented during explicit numeric conversion.
    """
    total = len(series)
    orig_dtype = str(series.dtype)

    parsed_values: list[Any] = [None] * total
    affected_cells: list[dict[str, Any]] = []

    # Metrics
    already_numeric_cnt = 0
    numeric_strings_cnt = 0
    formatted_numeric_cnt = 0
    written_numbers_cnt = 0
    unparseable_text_cnt = 0
    missing_cnt = 0
    col_name = str(series.name) if series.name else ""

    for idx, raw in enumerate(series):
        if pd.isna(raw):
            parsed_values[idx] = None
            missing_cnt += 1
            continue

        if isinstance(raw, (int, float, np.integer, np.floating)):
            if np.isnan(raw) or np.isinf(raw):
                parsed_values[idx] = None
                missing_cnt += 1
            else:
                parsed_values[idx] = float(raw)
                already_numeric_cnt += 1
            continue

        s = str(raw).strip()
        if not s or s.lower() in MISSING_SENTINEL_STRINGS:
            parsed_values[idx] = None
            missing_cnt += 1
            continue

        # 1. Standard numeric digits or float strings
        try:
            val = float(s)
            parsed_values[idx] = val
            numeric_strings_cnt += 1
            continue
        except (ValueError, TypeError):
            pass

        # 2. Magnitude abbreviation strings (50K, 1.5M, $50K, 2B PKR)
        mag_val = parse_magnitude_abbreviation(s)
        if mag_val is not None:
            parsed_values[idx] = mag_val
            formatted_numeric_cnt += 1
            continue

        # 3. Written natural language numbers ("ten", "five", "one hundred", "twenty five thousand")
        written_val = parse_written_number(s)
        if written_val is not None:
            parsed_values[idx] = written_val
            written_numbers_cnt += 1
            continue

        # 4. Formatted numeric values (currency, percentage, locale separators, negative parentheses)
        generic_val = parse_numeric_value(s)
        if generic_val is not None:
            parsed_values[idx] = generic_val
            formatted_numeric_cnt += 1
            continue

        # 5. Non-numeric text that cannot be interpreted as a number -> pd.NA
        parsed_values[idx] = None
        unparseable_text_cnt += 1
        affected_cells.append({
            "row_index": idx,
            "column": col_name,
            "original_value": raw,
            "cleaned_value": pd.NA,
            "reason": "not reliably interpretable as numeric",
            "transformation": "explicit_numeric_conversion",
            "confidence": 1.0,
        })

    # Construct clean nullable pandas series
    final_series = pd.Series(parsed_values, index=series.index)

    # Choose Int64 (nullable int) vs Float64 (nullable float)
    valid_nums = final_series.dropna()
    if len(valid_nums) > 0 and (valid_nums % 1 == 0).all():
        converted_series = final_series.round().astype("Int64")
    else:
        converted_series = final_series.astype("Float64")

    total_unavailable = missing_cnt + unparseable_text_cnt

    summary_metrics = {
        "already_numeric": already_numeric_cnt,
        "numeric_strings": numeric_strings_cnt,
        "formatted_numeric_values": formatted_numeric_cnt,
        "written_numbers": written_numbers_cnt,
        "unparseable_text_to_na": unparseable_text_cnt,
        "categorical_converted_to_numeric_codes": 0,
        "missing_values": missing_cnt,
        "unavailable_values": total_unavailable,
        "category_mapping": {},
        "affected_cells": affected_cells,
        "original_dtype": orig_dtype,
        "final_dtype": str(converted_series.dtype),
    }

    return converted_series, summary_metrics


def convert_to_numeric(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, RepairRecord]:
    """Explicitly convert specified columns to numeric dtype (Int64 or Float64).

    Applies explicit 4-step numeric conversion hierarchy:
    1. Genuine numeric & numeric strings
    2. Formatted numbers (currency, percentage, magnitude K/M/B/T)
    3. Natural language written numbers ("ten", "five", "one hundred", "twenty five thousand")
    4. Unparseable arbitrary text -> pd.NA (with lineage logging)

    Guarantees row count and column count preservation.

    Args:
        df: Source DataFrame (not modified).
        columns: Column names to convert.

    Returns:
        Tuple of (DataFrame with converted columns, RepairRecord).
    """
    result = df.copy()
    converted: list[str] = []
    failed: list[str] = []
    all_metrics: dict[str, Any] = {}
    all_affected_cells: list[dict[str, Any]] = []

    for col in columns:
        if col not in result.columns:
            failed.append(col)
            continue
        try:
            converted_series, metrics = convert_series_to_numeric_explicit(result[col])
            result[col] = converted_series
            converted.append(col)
            all_metrics[col] = metrics
            if "affected_cells" in metrics:
                all_affected_cells.extend(metrics["affected_cells"])
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
        column=", ".join(converted) if converted else None,
        issue_type="DATATYPE",
        method="explicit_user_numeric_conversion",
        confidence=1.0,
        reason=f"Explicitly converted {len(converted)} column(s) to numeric dtype",
        status="applied",
        risk_level="safe",
        warnings=[f"Failed to convert: {c}" for c in failed],
        details={
            "converted": converted,
            "failed": failed,
            "target_type": "numeric",
            "metrics": all_metrics,
            "affected_cells": all_affected_cells,
        },
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
