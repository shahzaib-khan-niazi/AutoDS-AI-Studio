"""Data quality assessment for dataset inspection.

Detects missing values, duplicates, empty rows/columns, mixed types,
and computes a deterministic quality score.
"""

import numpy as np
import pandas as pd

from core.logging import logger
from core.constants import HIGH_CARDINALITY_THRESHOLD, CONSTANT_COLUMN_THRESHOLD
from models.inspection import QualityReport


def assess_quality(df: pd.DataFrame) -> QualityReport:
    """Assess the data quality of a DataFrame.

    Args:
        df: Source DataFrame.

    Returns:
        QualityReport with all detected issues and a quality score.
    """
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    missing_pct = missing_cells / total_cells if total_cells > 0 else 0.0

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Completely empty rows (all NaN)
    empty_rows = int(df.isna().all(axis=1).sum())

    # Completely empty columns (all NaN)
    empty_columns: list[str] = []
    for col in df.columns:
        if df[col].isna().all():
            empty_columns.append(str(col))

    # Constant columns (only one unique non-null value)
    constant_columns: list[str] = []
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) > 0 and non_null.nunique() <= CONSTANT_COLUMN_THRESHOLD:
            constant_columns.append(str(col))

    # High-cardinality columns (almost all unique values)
    high_cardinality_columns: list[str] = []
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            ratio = non_null.nunique() / len(non_null)
            if ratio >= HIGH_CARDINALITY_THRESHOLD and len(non_null) > 10:
                high_cardinality_columns.append(str(col))

    # Possible ID columns (unique values = row count, non-numeric or sequential int)
    possible_id_columns: list[str] = []
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) > 0 and non_null.nunique() == len(non_null):
            col_str = str(col).lower()
            if any(kw in col_str for kw in ["id", "key", "code", "index", "number"]):
                possible_id_columns.append(str(col))

    # Mixed-type columns (object columns with multiple Python types)
    mixed_type_columns: list[str] = []
    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                types_found = set()
                for val in non_null.head(100):
                    types_found.add(type(val).__name__)
                if len(types_found) > 1:
                    mixed_type_columns.append(str(col))

    # Suspicious column names
    suspicious_column_names: list[str] = []
    for col in df.columns:
        col_str = str(col)
        if col_str.startswith("Unnamed:"):
            suspicious_column_names.append(col_str)
        elif col_str.strip() == "":
            suspicious_column_names.append(repr(col_str))
        elif col_str != col_str.strip():
            suspicious_column_names.append(col_str)

    # Check for duplicate column names
    col_names = [str(c) for c in df.columns]
    seen: set[str] = set()
    for name in col_names:
        if name in seen:
            if name not in suspicious_column_names:
                suspicious_column_names.append(f"{name} (duplicate)")
        seen.add(name)

    # Quality score (0.0 to 1.0)
    quality_score, quality_notes = _compute_quality_score(
        total_cells=total_cells,
        missing_pct=missing_pct,
        duplicate_rows=duplicate_rows,
        row_count=len(df),
        empty_rows=empty_rows,
        empty_columns=empty_columns,
        constant_columns=constant_columns,
        mixed_type_columns=mixed_type_columns,
        suspicious_column_names=suspicious_column_names,
    )

    return QualityReport(
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_percentage=missing_pct,
        duplicate_rows=duplicate_rows,
        empty_rows=empty_rows,
        empty_columns=empty_columns,
        constant_columns=constant_columns,
        mixed_type_columns=mixed_type_columns,
        high_cardinality_columns=high_cardinality_columns,
        possible_id_columns=possible_id_columns,
        suspicious_column_names=suspicious_column_names,
        quality_score=quality_score,
        quality_notes=quality_notes,
    )


def _compute_quality_score(
    total_cells: int,
    missing_pct: float,
    duplicate_rows: int,
    row_count: int,
    empty_rows: int,
    empty_columns: list[str],
    constant_columns: list[str],
    mixed_type_columns: list[str],
    suspicious_column_names: list[str],
) -> tuple[float, list[str]]:
    """Compute a deterministic, explainable quality score.

    Score starts at 1.0 and is reduced by detected issues.

    Returns:
        Tuple of (score, list of explanatory notes).
    """
    score = 1.0
    notes: list[str] = []

    # Penalize missing values
    if missing_pct > 0:
        penalty = min(missing_pct * 0.5, 0.3)
        score -= penalty
        notes.append(f"Missing values: {missing_pct:.1%} (-{penalty:.2f})")

    # Penalize duplicates
    if duplicate_rows > 0 and row_count > 0:
        dup_pct = duplicate_rows / row_count
        penalty = min(dup_pct * 0.3, 0.15)
        score -= penalty
        notes.append(f"Duplicate rows: {duplicate_rows} (-{penalty:.2f})")

    # Penalize empty rows
    if empty_rows > 0:
        penalty = min(0.1, empty_rows / max(row_count, 1) * 0.2)
        score -= penalty
        notes.append(f"Empty rows: {empty_rows} (-{penalty:.2f})")

    # Penalize empty columns
    if len(empty_columns) > 0:
        penalty = min(0.1, len(empty_columns) * 0.03)
        score -= penalty
        notes.append(f"Empty columns: {len(empty_columns)} (-{penalty:.2f})")

    # Penalize constant columns
    if len(constant_columns) > 0:
        penalty = min(0.05, len(constant_columns) * 0.01)
        score -= penalty
        notes.append(f"Constant columns: {len(constant_columns)} (-{penalty:.2f})")

    # Penalize mixed types
    if len(mixed_type_columns) > 0:
        penalty = min(0.1, len(mixed_type_columns) * 0.03)
        score -= penalty
        notes.append(f"Mixed-type columns: {len(mixed_type_columns)} (-{penalty:.2f})")

    # Penalize suspicious column names
    if len(suspicious_column_names) > 0:
        penalty = min(0.1, len(suspicious_column_names) * 0.02)
        score -= penalty
        notes.append(f"Suspicious column names: {len(suspicious_column_names)} (-{penalty:.2f})")

    # Clamp
    score = max(0.0, min(1.0, score))

    if len(notes) == 0:
        notes.append("No quality issues detected")

    return score, notes
