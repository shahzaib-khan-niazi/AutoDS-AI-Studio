"""Duplicate detection and repair module.

Provides:
- Detection of exact duplicate rows, near-duplicate rows (formatting/whitespace/case differences), and primary key identifier collisions.
- Detailed reporting: duplicate type, affected rows, reason, recommended action (never blindly auto-deletes without audit).
- Safe removal with audit logging.
"""

from typing import Any, Literal, Optional
import pandas as pd
from datetime import datetime
from models.repair import DuplicateDetectionResult, IssueTaxonomy, RepairRecord
from core.logging import logger
from services.repair.auto_dtypes import _is_identifier_column


def detect_duplicates(
    df: pd.DataFrame,
    id_columns: Optional[list[str]] = None,
) -> DuplicateDetectionResult:
    """Analyze dataset for exact duplicates, near-duplicates, and identifier collisions.

    Args:
        df: Source DataFrame.
        id_columns: Optional explicit list of identifier columns to check uniqueness for.

    Returns:
        DuplicateDetectionResult with diagnostics and recommendations.
    """
    if df.empty:
        return DuplicateDetectionResult()

    details: list[dict[str, Any]] = []
    affected_rows_set: set[int] = set()

    # 1. Exact Duplicate Rows
    exact_mask = df.duplicated(keep=False)
    exact_count = int(df.duplicated(keep="first").sum())
    if exact_count > 0:
        exact_indices = df.index[exact_mask].tolist()
        affected_rows_set.update(exact_indices)
        details.append({
            "duplicate_type": "exact_row_duplicate",
            "count": exact_count,
            "affected_rows": exact_indices[:50],
            "reason": f"{exact_count} row(s) have identical values across all columns",
            "recommended_action": "remove_exact_duplicates",
        })

    # 2. Near-Duplicate Rows (minor whitespace or casing differences in text columns)
    str_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    near_count = 0
    if str_cols and len(df) > 1 and len(df) <= 10000:
        norm_df = df.copy()
        for col in str_cols:
            norm_df[col] = norm_df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()

        norm_dups = norm_df.duplicated(keep=False)
        near_mask = norm_dups & (~exact_mask)
        near_count = int(norm_df.duplicated(keep="first").sum()) - exact_count
        if near_count > 0:
            near_indices = df.index[near_mask].tolist()
            affected_rows_set.update(near_indices)
            details.append({
                "duplicate_type": "near_row_duplicate",
                "count": near_count,
                "affected_rows": near_indices[:50],
                "reason": f"{near_count} row(s) differ only by capitalization or whitespace formatting",
                "recommended_action": "standardize_then_review_duplicates",
            })

    # 3. Duplicate Identifiers
    id_cols_to_check = id_columns if id_columns else [c for c in df.columns if _is_identifier_column(df[c])]
    id_duplicates_map: dict[str, int] = {}

    for id_col in id_cols_to_check:
        if id_col in df.columns:
            non_null_s = df[id_col].dropna()
            id_dup_count = non_null_s.duplicated().sum()
            if id_dup_count > 0:
                id_duplicates_map[id_col] = id_dup_count
                dup_id_mask = df[id_col].duplicated(keep=False) & df[id_col].notna()
                dup_id_indices = df.index[dup_id_mask].tolist()
                affected_rows_set.update(dup_id_indices)
                details.append({
                    "duplicate_type": "duplicate_identifier",
                    "column": id_col,
                    "count": id_dup_count,
                    "affected_rows": dup_id_indices[:50],
                    "reason": f"Identifier column '{id_col}' has {id_dup_count} duplicate key(s) violating unique identity",
                    "recommended_action": "investigate_key_collisions",
                })

    rec_action = "none"
    if exact_count > 0:
        rec_action = "remove_exact_duplicates"
    elif near_count > 0:
        rec_action = "review_near_duplicates"
    elif id_duplicates_map:
        rec_action = "investigate_key_collisions"

    return DuplicateDetectionResult(
        exact_duplicates_count=exact_count,
        near_duplicates_count=max(0, near_count),
        identifier_duplicates=id_duplicates_map,
        affected_rows=sorted(list(affected_rows_set)),
        recommended_action=rec_action,
        details=details,
    )


def remove_duplicate_rows(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None,
    keep: Literal["first", "last", False] = "first",
) -> tuple[pd.DataFrame, RepairRecord]:
    """Remove exact duplicate rows from a DataFrame with audit logging.

    Args:
        df: Source DataFrame (not modified).
        subset: Optional columns to consider for identifying duplicates.
        keep: 'first', 'last', or False (drop all duplicates).

    Returns:
        Tuple of (new DataFrame without duplicates, RepairRecord).
    """
    rows_before = len(df)
    result = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    rows_after = len(result)
    duplicates_removed = rows_before - rows_after

    logger.info("Removed {} duplicate rows ({} → {})", duplicates_removed, rows_before, rows_after)

    record = RepairRecord(
        operation="remove_duplicate_rows",
        timestamp=datetime.now(),
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        column=", ".join(subset) if subset else "all_columns",
        issue_type=IssueTaxonomy.DUPLICATE.value,
        original_value=f"{duplicates_removed} duplicate row(s)",
        new_value="unique rows preserved",
        method="deterministic_deduplication",
        confidence=1.0,
        reason=f"Removed {duplicates_removed} exact duplicate row(s)",
        status="applied",
        risk_level="safe",
        details={
            "duplicates_removed": duplicates_removed,
            "subset": subset,
            "keep": keep,
        },
    )

    return result, record
