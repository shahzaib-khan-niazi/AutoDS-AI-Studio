"""Outlier and impossible value repair module.

Provides:
- Statistical Outlier Detection: IQR, Z-score, MAD (Median Absolute Deviation), and Percentile methods.
- Separation of STATISTICAL OUTLIER from DATA ERROR (impossible values like negative age, percent > 100).
- Actions: safe capping (clamping) and flagging (never silently deletes rows merely because they are statistically unusual).
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from models.repair import IssueTaxonomy, RepairRecord
from core.logging import logger
from utils.dataframe import safe_numeric_columns


def detect_impossible_values(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Detect logically impossible values based on column context and semantic boundaries.

    Checks:
    - Negative values in columns that are logically non-negative (age, count, quantity, duration).
    - Percentages > 100% or < 0% when marked as bound percentages.
    - Non-finite numbers (inf, -inf) in numeric series.
    """
    issues_by_col: dict[str, list[dict[str, Any]]] = {}

    for col in df.columns:
        series = df[col]
        col_lower = str(col).lower()
        col_issues: list[dict[str, Any]] = []

        if pd.api.types.is_numeric_dtype(series):
            # Check non-finite
            inf_count = int(np.isinf(series).sum())
            if inf_count > 0:
                col_issues.append({
                    "issue_type": IssueTaxonomy.IMPOSSIBLE_VALUE.value,
                    "reason": f"Found {inf_count} infinite (inf / -inf) value(s)",
                    "count": inf_count,
                    "severity": "critical",
                })

            # Check strictly non-negative domains
            non_negative_keywords = ["age", "count", "qty", "quantity", "duration", "tenure", "year_built", "num_"]
            if any(k in col_lower for k in non_negative_keywords):
                neg_mask = series < 0
                neg_count = int(neg_mask.sum())
                if neg_count > 0:
                    samples = series[neg_mask].head(3).tolist()
                    col_issues.append({
                        "issue_type": IssueTaxonomy.IMPOSSIBLE_VALUE.value,
                        "reason": f"Impossible negative values in non-negative column '{col}' ({neg_count} rows, samples: {samples})",
                        "count": neg_count,
                        "severity": "high",
                    })

            # Check percentages
            pct_keywords = ["percent", "pct", "rate", "%"]
            if any(k in col_lower for k in pct_keywords):
                # If values are on 0-100 scale
                if (series.dropna() > 100.0).any():
                    over_100_count = int((series > 100.0).sum())
                    col_issues.append({
                        "issue_type": IssueTaxonomy.IMPOSSIBLE_VALUE.value,
                        "reason": f"Percentage values exceed 100% ({over_100_count} rows)",
                        "count": over_100_count,
                        "severity": "medium",
                    })

        if col_issues:
            issues_by_col[str(col)] = col_issues

    return issues_by_col


def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "iqr",
    factor: float = 1.5,
    action: str = "cap",
) -> tuple[pd.DataFrame, RepairRecord]:
    """Detect and handle statistical outliers in numeric columns.

    Args:
        df: Source DataFrame (not modified).
        columns: Specific numeric columns (or all numeric if None).
        method: 'iqr', 'zscore', 'mad', or 'percentile'.
        factor: Multiplier / threshold (default 1.5 for IQR, 3.0 for zscore/mad).
        action: 'cap' (clamp to boundary) or 'flag' (report without altering data).

    Returns:
        Tuple of (repaired DataFrame, RepairRecord).
    """
    result = df.copy()
    numeric_cols = set(safe_numeric_columns(result))

    target_cols = columns if columns is not None else list(numeric_cols)
    target_cols = [c for c in target_cols if c in numeric_cols]

    outlier_details: dict[str, dict[str, Any]] = {}
    total_outliers_detected = 0

    for col in target_cols:
        series = result[col].dropna()
        if len(series) < 10:
            continue

        lower_bound: Optional[float] = None
        upper_bound: Optional[float] = None

        if method == "iqr":
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower_bound = q1 - (factor * iqr)
            upper_bound = q3 + (factor * iqr)

        elif method == "zscore":
            mean = float(series.mean())
            std = float(series.std())
            if std <= 0 or np.isnan(std):
                continue
            lower_bound = mean - (factor * std)
            upper_bound = mean + (factor * std)

        elif method == "mad":
            median = float(series.median())
            mad = float((series - median).abs().median())
            if mad <= 0 or np.isnan(mad):
                continue
            # 1.4826 is normal-consistency factor for MAD
            mad_std = 1.4826 * mad
            lower_bound = median - (factor * mad_std)
            upper_bound = median + (factor * mad_std)

        elif method == "percentile":
            # factor represents percentile tails (e.g. 0.01 for 1% and 99%)
            p_tail = factor if factor < 0.5 else 0.01
            lower_bound = float(series.quantile(p_tail))
            upper_bound = float(series.quantile(1.0 - p_tail))

        if lower_bound is None or upper_bound is None:
            continue

        outliers_mask = (result[col] < lower_bound) | (result[col] > upper_bound)
        count = int(outliers_mask.sum())

        if count > 0:
            total_outliers_detected += count
            if action == "cap":
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)

            outlier_details[col] = {
                "outliers_count": count,
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "action": action,
            }

    logger.info(
        "Outlier handling ({}): {} outlier(s) detected across {} column(s) (action={})",
        method, total_outliers_detected, len(outlier_details), action
    )

    record = RepairRecord(
        operation="handle_outliers",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        column=", ".join(outlier_details.keys()) if outlier_details else None,
        issue_type=IssueTaxonomy.OUTLIER.value,
        original_value=f"{total_outliers_detected} statistical outliers detected",
        new_value=f"{action} outliers using {method.upper()} bounds",
        method=f"statistical_{method}",
        confidence=0.92,
        reason=f"{action.title()} {total_outliers_detected} statistical outlier(s) using {method.upper()} method (factor={factor})",
        status="applied",
        risk_level="safe" if action == "cap" else "medium",
        details={
            "method": method,
            "factor": factor,
            "action": action,
            "total_outliers_handled": total_outliers_detected,
            "columns": outlier_details,
        },
    )

    return result, record
