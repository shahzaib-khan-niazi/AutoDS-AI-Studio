"""Deterministic Cleaning Executor for AutoDS AI Studio.

Executes approved, validated cleaning operations on a defensive copy of a Pandas DataFrame.
Never runs arbitrary code strings or unvalidated instructions.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from core.logging import logger
from core.schemas.cleaning import CleaningAction, CleaningExecutionResult


class CleaningExecutor:
    """Executes validated CleaningAction operations deterministically."""

    @classmethod
    def execute_plan(
        cls, df: pd.DataFrame, actions: List[CleaningAction]
    ) -> Tuple[pd.DataFrame, CleaningExecutionResult]:
        """Execute a list of approved CleaningAction objects on a defensive copy of df.

        Args:
            df: Source DataFrame.
            actions: List of validated CleaningAction objects.

        Returns:
            Tuple of (cleaned_df: pd.DataFrame, result: CleaningExecutionResult).
        """
        # Defensive copy
        cleaned_df = df.copy()

        rows_before = len(cleaned_df)
        cols_before = len(cleaned_df.columns)
        missing_before = int(cleaned_df.isna().sum().sum())
        try:
            duplicates_before = int(cleaned_df.duplicated().sum())
        except Exception:
            duplicates_before = 0

        operations_applied: List[str] = []
        changed_columns: List[str] = []

        for action in actions:
            try:
                op = action.operation
                col = action.column

                if op == "drop_duplicates":
                    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
                    operations_applied.append("Drop duplicate rows")

                elif op == "fill_missing_numeric" and col and col in cleaned_df.columns:
                    method = (action.method or "median").lower()
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        if method == "mean":
                            val = cleaned_df[col].mean()
                        elif method == "zero":
                            val = 0.0
                        else:  # median
                            val = cleaned_df[col].median()
                        cleaned_df[col] = cleaned_df[col].fillna(val)
                        operations_applied.append(f"Fill missing '{col}' with {method} ({val:.2f})")
                        changed_columns.append(col)

                elif op == "fill_missing_categorical" and col and col in cleaned_df.columns:
                    method = (action.method or "mode").lower()
                    if method == "missing_token":
                        val = "Missing"
                    else:  # mode
                        mode_series = cleaned_df[col].mode(dropna=True)
                        val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                    cleaned_df[col] = cleaned_df[col].fillna(val)
                    operations_applied.append(f"Fill missing '{col}' with {method} ('{val}')")
                    changed_columns.append(col)

                elif op == "drop_missing_column" and col and col in cleaned_df.columns:
                    cleaned_df = cleaned_df.drop(columns=[col])
                    operations_applied.append(f"Drop column '{col}' (>60% missing)")
                    changed_columns.append(col)

                elif op == "drop_missing_rows" and col and col in cleaned_df.columns:
                    cleaned_df = cleaned_df.dropna(subset=[col]).reset_index(drop=True)
                    operations_applied.append(f"Drop rows missing in '{col}'")
                    changed_columns.append(col)

                elif op == "remove_constant_column" and col and col in cleaned_df.columns:
                    cleaned_df = cleaned_df.drop(columns=[col])
                    operations_applied.append(f"Remove constant column '{col}'")
                    changed_columns.append(col)

                elif op == "strip_whitespace" and col and col in cleaned_df.columns:
                    if pd.api.types.is_object_dtype(cleaned_df[col]) or pd.api.types.is_string_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                        operations_applied.append(f"Strip whitespace on '{col}'")
                        changed_columns.append(col)

                elif op == "normalize_categories" and col and col in cleaned_df.columns:
                    if pd.api.types.is_object_dtype(cleaned_df[col]) or pd.api.types.is_string_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.title()
                        operations_applied.append(f"Standardize categorical casing on '{col}'")
                        changed_columns.append(col)

                elif op == "cap_outliers" and col and col in cleaned_df.columns:
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        q25 = cleaned_df[col].quantile(0.25)
                        q75 = cleaned_df[col].quantile(0.75)
                        iqr = q75 - q25
                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr
                        cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                        operations_applied.append(f"Cap IQR outliers on '{col}' [{lower_bound:.2f}, {upper_bound:.2f}]")
                        changed_columns.append(col)

                elif op == "convert_dtype" and col and col in cleaned_df.columns:
                    target_type = action.parameters.get("target_dtype", "numeric").lower()
                    if target_type == "numeric":
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
                    elif target_type == "datetime":
                        cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors="coerce")
                    elif target_type == "category":
                        cleaned_df[col] = cleaned_df[col].astype("category")
                    operations_applied.append(f"Cast '{col}' to {target_type}")
                    changed_columns.append(col)

            except Exception as exc:
                logger.warning("Error executing cleaning action '{}' on col '{}': {}", action.operation, action.column, str(exc))

        rows_after = len(cleaned_df)
        cols_after = len(cleaned_df.columns)
        missing_after = int(cleaned_df.isna().sum().sum())
        try:
            duplicates_after = int(cleaned_df.duplicated().sum())
        except Exception:
            duplicates_after = 0

        diff_summary = (
            f"Rows: {rows_before} → {rows_after} | Cols: {cols_before} → {cols_after} | "
            f"Missing cells: {missing_before} → {missing_after} | Duplicates: {duplicates_before} → {duplicates_after}"
        )

        result = CleaningExecutionResult(
            rows_before=rows_before,
            rows_after=rows_after,
            cols_before=cols_before,
            cols_after=cols_after,
            missing_before=missing_before,
            missing_after=missing_after,
            duplicates_before=duplicates_before,
            duplicates_after=duplicates_after,
            operations_applied=operations_applied,
            changed_columns=list(set(changed_columns)),
            diff_summary=diff_summary,
        )

        logger.info("Cleaning execution finished: {}", diff_summary)
        return cleaned_df, result
