"""Repair service — public API for dataset repair operations.

Orchestrates: copy → dispatch → validate → commit or reject.
"""

import pandas as pd
from typing import Any, Optional

from core.logging import logger
from core.exceptions import RepairError, RepairValidationError
from models.repair import RepairAction, RepairOperation, RepairResult, RepairRecord
from services.repair.dispatcher import dispatch
from services.repair.validator import validate_repair
from services.repair.history import RepairHistory
from utils.dataframe import safe_copy


class RepairService:
    """Service for executing validated repair operations on a dataset."""

    @classmethod
    def repair(
        cls,
        df: pd.DataFrame,
        actions: list[RepairAction],
    ) -> tuple[pd.DataFrame, RepairResult]:
        """Execute a sequence of repair actions on a DataFrame.

        Each action goes through: copy → execute → validate → accept/reject.

        Args:
            df: Source DataFrame (not modified).
            actions: List of repair actions to apply sequentially.

        Returns:
            Tuple of (repaired DataFrame, RepairResult).

        Raises:
            RepairError: If a critical repair failure occurs.
        """
        logger.info("Starting repair with {} action(s)", len(actions))

        current_df = safe_copy(df)
        records: list[RepairRecord] = []
        actions_applied: list[str] = []
        all_warnings: list[str] = []
        all_errors: list[str] = []

        for action in actions:
            operation_name = action.operation.value
            logger.info("Executing repair: {}", operation_name)

            try:
                # Execute the repair on a copy
                repaired_df, record = dispatch(current_df, action)

                # Validate the result
                validation = validate_repair(
                    original_df=current_df,
                    repaired_df=repaired_df,
                    operation=operation_name,
                )

                if validation.valid:
                    # Commit the repair
                    current_df = repaired_df
                    record.success = True
                    records.append(record)
                    actions_applied.append(operation_name)

                    if validation.warnings:
                        all_warnings.extend(validation.warnings)

                    logger.info("Repair committed: {}", operation_name)
                else:
                    # Reject the repair
                    record.success = False
                    record.warnings = validation.errors
                    records.append(record)
                    all_errors.extend(validation.errors)

                    logger.warning(
                        "Repair rejected: {} — {}",
                        operation_name,
                        validation.errors,
                    )

            except RepairError as e:
                all_errors.append(f"{operation_name}: {e.message}")
                logger.error("Repair failed: {} — {}", operation_name, e.message)

            except Exception as e:
                all_errors.append(f"{operation_name}: {str(e)}")
                logger.exception("Unexpected repair error: {}", str(e))

        # Build final result
        overall_success = len(actions_applied) > 0 and len(all_errors) == 0

        result = RepairResult(
            success=overall_success,
            actions_applied=actions_applied,
            records=records,
            warnings=all_warnings,
            errors=all_errors,
        )

        logger.info(
            "Repair complete: {}/{} actions applied, {} warnings, {} errors",
            len(actions_applied),
            len(actions),
            len(all_warnings),
            len(all_errors),
        )

        return current_df, result

    @classmethod
    def auto_detect_repairs(cls, df: pd.DataFrame) -> list[RepairAction]:
        """Auto-detect safe repair actions for a DataFrame.

        Only suggests deterministic, safe operations.

        Args:
            df: Source DataFrame.

        Returns:
            List of suggested RepairAction objects.
        """
        suggestions: list[RepairAction] = []

        # 0. Check for packed multi-value cells (e.g. Category | Amount delimited lists)
        from services.repair.structural import detect_multi_value_cells
        multi_val_info = detect_multi_value_cells(df)
        if multi_val_info and multi_val_info.get("confidence", 0) >= 0.85:
            sync_cols = multi_val_info.get("synchronized_columns", multi_val_info.get("multi_value_columns", []))
            delim = multi_val_info.get("delimiter", "|")
            suggestions.append(RepairAction(
                operation=RepairOperation.EXPLODE_MULTI_VALUE_CELLS,
                target=sync_cols,
                parameters={"delimiter": delim, "fill_mismatched": True},
                reason=f"Packed multi-record cells detected with delimiter '{delim}' across {len(sync_cols)} synchronized column(s): {', '.join(sync_cols)}",
                confidence=multi_val_info.get("confidence", 0.95),
                safe=True,
            ))

        # 1. Check for repeated headers in data
        if len(df) > 2:
            header_names_normalized = [str(c).strip().lower() for c in df.columns]
            has_repeated_hdr = False
            for idx in range(min(50, len(df))):
                row = df.iloc[idx]
                match_count = sum(1 for col_idx, col in enumerate(df.columns) if pd.notna(row[col]) and str(row[col]).strip().lower() == header_names_normalized[col_idx])
                if match_count >= 2 and match_count >= len(df.columns) * 0.6:
                    has_repeated_hdr = True
                    break
            if has_repeated_hdr:
                suggestions.append(RepairAction(
                    operation=RepairOperation.REMOVE_REPEATED_HEADERS,
                    reason="Embedded duplicate header rows detected inside data",
                    confidence=1.0,
                    safe=True,
                ))

        # 2. Check for duplicate rows
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            suggestions.append(RepairAction(
                operation=RepairOperation.REMOVE_DUPLICATE_ROWS,
                reason=f"{dup_count} duplicate rows detected",
                confidence=1.0,
                safe=True,
            ))

        # 3. Check for column name issues
        has_unnamed = any(c.startswith("Unnamed:") for c in df.columns)
        has_empty = any(c.strip() == "" for c in df.columns)
        col_names = list(df.columns)
        has_duplicates = len(col_names) != len(set(col_names))
        has_whitespace = any(c != c.strip() for c in df.columns)

        if has_unnamed or has_empty or has_duplicates or has_whitespace:
            reasons: list[str] = []
            if has_unnamed:
                reasons.append("unnamed columns")
            if has_empty:
                reasons.append("empty column names")
            if has_duplicates:
                reasons.append("duplicate column names")
            if has_whitespace:
                reasons.append("whitespace in column names")

            suggestions.append(RepairAction(
                operation=RepairOperation.RENAME_COLUMNS,
                reason=f"Column issues: {', '.join(reasons)}",
                confidence=1.0,
                safe=True,
            ))

        # 4. Check for empty rows
        empty_rows = df.isna().all(axis=1).sum()
        if empty_rows > 0:
            suggestions.append(RepairAction(
                operation=RepairOperation.DROP_EMPTY_ROWS,
                reason=f"{empty_rows} completely empty rows",
                confidence=1.0,
                safe=True,
            ))

        # 5. Check for empty columns
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            suggestions.append(RepairAction(
                operation=RepairOperation.DROP_EMPTY_COLUMNS,
                target=empty_cols,
                reason=f"{len(empty_cols)} completely empty column(s): {', '.join(empty_cols)}",
                confidence=1.0,
                safe=True,
            ))

        # 6. Check for excessive high-missing columns (>80% missing, but not 100%)
        if len(df) > 10:
            high_missing = [
                c for c in df.columns
                if (df[c].isna().sum() / len(df)) >= 0.8 and not df[c].isna().all()
            ]
            if high_missing:
                suggestions.append(RepairAction(
                    operation=RepairOperation.DROP_HIGH_MISSING_COLUMNS,
                    target=high_missing,
                    parameters={"threshold": 0.8},
                    reason=f"{len(high_missing)} column(s) with >80% missing data: {', '.join(high_missing)}",
                    confidence=0.9,
                    safe=True,
                ))

        # 7. Check for whitespace issues in cell values
        whitespace_detected = False
        for col in df.columns:
            if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
                continue
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            str_vals = non_null.astype(str)
            has_edge_ws = (str_vals != str_vals.str.strip()).any()
            has_multi_space = str_vals.str.contains(r"  +", regex=True).any()
            if has_edge_ws or has_multi_space:
                whitespace_detected = True
                break

        if whitespace_detected:
            suggestions.append(RepairAction(
                operation=RepairOperation.STRIP_WHITESPACE,
                reason="String columns contain leading/trailing or extra internal whitespace",
                confidence=1.0,
                safe=True,
            ))

        # 8. Check for inconsistent categorical values
        from services.repair.standardize import detect_inconsistent_columns
        inconsistent_cols = detect_inconsistent_columns(df)
        if inconsistent_cols:
            suggestions.append(RepairAction(
                operation=RepairOperation.STANDARDIZE_VALUES,
                target=inconsistent_cols,
                reason=f"{len(inconsistent_cols)} column(s) have inconsistent values (case/typo/format variants): {', '.join(inconsistent_cols[:5])}",
                confidence=0.9,
                safe=True,
            ))

        # 9. Check for object columns that should be numeric/datetime/boolean
        from services.repair.auto_dtypes import detect_mistyped_columns
        mistyped = detect_mistyped_columns(df)
        if mistyped:
            col_summaries = [f"{m['column']}->{m['suggested_type']}" for m in mistyped[:5]]
            suggestions.append(RepairAction(
                operation=RepairOperation.AUTO_DTYPES,
                target=[m["column"] for m in mistyped],
                reason=f"{len(mistyped)} column(s) stored as text but appear to be typed data: {', '.join(col_summaries)}",
                confidence=0.85,
                safe=True,
            ))

        # 10. Check for remaining missing values that can be auto-imputed
        remaining_missing = int(df.isna().sum().sum()) - (len(empty_cols) * len(df) if empty_cols else 0)
        if remaining_missing > 0:
            suggestions.append(RepairAction(
                operation=RepairOperation.FILL_MISSING,
                parameters={"strategy": "auto"},
                reason="Impute remaining missing values using dtype-aware strategy (median/mode)",
                confidence=0.85,
                safe=True,
            ))

        return suggestions

    @classmethod
    def generate_repair_preview(
        cls, df: pd.DataFrame, actions: list[RepairAction]
    ) -> dict[str, Any]:
        """Generate a complete before/after repair preview without modifying the dataset.

        Computes:
        - Affected columns, rows, and cell counts
        - Datatype modifications
        - Potential data loss detection
        - Itemized preview table rows
        """
        if df.empty or not actions:
            return {
                "columns_changed": [],
                "rows_affected": 0,
                "values_changed": 0,
                "datatype_changes": {},
                "potential_data_loss": 0,
                "preview_table": [],
            }

        simulated_df = df.copy()
        preview_rows: list[dict[str, Any]] = []
        columns_changed: set[str] = set()
        total_values_changed = 0
        total_loss = 0
        datatype_changes: dict[str, dict[str, str]] = {}

        for action in actions:
            op_name = action.operation.value
            target_cols = action.target or list(simulated_df.columns)

            try:
                temp_res, record = dispatch(simulated_df, action)

                # Track datatype changes
                for col in simulated_df.columns:
                    if col in temp_res.columns:
                        old_dt = str(simulated_df[col].dtype)
                        new_dt = str(temp_res[col].dtype)
                        if old_dt != new_dt:
                            datatype_changes[str(col)] = {"from": old_dt, "to": new_dt}
                            columns_changed.add(str(col))

                # Track null changes (potential data loss)
                for col in simulated_df.columns:
                    if col in temp_res.columns:
                        old_nulls = int(simulated_df[col].isna().sum())
                        new_nulls = int(temp_res[col].isna().sum())
                        if new_nulls > old_nulls and op_name not in ("drop_empty_rows", "drop_empty_columns"):
                            loss = new_nulls - old_nulls
                            total_loss += loss

                # Track row / value diffs
                if len(simulated_df) == len(temp_res) and set(simulated_df.columns) == set(temp_res.columns):
                    for col in simulated_df.columns:
                        try:
                            diff_mask = (simulated_df[col] != temp_res[col]) & ~(simulated_df[col].isna() & temp_res[col].isna())
                            num_diff = int(diff_mask.sum())
                            if num_diff > 0:
                                total_values_changed += num_diff
                                columns_changed.add(str(col))
                        except Exception:
                            pass

                # Extract preview table rows from record details or action
                conf = action.confidence
                risk = "safe" if conf >= 0.95 else ("medium" if conf >= 0.85 else "high")
                target_str = ", ".join(target_cols[:3]) if target_cols else "dataset"

                preview_rows.append({
                    "column": record.column or target_str,
                    "original_value": record.original_value or f"{record.rows_before} rows × {record.columns_before} cols",
                    "proposed_value": record.new_value or f"{record.rows_after} rows × {record.columns_after} cols",
                    "reason": action.reason,
                    "confidence": conf,
                    "risk_level": risk,
                    "method": record.method or op_name,
                    "status": record.status or "proposed",
                    "operation": op_name,
                })

                simulated_df = temp_res
            except Exception as e:
                logger.debug("Preview simulation error for '{}': {}", op_name, str(e))

        rows_affected = abs(len(df) - len(simulated_df))
        if rows_affected == 0 and total_values_changed > 0:
            rows_affected = min(len(df), total_values_changed)

        return {
            "columns_changed": sorted(list(columns_changed)),
            "rows_affected": rows_affected,
            "values_changed": total_values_changed,
            "datatype_changes": datatype_changes,
            "potential_data_loss": total_loss,
            "preview_table": preview_rows,
        }


