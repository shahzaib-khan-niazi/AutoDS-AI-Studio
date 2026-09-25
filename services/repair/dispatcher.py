"""Repair dispatcher — routes repair actions to the correct module."""

import pandas as pd
from models.repair import RepairAction, RepairOperation, RepairRecord
from core.logging import logger
from core.exceptions import RepairError
from services.repair.duplicates import remove_duplicate_rows
from services.repair.columns import (
    rename_columns,
    remove_specific_columns,
    remove_specific_rows,
    remove_rows_and_columns,
    replace_value_in_column,
)
from services.repair.missing import drop_empty_rows, drop_empty_columns, drop_high_missing_columns, fill_missing
from services.repair.types import convert_to_numeric, convert_to_datetime
from services.repair.structural import (
    unpivot,
    flatten_headers,
    explode_multi_value_cells,
    remove_repeated_headers,
    remove_metadata_rows,
    remove_spacer_rows_cols,
    unpivot_horizontal_category_blocks,
    remove_subtotal_elements,
    auto_reconstruct_structure,
    reconstruct_embedded_records,
)

from services.repair.outliers import handle_outliers
from services.repair.whitespace import strip_whitespace
from services.repair.standardize import standardize_values
from services.repair.auto_dtypes import auto_detect_dtypes
from services.repair.dates import normalize_date_column, DEFAULT_DISPLAY_FORMAT


def dispatch(
    df: pd.DataFrame, action: RepairAction
) -> tuple[pd.DataFrame, RepairRecord]:
    """Dispatch a repair action to the appropriate module.

    Args:
        df: Source DataFrame (not modified).
        action: The repair action to execute.

    Returns:
        Tuple of (repaired DataFrame, RepairRecord).

    Raises:
        RepairError: If the operation is unknown or fails.
    """
    operation = action.operation

    logger.info("Dispatching repair: {}", operation.value)

    try:
        if operation == RepairOperation.REMOVE_DUPLICATE_ROWS:
            return remove_duplicate_rows(df)

        elif operation == RepairOperation.RENAME_COLUMNS:
            return rename_columns(df)

        elif operation == RepairOperation.DROP_EMPTY_ROWS:
            return drop_empty_rows(df)

        elif operation == RepairOperation.DROP_EMPTY_COLUMNS:
            return drop_empty_columns(df)

        elif operation == RepairOperation.DROP_HIGH_MISSING_COLUMNS:
            threshold = float(action.parameters.get("threshold", 0.8))
            return drop_high_missing_columns(df, threshold=threshold)

        elif operation == RepairOperation.FILL_MISSING:
            strategy = str(action.parameters.get("strategy", "auto"))
            custom_val = action.parameters.get("custom_value")
            cols = action.target if action.target else None
            return fill_missing(df, strategy=strategy, columns=cols, custom_value=custom_val)

        elif operation == RepairOperation.HANDLE_OUTLIERS:
            method = str(action.parameters.get("method", "iqr"))
            factor = float(action.parameters.get("factor", 1.5))
            action_type = str(action.parameters.get("action", "cap"))
            cols = action.target if action.target else None
            return handle_outliers(df, columns=cols, method=method, factor=factor, action=action_type)

        elif operation == RepairOperation.CONVERT_TYPES:
            target_type = action.parameters.get("target_type", "numeric")
            columns = action.target
            if target_type == "datetime":
                return convert_to_datetime(df, columns)
            else:
                return convert_to_numeric(df, columns)

        elif operation == RepairOperation.EXPLODE_MULTI_VALUE_CELLS:
            cols = action.target if action.target else None
            delim = action.parameters.get("delimiter")
            fill_mismatch = bool(action.parameters.get("fill_mismatched", True))
            id_cols = action.parameters.get("id_columns")
            return explode_multi_value_cells(
                df,
                columns=cols,
                delimiter=delim,
                fill_mismatched=fill_mismatch,
                id_columns=id_cols,
            )

        elif operation == RepairOperation.REMOVE_REPEATED_HEADERS:
            return remove_repeated_headers(df)

        elif operation == RepairOperation.REMOVE_METADATA_ROWS:
            return remove_metadata_rows(df)

        elif operation == RepairOperation.UNPIVOT:
            return unpivot(
                df,
                id_columns=action.parameters.get("id_columns"),
                value_columns=action.parameters.get("value_columns"),
                var_name=action.parameters.get("var_name", "variable"),
                value_name=action.parameters.get("value_name", "value"),
            )

        elif operation == RepairOperation.FLATTEN_HEADERS:
            header_rows = action.parameters.get("header_rows", 1)
            return flatten_headers(df, header_rows=header_rows)

        elif operation == RepairOperation.STRIP_WHITESPACE:
            cols = action.target if action.target else None
            return strip_whitespace(df, columns=cols)

        elif operation == RepairOperation.STANDARDIZE_VALUES:
            cols = action.target if action.target else None
            extra_abbrev = action.parameters.get("extra_abbreviations")
            max_card = action.parameters.get("max_cardinality", 150)
            min_conf = float(action.parameters.get("min_confidence", 0.85))
            return standardize_values(
                df,
                columns=cols,
                max_cardinality=max_card,
                extra_abbreviations=extra_abbrev,
                min_confidence=min_conf,
            )

        elif operation == RepairOperation.AUTO_DTYPES:
            cols = action.target if action.target else None
            threshold = float(action.parameters.get("threshold", 0.85))
            return auto_detect_dtypes(df, columns=cols, threshold=threshold)

        elif operation == RepairOperation.NORMALIZE_DATES:
            target_cols = action.target if action.target else [c for c in df.columns if df[c].dtype == "object"]
            current_df = df.copy()
            last_record = None
            for col in target_cols:
                dayfirst_param = action.parameters.get("dayfirst")
                current_df, rec = normalize_date_column(
                    current_df,
                    column=col,
                    dayfirst=dayfirst_param,
                    display_format=DEFAULT_DISPLAY_FORMAT,
                )
                last_record = rec
            return current_df, last_record if last_record is not None else RepairRecord(operation="normalize_dates", success=True)

        elif operation == RepairOperation.REMOVE_SPACER_ROWS_COLS:
            return remove_spacer_rows_cols(df)

        elif operation == RepairOperation.UNPIVOT_HORIZONTAL_CATEGORY_BLOCKS:
            return unpivot_horizontal_category_blocks(df)

        elif operation == RepairOperation.REMOVE_SUBTOTAL_ELEMENTS:
            preserve_totals = bool(action.parameters.get("preserve_totals_if_records", True))
            return remove_subtotal_elements(df, preserve_totals_if_records=preserve_totals)

        elif operation == RepairOperation.AUTO_RECONSTRUCT_STRUCTURE:
            repaired_df, recs = auto_reconstruct_structure(df)
            last_rec = recs[-1] if recs else RepairRecord(operation="auto_reconstruct_structure", success=True)
            return repaired_df, last_rec

        elif operation == RepairOperation.RECONSTRUCT_EMBEDDED_RECORDS:
            labels = action.target or action.parameters.get("field_labels")
            return reconstruct_embedded_records(df, field_labels=labels)

        elif operation == RepairOperation.REMOVE_ROWS:
            row_indices = action.parameters.get("row_indices", [])
            cond_desc = action.parameters.get("condition_desc")
            reason = action.reason or "User removed specific rows"
            return remove_specific_rows(df, row_indices=row_indices, condition_desc=cond_desc, reason=reason)

        elif operation == RepairOperation.REMOVE_COLUMNS:
            cols = action.target or action.parameters.get("columns", [])
            reason = action.reason or "User removed specific columns"
            return remove_specific_columns(df, columns=cols, reason=reason)

        elif operation == RepairOperation.REMOVE_ROWS_AND_COLUMNS:
            row_indices = action.parameters.get("row_indices", [])
            cols = action.target or action.parameters.get("columns", [])
            cond_desc = action.parameters.get("condition_desc")
            reason = action.reason or "User removed specific rows and columns"
            return remove_rows_and_columns(df, row_indices=row_indices, columns=cols, condition_desc=cond_desc, reason=reason)

        elif operation == RepairOperation.REPLACE_VALUES:
            col = action.parameters.get("column") or (action.target[0] if action.target else None)
            old_val = action.parameters.get("old_value", "")
            new_val = action.parameters.get("new_value", "")
            reason = action.reason or f"User replaced '{old_val}' with '{new_val if new_val != '' else '[EMPTY]'}' in column '{col}'"
            return replace_value_in_column(df, column=col, old_value=old_val, new_value=new_val, reason=reason)

        else:
            raise RepairError(
                message=f"Unknown repair operation: {operation.value}",
            )

    except RepairError:
        raise
    except Exception as e:
        raise RepairError(
            message=f"Repair operation '{operation.value}' failed",
            details=str(e),
        )
