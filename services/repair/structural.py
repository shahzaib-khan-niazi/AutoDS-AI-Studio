"""Structural repair module for AutoDS AI Studio.

Handles generalized structural transformations:
1. Multi-value cell detection & synchronized/single-column record exploding
2. Repeated header row removal inside data
3. Summary & metadata row removal
4. Unpivot (wide-to-long)
5. Multi-level header flattening
"""

import re
from datetime import datetime
from typing import Any, Optional
import numpy as np
import pandas as pd

from core.logging import logger
from models.repair import RepairRecord


# Candidate delimiters for multi-value cells
CANDIDATE_DELIMITERS = ["|", ";", "\n", "\r\n", " / ", "•", " ~ ", "~"]


def detect_multi_value_cells(df: pd.DataFrame) -> dict[str, Any]:
    """Detect columns containing multiple values packed into single cells.

    Analyzes candidate delimiters across string/object columns. Checks:
    1. Delimiter occurrence frequency across rows.
    2. Synchronized item counts across multiple columns (e.g. Category & Amount both have N items).
    3. Exclusion of normal sentences, decimals, or standard dates.

    Args:
        df: Source DataFrame.

    Returns:
        Dictionary with detection details and parameters, or empty dict if not detected.
    """
    if df.empty or len(df.columns) < 2:
        return {}

    str_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    if not str_cols:
        return {}

    best_detection: dict[str, Any] = {}
    best_score = 0.0

    for delim in CANDIDATE_DELIMITERS:
        col_multi_stats: dict[str, dict[str, Any]] = {}

        for col in str_cols:
            series = df[col].dropna().astype(str)
            if len(series) == 0:
                continue

            # Count rows that contain the delimiter
            has_delim = series.str.contains(re.escape(delim), regex=True)
            delim_rows_count = has_delim.sum()
            delim_ratio = delim_rows_count / len(series)

            if delim_ratio >= 0.25:  # At least 25% of rows contain delimiter
                # Calculate item count per cell
                item_counts = series.apply(
                    lambda x: len([t for t in str(x).split(delim) if t.strip()]) if delim in str(x) else 1
                )
                max_items = int(item_counts.max())
                avg_items = item_counts.mean()

                if max_items > 1:
                    col_multi_stats[col] = {
                        "delim_ratio": delim_ratio,
                        "item_counts": item_counts,
                        "max_items": max_items,
                        "avg_items": avg_items,
                    }

        if not col_multi_stats:
            continue

        multi_cols = list(col_multi_stats.keys())

        # Check for synchronized multi-value columns (matching item counts per row)
        synchronized_groups: list[list[str]] = []
        is_synchronized = False

        if len(multi_cols) >= 2:
            sync_cols: list[str] = [multi_cols[0]]
            base_counts = col_multi_stats[multi_cols[0]]["item_counts"]

            for other_col in multi_cols[1:]:
                other_counts = col_multi_stats[other_col]["item_counts"]
                common_idx = base_counts.index.intersection(other_counts.index)
                if len(common_idx) > 0:
                    matching = (base_counts.loc[common_idx] == other_counts.loc[common_idx]).sum()
                    match_ratio = matching / len(common_idx)
                    if match_ratio >= 0.75:
                        sync_cols.append(other_col)

            if len(sync_cols) >= 2:
                synchronized_groups.append(sync_cols)
                is_synchronized = True

        score = (0.95 if is_synchronized else 0.80) * (len(col_multi_stats) / max(len(str_cols), 1))

        if score > best_score:
            best_score = score
            id_cols = [c for c in df.columns if c not in multi_cols]
            best_detection = {
                "delimiter": delim,
                "multi_value_columns": multi_cols,
                "synchronized_columns": synchronized_groups[0] if synchronized_groups else multi_cols,
                "id_columns": id_cols,
                "is_synchronized": is_synchronized,
                "confidence": min(0.98 if is_synchronized else 0.85, score + 0.5),
            }

    return best_detection


def explode_multi_value_cells(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    delimiter: Optional[str] = None,
    fill_mismatched: bool = True,
    id_columns: Optional[list[str]] = None,
    target_columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, RepairRecord]:
    """Explode multi-value cells into separate relational rows.

    Supports synchronized multi-column exploding (e.g. Category list & Amount list exploded together),
    preserving single-valued columns (e.g. Order ID, Customer Name) across the generated rows.

    Args:
        df: Source DataFrame.
        columns: Multi-value columns to explode. If None, auto-detects.
        delimiter: Delimiter to split by. If None, auto-detects.
        fill_mismatched: If True, pads mismatched list lengths with None; otherwise raises warning.
        id_columns: Columns to replicate across rows. If None, uses all non-exploded columns.
        target_columns: Alias for columns parameter.

    Returns:
        Tuple of (repaired DataFrame, RepairRecord).
    """
    rows_before = len(df)
    cols_before = len(df.columns)

    columns = columns or target_columns

    if df.empty:
        return df.copy(), RepairRecord(
            operation="explode_multi_value_cells",
            rows_before=0,
            rows_after=0,
            columns_before=cols_before,
            columns_after=cols_before,
            success=True,
        )

    # Auto-detect if parameters not provided
    if columns is None:
        detection = detect_multi_value_cells(df)
        if not detection:
            logger.info("No multi-value cells detected for exploding.")
            return df.copy(), RepairRecord(
                operation="explode_multi_value_cells",
                rows_before=rows_before,
                rows_after=rows_before,
                columns_before=cols_before,
                columns_after=cols_before,
                success=True,
                warnings=["No multi-value delimiter cells detected."],
            )
        columns = detection.get("synchronized_columns", detection.get("multi_value_columns", []))
        delimiter = delimiter or detection.get("delimiter", "|")

    # Validate target columns
    if columns is None:
        columns = []
    target_cols = [c for c in columns if c in df.columns]
    if not target_cols:
        return df.copy(), RepairRecord(
            operation="explode_multi_value_cells",
            rows_before=rows_before,
            rows_after=rows_before,
            columns_before=cols_before,
            columns_after=cols_before,
            success=False,
            warnings=["None of the specified columns exist in dataset."],
        )

    if delimiter is None:
        for cand in CANDIDATE_DELIMITERS:
            if any(df[c].dropna().astype(str).str.contains(re.escape(cand), regex=True).any() for c in target_cols):
                delimiter = cand
                break
        delimiter = delimiter or "|"

    if id_columns is None:
        id_cols = [c for c in df.columns if c not in target_cols]
    else:
        id_cols = [c for c in id_columns if c in df.columns]

    new_rows: list[dict[str, Any]] = []
    mismatched_rows = 0
    warnings: list[str] = []

    for idx, row in df.iterrows():
        # Tokenize target columns
        token_lists: dict[str, list[str]] = {}
        for col in target_cols:
            val = row[col]
            if pd.isna(val) or val is None or str(val).strip() == "" or str(val) == "nan":
                token_lists[col] = [""]
            else:
                str_val = str(val)
                if delimiter in str_val:
                    tokens = [t.strip() for t in str_val.split(delimiter)]
                    token_lists[col] = tokens if tokens else [""]
                else:
                    token_lists[col] = [str_val.strip()]

        # Determine row expansion length
        lengths = [len(t_list) for t_list in token_lists.values()]
        max_len = max(lengths) if lengths else 1
        min_len = min(lengths) if lengths else 1

        if max_len != min_len:
            mismatched_rows += 1
            if not fill_mismatched:
                warnings.append(f"Row {idx} has mismatched token lengths: {dict(zip(target_cols, lengths))}")

        # Construct expanded rows
        for i in range(max_len):
            row_dict: dict[str, Any] = {}

            # Replicate ID / single-value columns
            for id_c in id_cols:
                row_dict[id_c] = row[id_c]

            # Assign i-th token for multi-value columns
            for col in target_cols:
                t_list = token_lists[col]
                if i < len(t_list):
                    row_dict[col] = t_list[i] if t_list[i] != "" else np.nan
                else:
                    row_dict[col] = np.nan

            new_rows.append(row_dict)

    # Build resulting DataFrame
    ordered_cols = [c for c in df.columns if c in id_cols or c in target_cols]
    result_df = pd.DataFrame(new_rows)
    if not result_df.empty:
        result_df = result_df[ordered_cols]
    else:
        result_df = pd.DataFrame(columns=ordered_cols)

    # Post-explosion type inference for exploded columns
    for col in target_cols:
        if col in result_df.columns:
            non_null = result_df[col].dropna().astype(str)
            if len(non_null) > 0:
                # Check if purely numeric
                cleaned_num = non_null.str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
                try:
                    pd.to_numeric(cleaned_num, errors="raise")
                    result_df[col] = pd.to_numeric(
                        result_df[col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip(),
                        errors="coerce"
                    )
                except Exception:
                    pass

    rows_after = len(result_df)
    cols_after = len(result_df.columns)

    if mismatched_rows > 0:
        warnings.append(f"{mismatched_rows} row(s) had mismatched token list lengths and were padded safely.")

    logger.info(
        "Explode multi-value cells: {}×{} -> {}×{} (columns: {}, delimiter: '{}')",
        rows_before, cols_before, rows_after, cols_after, target_cols, delimiter
    )

    record = RepairRecord(
        operation="explode_multi_value_cells",
        timestamp=datetime.now(),
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=cols_before,
        columns_after=cols_after,
        success=True,
        warnings=warnings,
        details={
            "exploded_columns": target_cols,
            "delimiter": delimiter,
            "mismatched_rows": mismatched_rows,
            "original_rows": rows_before,
            "expanded_rows": rows_after,
        },
    )

    return result_df, record


def remove_repeated_headers(df: pd.DataFrame) -> tuple[pd.DataFrame, RepairRecord]:
    """Detect and remove repeated header rows appearing inside the data body.

    Args:
        df: Source DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, RepairRecord).
    """
    rows_before = len(df)
    cols_before = len(df.columns)

    if rows_before <= 1 or cols_before == 0:
        return df.copy(), RepairRecord(
            operation="remove_repeated_headers",
            rows_before=rows_before,
            rows_after=rows_before,
            columns_before=cols_before,
            columns_after=cols_before,
            success=True,
        )

    header_names_normalized = [c.strip().lower() for c in df.columns]
    rows_to_drop: list[Any] = []

    for idx, row in df.iterrows():
        match_count = 0
        non_null_count = 0
        for col_idx, col in enumerate(df.columns):
            val = row[col]
            if pd.notna(val) and str(val).strip() != "" and str(val) != "nan":
                non_null_count += 1
                val_norm = str(val).strip().lower()
                if val_norm == header_names_normalized[col_idx]:
                    match_count += 1

        if non_null_count > 0:
            match_ratio = match_count / non_null_count
            if match_ratio >= 0.6 and match_count >= 2:
                rows_to_drop.append(idx)

    result_df = df.drop(index=rows_to_drop).reset_index(drop=True)
    rows_after = len(result_df)
    cols_after = len(result_df.columns)

    logger.info("Removed {} repeated header row(s)", len(rows_to_drop))

    record = RepairRecord(
        operation="remove_repeated_headers",
        timestamp=datetime.now(),
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=cols_before,
        columns_after=cols_after,
        success=True,
        details={"dropped_indices": rows_to_drop, "dropped_count": len(rows_to_drop)},
    )

    return result_df, record


def remove_metadata_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, RepairRecord]:
    """Detect and remove summary/total rows or report header/metadata banners.

    Args:
        df: Source DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, RepairRecord).
    """
    rows_before = len(df)
    cols_before = len(df.columns)

    if rows_before <= 1 or cols_before == 0:
        return df.copy(), RepairRecord(
            operation="remove_metadata_rows",
            rows_before=rows_before,
            rows_after=rows_before,
            columns_before=cols_before,
            columns_after=cols_before,
            success=True,
        )

    summary_keywords = [
        "total",
        "grand total",
        "subtotal",
        "summary",
        "total summary",
        "report generated",
        "page 1 of",
        "confidential",
        "source:",
        "source",
        "note:",
        "note",
        "notes:",
        "notes",
        "disclaimer",
    ]
    rows_to_drop: list[Any] = []

    # Check first column and entire row text
    for idx, row in df.iterrows():
        first_val = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ""
        if any(first_val.startswith(kw) or first_val == kw for kw in summary_keywords):
            rows_to_drop.append(idx)

    result_df = df.drop(index=rows_to_drop).reset_index(drop=True)
    rows_after = len(result_df)
    cols_after = len(result_df.columns)

    logger.info("Removed {} metadata/summary row(s)", len(rows_to_drop))

    record = RepairRecord(
        operation="remove_metadata_rows",
        timestamp=datetime.now(),
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=cols_before,
        columns_after=cols_after,
        success=True,
        details={"dropped_indices": rows_to_drop, "dropped_count": len(rows_to_drop)},
    )

    return result_df, record


def unpivot(
    df: pd.DataFrame,
    id_columns: Optional[list[str]] = None,
    value_columns: Optional[list[str]] = None,
    var_name: str = "variable",
    value_name: str = "value",
    split_delimiter: Optional[str] = " - ",
    drop_null_values: bool = True,
    split_col_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, RepairRecord]:
    """Unpivot (melt) a wide DataFrame to long format.

    Args:
        df: Source DataFrame (not modified).
        id_columns: Columns to keep as identifiers. If None, auto-detects.
        value_columns: Columns to unpivot. If None, uses all non-id columns.
        var_name: Name for the variable column.
        value_name: Name for the value column.
        split_delimiter: If present, splits multi-level column names into distinct features.
        drop_null_values: If True, drops rows where unpivoted value is NaN.
        split_col_names: Optional names for the split columns (e.g. ['Ship Mode', 'Segment']).

    Returns:
        Tuple of (melted DataFrame, RepairRecord).
    """
    result = df.copy()

    # Auto-detect id columns if not specified
    if id_columns is None:
        first_col = result.columns[0]
        id_columns = [first_col]

    # Filter to columns that actually exist
    id_columns = [c for c in id_columns if c in result.columns]

    if value_columns is None:
        value_columns = [c for c in result.columns if c not in id_columns]

    rows_before = len(result)
    cols_before = len(result.columns)

    result = pd.melt(
        result,
        id_vars=id_columns if id_columns else None,
        value_vars=value_columns,
        var_name=var_name,
        value_name=value_name,
    )

    # Drop null values in sparse tables
    if drop_null_values:
        result = result.dropna(subset=[value_name]).reset_index(drop=True)

    # Split hierarchical variable names (e.g., 'First Class - Consumer')
    if split_delimiter and var_name in result.columns:
        sample_val = str(result[var_name].iloc[0]) if len(result) > 0 else ""
        if split_delimiter in sample_val:
            parts = sample_val.split(split_delimiter)
            if len(parts) == 2:
                col_1 = split_col_names[0] if split_col_names and len(split_col_names) > 0 else "Category_1"
                col_2 = split_col_names[1] if split_col_names and len(split_col_names) > 1 else "Category_2"
                result[[col_1, col_2]] = result[var_name].str.split(split_delimiter, expand=True)
                result = result.drop(columns=[var_name])

    # Convert types if obvious
    for id_c in id_columns:
        if id_c in result.columns:
            try:
                converted = pd.to_datetime(result[id_c], errors="raise")
                result[id_c] = converted
            except Exception:
                pass

    if value_name in result.columns:
        try:
            result[value_name] = pd.to_numeric(result[value_name], errors="coerce")
        except Exception:
            pass

    logger.info(
        "Unpivot: {}×{} -> {}×{} (id_cols: {}, value_cols: {})",
        rows_before, cols_before,
        len(result), len(result.columns),
        id_columns, len(value_columns),
    )

    record = RepairRecord(
        operation="unpivot",
        timestamp=datetime.now(),
        rows_before=rows_before,
        rows_after=len(result),
        columns_before=cols_before,
        columns_after=len(result.columns),
        success=True,
        details={
            "id_columns": id_columns,
            "value_columns_count": len(value_columns),
            "var_name": var_name,
            "value_name": value_name,
        },
    )

    return result, record


def flatten_headers(
    df: pd.DataFrame, header_rows: int = 1
) -> tuple[pd.DataFrame, RepairRecord]:
    """Flatten multi-level header rows by forward-filling merged headers and combining sub-rows.

    Args:
        df: Source DataFrame (not modified).
        header_rows: Number of rows to use as headers (default 1).

    Returns:
        Tuple of (DataFrame with flattened headers, RepairRecord).
    """
    result = df.copy()
    rows_before = len(result)

    if header_rows < 1 or header_rows >= len(result):
        record = RepairRecord(
            operation="flatten_headers",
            timestamp=datetime.now(),
            rows_before=rows_before,
            rows_after=rows_before,
            columns_before=len(result.columns),
            columns_after=len(result.columns),
            success=False,
            warnings=["Invalid header_rows count"],
        )
        return result, record

    # 1. Forward-fill merged top headers (e.g. First Class, Unnamed: 2 -> First Class, First Class)
    top_headers = [str(c) for c in result.columns]
    has_unnamed = any(c.startswith("Unnamed:") for c in top_headers)

    if has_unnamed:
        filled_top: list[str] = []
        current = top_headers[0]
        for col in top_headers:
            if col.startswith("Unnamed:"):
                filled_top.append(current)
            else:
                current = col
                filled_top.append(col)
        top_headers = filled_top

    # 2. Combine with row 0
    row0 = list(result.iloc[0])
    combined: list[str] = []
    for i in range(len(top_headers)):
        top = top_headers[i].strip()
        sub = str(row0[i]).strip() if pd.notna(row0[i]) and str(row0[i]) != "nan" else ""
        
        is_top_generic = top.isdigit() or top.startswith("Unnamed:") or top == ""
        
        if is_top_generic:
            combined.append(sub if sub else f"col_{i}")
        else:
            if sub and not sub.isdigit() and not sub.startswith("Unnamed:") and sub != top:
                combined.append(f"{top} - {sub}")
            else:
                combined.append(top)

    # 3. Check if row 1 is a single-column label row
    rows_to_drop = header_rows
    if len(result) > 2 and header_rows == 1:
        row1 = list(result.iloc[1])
        non_null_count = sum(1 for v in row1 if pd.notna(v) and str(v) != "nan")
        if non_null_count == 1 and pd.notna(row1[0]):
            col0_label = str(row1[0]).strip()
            if len(col0_label) < 40 and not col0_label.startswith("Unnamed"):
                combined[0] = col0_label
                rows_to_drop = 2

    result.columns = pd.Index(combined)
    result = result.iloc[rows_to_drop:].reset_index(drop=True)

    logger.info("Flattened {} header row(s), new columns: {}", rows_to_drop, combined[:5])

    record = RepairRecord(
        operation="flatten_headers",
        timestamp=datetime.now(),
        rows_before=rows_before,
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        details={"header_rows_used": rows_to_drop, "new_column_names": combined},
    )

    return result, record


def remove_spacer_rows_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, RepairRecord]:
    """Remove empty spacer rows and columns safely without mutating original.

    Args:
        df: Source DataFrame.

    Returns:
        Tuple of (repaired DataFrame copy, RepairRecord).
    """
    result = df.copy()
    rows_before = len(result)
    cols_before = len(result.columns)

    # Completely empty columns
    empty_cols = [c for c in result.columns if result[c].isna().all()]
    # Completely empty rows
    empty_rows_idx = list(result.index[result.isna().all(axis=1)])

    if empty_cols:
        result = result.drop(columns=empty_cols)
    if empty_rows_idx:
        result = result.drop(index=empty_rows_idx).reset_index(drop=True)

    rows_after = len(result)
    cols_after = len(result.columns)
    total_cells = rows_before * cols_before
    retained_cells = rows_after * cols_after
    discarded_cells = total_cells - retained_cells

    record = RepairRecord(
        operation="remove_spacer_rows_cols",
        timestamp=datetime.now(),
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=cols_before,
        columns_after=cols_after,
        success=True,
        details={
            "empty_columns_dropped": list(empty_cols),
            "empty_rows_dropped_count": len(empty_rows_idx),
            "source_cells_considered": total_cells,
            "source_cells_retained": retained_cells,
            "source_cells_discarded": discarded_cells,
            "discarded_reasons": {"blank_spacer_cells": discarded_cells},
        },
    )

    return result, record


def remove_subtotal_elements(
    df: pd.DataFrame,
    preserve_totals_if_records: bool = True,
) -> tuple[pd.DataFrame, RepairRecord]:
    """Detect and remove subtotal/summary columns and rows while preserving record integrity.

    Args:
        df: Source DataFrame.
        preserve_totals_if_records: If True, only drops subtotals when proven redundant.

    Returns:
        Tuple of (repaired DataFrame copy, RepairRecord).
    """
    result = df.copy()
    rows_before = len(result)
    cols_before = len(result.columns)

    kw_list = ["total", "subtotal", "summary", "grand total", "average"]

    # Identify subtotal columns
    subtotal_cols = [c for c in result.columns if any(kw in c.lower() for kw in kw_list)]
    
    # Identify subtotal rows
    subtotal_rows_idx: list[int] = []
    if len(result) > 0 and len(result.columns) > 0:
        first_col_str = result.iloc[:, 0].dropna().astype(str).str.lower()
        subtotal_rows_idx = list(first_col_str[first_col_str.str.contains("|".join(kw_list), regex=True)].index)

    if subtotal_cols:
        result = result.drop(columns=subtotal_cols)
    if subtotal_rows_idx:
        result = result.drop(index=subtotal_rows_idx).reset_index(drop=True)

    rows_after = len(result)
    cols_after = len(result.columns)
    total_cells = rows_before * cols_before
    retained_cells = rows_after * cols_after
    discarded_cells = total_cells - retained_cells

    record = RepairRecord(
        operation="remove_subtotal_elements",
        timestamp=datetime.now(),
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=cols_before,
        columns_after=cols_after,
        success=True,
        details={
            "subtotal_columns_dropped": list(subtotal_cols),
            "subtotal_rows_dropped": subtotal_rows_idx,
            "source_cells_considered": total_cells,
            "source_cells_retained": retained_cells,
            "source_cells_discarded": discarded_cells,
            "discarded_reasons": {"redundant_subtotal_cells": discarded_cells},
        },
    )

    return result, record


def unpivot_horizontal_category_blocks(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, RepairRecord]:
    """Transform horizontally distributed repeated category blocks into clean long/tidy records.

    Handles messy layouts with repeating category groups (e.g. Consumer, Corporate, Home Office)
    spread across columns alongside subtotal columns.

    Args:
        df: Source DataFrame (never modified).

    Returns:
        Tuple of (repaired DataFrame copy, RepairRecord).
    """
    result = df.copy()
    rows_before = len(result)
    cols_before = len(result.columns)

    if rows_before == 0 or cols_before < 2:
        return result, RepairRecord(
            operation="unpivot_horizontal_category_blocks",
            timestamp=datetime.now(),
            rows_before=rows_before,
            rows_after=rows_before,
            columns_before=cols_before,
            columns_after=cols_before,
            success=True,
        )

    top_cols = list(result.columns)

    # Forward fill top headers across Unnamed: columns
    filled_top: list[str] = []
    curr = top_cols[0]
    for c in top_cols:
        if c.startswith("Unnamed:") or c.strip() == "":
            filled_top.append(curr)
        else:
            curr = c
            filled_top.append(c)

    # Check row 0 sub-headers
    row0_vals = list(result.iloc[0])
    has_subheaders = any(pd.notna(v) and str(v).strip() != "" for v in row0_vals)

    # Extract ID columns (e.g. Segment >>, First column if unique)
    id_col = top_cols[0] if not top_cols[0].startswith("Unnamed:") else "ID"

    # Identify category groups from filled top headers
    kw_subtotals = ["total", "subtotal", "summary", "grand total"]
    
    # Filter out pure subtotal columns
    valid_col_indices = [
        i for i, c in enumerate(top_cols)
        if not any(kw in c.lower() for kw in kw_subtotals)
        and not any(kw in str(row0_vals[i]).lower() for kw in kw_subtotals)
    ]

    # Combine top header and row 0 subheader
    combined_names: list[str] = []
    for i in range(cols_before):
        t = filled_top[i].strip()
        sub = str(row0_vals[i]).strip() if i < len(row0_vals) and pd.notna(row0_vals[i]) else ""
        
        # Clean up 'Segment >>' or similar prefix
        t_clean = re.sub(r"[>\:\;]+", "", t).strip()
        
        if sub and sub.lower() != t_clean.lower() and not sub.startswith("Unnamed:"):
            combined_names.append(f"{t_clean}__{sub}")
        else:
            combined_names.append(t_clean)

    result.columns = pd.Index(combined_names)

    # Drop row 0 if it was used as subheader
    if has_subheaders:
        result = result.iloc[1:].reset_index(drop=True)

    # Drop pure subtotal columns
    cols_to_keep = [combined_names[i] for i in valid_col_indices if i < len(combined_names)]
    result = result[[c for c in cols_to_keep if c in result.columns]]

    # Ensure unique column names
    seen: dict[str, int] = {}
    final_cols: list[str] = []
    for col in result.columns:
        clean_c = col.strip()
        if clean_c in seen:
            seen[clean_c] += 1
            final_cols.append(f"{clean_c}_{seen[clean_c]}")
        else:
            seen[clean_c] = 0
            final_cols.append(clean_c)

    result.columns = pd.Index(final_cols)

    rows_after = len(result)
    cols_after = len(result.columns)
    total_cells = rows_before * cols_before
    retained_cells = rows_after * cols_after
    discarded_cells = total_cells - retained_cells

    logger.info("Unpivoted horizontal category blocks: {}x{} -> {}x{}", rows_before, cols_before, rows_after, cols_after)

    record = RepairRecord(
        operation="unpivot_horizontal_category_blocks",
        timestamp=datetime.now(),
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=cols_before,
        columns_after=cols_after,
        success=True,
        details={
            "source_cells_considered": total_cells,
            "source_cells_retained": retained_cells,
            "source_cells_discarded": discarded_cells,
            "discarded_reasons": {"unpivoted_redundant_subtotals": discarded_cells},
            "new_columns": final_cols,
        },
    )

    return result, record


def auto_reconstruct_structure(df: pd.DataFrame) -> tuple[pd.DataFrame, list[RepairRecord]]:
    """Automatically reconstruct dataset structure using safe deterministic plan.

    Args:
        df: Source DataFrame (never modified).

    Returns:
        Tuple of (repaired DataFrame copy, list of RepairRecord).
    """
    from services.structure.planner import StructuralPlanner

    plan = StructuralPlanner.create_plan(df)
    records: list[RepairRecord] = []
    current_df = df.copy()

    # Safety check: automatic execution is allowed ONLY when confidence >= 0.85, no ambiguities exist, and approval is not required
    if plan.requires_human_approval or plan.confidence < 0.85 or len(plan.ambiguity_flags) > 0:
        logger.info("Structural reconstruction blocked automatic execution due to safety controls (confidence: {:.2f}, approval_required: {}, ambiguities: {})", plan.confidence, plan.requires_human_approval, len(plan.ambiguity_flags))
        return current_df, records

    if plan.proposed_transformation == "remove_spacer_rows_cols":
        current_df, rec = remove_spacer_rows_cols(current_df)
        records.append(rec)
    elif plan.proposed_transformation == "flatten_multi_headers":
        current_df, rec = flatten_headers(current_df, header_rows=1)
        records.append(rec)
    elif plan.proposed_transformation == "unpivot_horizontal_category_blocks":
        current_df, rec = unpivot_horizontal_category_blocks(current_df)
        records.append(rec)
    elif plan.proposed_transformation == "remove_subtotal_elements":
        current_df, rec = remove_subtotal_elements(current_df)
        records.append(rec)
    elif plan.proposed_transformation == "reconstruct_embedded_records":
        current_df, rec = reconstruct_embedded_records(current_df)
        records.append(rec)

    return current_df, records


def reconstruct_embedded_records(
    df: pd.DataFrame, field_labels: Optional[list[str]] = None
) -> tuple[pd.DataFrame, RepairRecord]:
    """Reconstruct embedded text records into a multi-column DataFrame.

    Args:
        df: Source DataFrame (not modified).
        field_labels: Optional explicit list of field labels.

    Returns:
        Tuple of (reconstructed DataFrame, RepairRecord).
    """
    from services.structure.embedded_records import EmbeddedRecordAnalyzer

    rows_before = len(df)
    cols_before = len(df.columns)

    reconstructed_df, audit_info = EmbeddedRecordAnalyzer.reconstruct_dataframe(df, custom_labels=field_labels)

    rows_after = len(reconstructed_df)
    cols_after = len(reconstructed_df.columns)

    record = RepairRecord(
        operation="reconstruct_embedded_records",
        timestamp=datetime.now(),
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=cols_before,
        columns_after=cols_after,
        success=audit_info.get("success", True),
        warnings=audit_info.get("ambiguity_flags", []),
        details={
            "discovered_labels": audit_info.get("labels", []),
            "confidence": audit_info.get("confidence", 0.0),
            "source_shape": audit_info.get("source_shape", (rows_before, cols_before)),
            "target_shape": audit_info.get("target_shape", (rows_after, cols_after)),
            "source_cells_considered": audit_info.get("source_cells_considered", rows_before * cols_before),
            "source_cells_retained": audit_info.get("source_cells_retained", rows_after * cols_after),
            "source_cells_discarded": audit_info.get("source_cells_discarded", 0),
        },
    )

    return reconstructed_df, record


