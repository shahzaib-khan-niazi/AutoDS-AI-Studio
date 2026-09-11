"""Column name repair and management module.

Handles:
- Manual single column rename with strict validation and audit trail
- Standardization with selectable strategies (snake_case, lowercase, whitespace, special chars, underscores)
- Duplicate column detection and disambiguation
- Unnamed column classification (index vs empty vs data)
- Audit log tracking for all operations
- Deterministic column cleanup (backward compatible)
"""

import re
from collections import Counter
from datetime import datetime
from typing import Any, Optional
import numpy as np
import pandas as pd
from models.repair import RepairRecord
from core.logging import logger
from core.exceptions import RepairError, RepairValidationError


def validate_column_rename(
    df: pd.DataFrame, old_name: str, new_name: str
) -> tuple[bool, Optional[str]]:
    """Validate whether renaming old_name -> new_name is safe and valid.

    Rules:
    - old_name must exist in df.columns
    - new_name cannot be empty or whitespace only
    - new_name cannot already exist in df.columns (unless it's unchanged)
    - new_name must be a valid string

    Args:
        df: Source DataFrame.
        old_name: Current column name.
        new_name: Target column name.

    Returns:
        Tuple of (is_valid, error_message_if_invalid).
    """
    col_names = [f"{c}" for c in df.columns]

    if old_name not in col_names:
        return False, f"Column '{old_name}' does not exist in dataset."

    if not isinstance(new_name, str) or not new_name.strip():
        return False, "New column name cannot be empty."

    cleaned_new = new_name.strip()
    if cleaned_new != old_name and cleaned_new in col_names:
        return False, f"Column '{cleaned_new}' already exists in dataset."

    return True, None


def rename_single_column(
    df: pd.DataFrame,
    old_name: str,
    new_name: str,
    method: str = "manual",
    confidence: float = 1.0,
) -> tuple[pd.DataFrame, RepairRecord]:
    """Rename a single DataFrame column safely without mutating the original.

    Preserves row count, column count, column data values, and data types.

    Args:
        df: Source DataFrame.
        old_name: Existing column name.
        new_name: Proposed new column name.
        method: Method used ('manual', 'ai_suggestion', etc.).
        confidence: Confidence score of the rename (1.0 for manual).

    Returns:
        Tuple of (repaired DataFrame, RepairRecord).

    Raises:
        RepairValidationError: If validation fails.
    """
    is_valid, err = validate_column_rename(df, old_name, new_name)
    if not is_valid:
        raise RepairValidationError(message=err or "Invalid column rename")

    cleaned_new = new_name.strip()
    result = df.copy()

    # Reconstruct columns list to handle potential duplicate or object headers
    new_cols = []
    renamed = False
    for col in result.columns:
        if col == old_name and not renamed:
            new_cols.append(cleaned_new)
            renamed = True
        else:
            new_cols.append(col)

    result.columns = pd.Index(new_cols)

    record = RepairRecord(
        operation="column_rename",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        details={
            "repair_type": "column_rename",
            "old_name": old_name,
            "new_name": cleaned_new,
            "method": method,
            "confidence": confidence,
            "status": "applied",
        },
    )

    logger.info("Renamed column '{}' -> '{}' via {}", old_name, cleaned_new, method)
    return result, record


def standardize_name(name: str, strategies: list[str]) -> str:
    """Standardize a single column name according to selectable strategies.

    Supported strategies:
    - 'snake_case': converts camelCase, replaces special chars/spaces with _, collapses __, lowercases
    - 'lowercase': converts to lower case
    - 'remove_whitespace': strips leading/trailing whitespace
    - 'replace_spaces': converts internal whitespace to underscores
    - 'remove_special_chars': strips symbols/punctuation except alphanumeric and underscores
    - 'normalize_underscores': collapses repeated underscores and strips leading/trailing underscores

    Args:
        name: Original column name.
        strategies: List of strategy keys.

    Returns:
        Standardized column name.
    """
    s = name

    if "snake_case" in strategies:
        # 1. Insert underscore between camelCase/PascalCase boundaries
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
        # 2. Replace hyphens, slashes, and spaces with underscores
        s = re.sub(r"[\s\-\/\\]+", "_", s)
        # 3. Strip non-alphanumeric chars except underscores
        s = re.sub(r"[^a-zA-Z0-9_]+", "", s)
        # 4. Collapse repeated underscores
        s = re.sub(r"_+", "_", s)
        # 5. Strip edge underscores and whitespace
        s = s.strip("_").strip()
        # 6. Lowercase
        s = s.lower()
        return s or "column"

    # Process individual strategies if snake_case wasn't selected
    if "remove_whitespace" in strategies:
        s = s.strip()

    if "replace_spaces" in strategies:
        s = re.sub(r"\s+", "_", s)

    if "remove_special_chars" in strategies:
        # Keep alphanumeric, spaces, and underscores
        s = re.sub(r"[^a-zA-Z0-9\s_]", "", s)

    if "normalize_underscores" in strategies:
        s = re.sub(r"_+", "_", s).strip("_")

    if "lowercase" in strategies:
        s = s.lower()

    return s.strip() or "column"


def get_column_name_standardizations(
    df: pd.DataFrame, strategies: list[str]
) -> list[dict[str, Any]]:
    """Compute proposed standardized column names without mutating the DataFrame.

    Ensures that all resulting column names are unique.

    Args:
        df: Source DataFrame.
        strategies: List of strategies to apply.

    Returns:
        List of dicts: [
            {"current_name": "...", "new_name": "...", "changed": True/False}
        ]
    """
    proposals: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for col in df.columns:
        curr = f"{col}"
        std = standardize_name(curr, strategies)

        # Disambiguate if collision occurs
        if std in seen:
            count = seen[std]
            unique_name = f"{std}_{count}"
            while unique_name in seen:
                count += 1
                unique_name = f"{std}_{count}"
            seen[std] = count + 1
            seen[unique_name] = 1
            final_name = unique_name
        else:
            seen[std] = 1
            final_name = std

        proposals.append({
            "current_name": curr,
            "new_name": final_name,
            "changed": curr != final_name,
        })

    return proposals


def standardize_column_names(
    df: pd.DataFrame,
    strategies: list[str],
    selected_columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, RepairRecord]:
    """Standardize column names across the DataFrame.

    Args:
        df: Source DataFrame.
        strategies: Strategies to apply.
        selected_columns: If provided, only standardize these columns.

    Returns:
        Tuple of (repaired DataFrame, RepairRecord).
    """
    proposals = get_column_name_standardizations(df, strategies)
    selected_set = set(selected_columns) if selected_columns else None

    new_cols = []
    changes = {}

    for p in proposals:
        curr = p["current_name"]
        new = p["new_name"]
        if selected_set is None or curr in selected_set:
            new_cols.append(new)
            if curr != new:
                changes[curr] = new
        else:
            new_cols.append(curr)

    result = df.copy()
    result.columns = pd.Index(new_cols)

    record = RepairRecord(
        operation="standardize_columns",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        details={
            "repair_type": "column_rename",
            "method": "standardize",
            "strategies": strategies,
            "changes": changes,
            "total_renamed": len(changes),
            "confidence": 1.0,
            "status": "applied",
        },
    )

    logger.info("Standardized {} column names", len(changes))
    return result, record


def detect_duplicate_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Detect duplicate column names in the DataFrame.

    Args:
        df: Source DataFrame.

    Returns:
        Dict with keys:
        - 'has_duplicates': bool
        - 'duplicates': dict[str, int] (column name -> appearance count)
        - 'proposed_resolution': list[dict[str, Any]] (index, current_name, proposed_name)
    """
    col_names = [f"{c}" for c in df.columns]
    counts = Counter(col_names)
    duplicates = {col: count for col, count in counts.items() if count > 1}

    proposed_resolution: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for idx, col in enumerate(col_names):
        if col in duplicates:
            seen[col] = seen.get(col, 0) + 1
            if seen[col] == 1:
                # Keep first occurrence as-is
                proposed_name = col
            else:
                proposed_name = f"{col}_{seen[col]}"
        else:
            proposed_name = col

        proposed_resolution.append({
            "index": idx,
            "current_name": col,
            "proposed_name": proposed_name,
            "is_duplicate": col in duplicates,
        })

    return {
        "has_duplicates": len(duplicates) > 0,
        "duplicates": duplicates,
        "proposed_resolution": proposed_resolution,
    }


def resolve_duplicate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, RepairRecord]:
    """Safely disambiguate duplicate column names by appending numeric suffixes.

    Args:
        df: Source DataFrame.

    Returns:
        Tuple of (repaired DataFrame, RepairRecord).
    """
    detection = detect_duplicate_columns(df)
    if not detection["has_duplicates"]:
        record = RepairRecord(
            operation="resolve_duplicate_columns",
            timestamp=datetime.now(),
            rows_before=len(df),
            rows_after=len(df),
            columns_before=len(df.columns),
            columns_after=len(df.columns),
            success=True,
            details={"changes": {}, "message": "No duplicate columns detected"},
        )
        return df.copy(), record

    new_names = [p["proposed_name"] for p in detection["proposed_resolution"]]
    changes = {
        p["current_name"]: p["proposed_name"]
        for p in detection["proposed_resolution"]
        if p["current_name"] != p["proposed_name"]
    }

    result = df.copy()
    result.columns = pd.Index(new_names)

    record = RepairRecord(
        operation="resolve_duplicate_columns",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        details={
            "repair_type": "column_rename",
            "method": "duplicate_resolution",
            "changes": changes,
            "total_resolved": len(changes),
            "confidence": 1.0,
            "status": "applied",
        },
    )

    logger.info("Resolved duplicate columns: {}", changes)
    return result, record


def detect_unnamed_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect unnamed or empty column names and classify their contents.

    Classifies into:
    - 'accidental_index': values match sequential row index (0..N-1 or 1..N)
    - 'empty_column': entire column is null or whitespace
    - 'legitimate_data': column has non-index values but lacks header

    Args:
        df: Source DataFrame.

    Returns:
        List of dicts describing detected unnamed columns.
    """
    detected: list[dict[str, Any]] = []

    for idx, col in enumerate(df.columns):
        col_str = f"{col}".strip()
        is_unnamed = (
            col is None
            or col_str == ""
            or col_str.lower() == "none"
            or bool(re.match(r"^unnamed:\s*\d+$", col_str, re.IGNORECASE))
        )

        if not is_unnamed:
            continue

        series = df.iloc[:, idx]
        total_rows = len(df)

        # 1. Check if completely empty
        if series.isna().all() or series.astype(str).str.strip().eq("").all():
            issue = "Completely empty column with no data."
            recommendation = "Remove"
            confidence = 0.95
            classification = "empty_column"

        # 2. Check if accidental index column
        elif _is_sequential_index(series, total_rows):
            issue = "Looks like an exported dataframe index."
            recommendation = "Remove"
            confidence = 0.98
            classification = "accidental_index"

        # 3. Legitimate unnamed data
        else:
            issue = "Contains meaningful data but lacks a descriptive column name."
            recommendation = "Keep / Rename"
            confidence = 0.85
            classification = "legitimate_data"

        detected.append({
            "index": idx,
            "column_name": f"{col}",
            "classification": classification,
            "issue": issue,
            "recommendation": recommendation,
            "confidence": confidence,
            "sample_values": series.dropna().head(3).tolist(),
        })

    return detected


def _is_sequential_index(series: pd.Series, total_rows: int) -> bool:
    """Helper to determine if a series matches a sequential row index."""
    if total_rows == 0:
        return False

    try:
        clean = series.dropna()
        if len(clean) != total_rows:
            return False

        # Try integer conversion
        num_vals = pd.to_numeric(clean, errors="coerce")
        if num_vals.isna().any():
            return False

        vals = num_vals.to_numpy(dtype=float, na_value=np.nan)
        # Match 0, 1, 2, ...
        if np.array_equal(vals, np.arange(total_rows, dtype=float)):
            return True
        # Match 1, 2, 3, ...
        if np.array_equal(vals, np.arange(1, total_rows + 1, dtype=float)):
            return True

        return False
    except Exception:
        return False


def remove_column(
    df: pd.DataFrame, column_name: str, reason: str = ""
) -> tuple[pd.DataFrame, RepairRecord]:
    """Safely drop a single column upon user confirmation.

    Args:
        df: Source DataFrame.
        column_name: Name of column to drop.
        reason: Justification for removal.

    Returns:
        Tuple of (repaired DataFrame, RepairRecord).
    """
    if column_name not in df.columns:
        raise RepairValidationError(f"Column '{column_name}' not found in dataset.")

    result = df.drop(columns=[column_name])

    record = RepairRecord(
        operation="drop_column",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        details={
            "repair_type": "column_drop",
            "column_dropped": column_name,
            "reason": reason or f"User confirmed removal of {column_name}",
            "status": "applied",
        },
    )

    logger.info("Dropped column '{}': {}", column_name, reason)
    return result, record


# Deterministic rename_columns for backward compatibility with autonomous pipeline
def rename_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, RepairRecord]:
    """Fix column name issues: unnamed, duplicates, whitespace (deterministic).

    Kept for backward compatibility with autonomous pipeline and existing test suite.
    """
    result = df.copy()
    original_names = [f"{c}" for c in result.columns]
    new_names: list[str] = []
    changes: dict[str, str] = {}

    unnamed_counter = 0

    for col in original_names:
        stripped = col.strip()

        # Handle unnamed/empty columns
        if stripped.startswith("Unnamed:") or stripped == "":
            new_name = f"unnamed_{unnamed_counter}"
            unnamed_counter += 1
            new_names.append(new_name)
            changes[col] = new_name
        elif stripped != col:
            # Strip whitespace
            new_names.append(stripped)
            changes[col] = stripped
        else:
            new_names.append(col)

    # Handle duplicates — only add suffix to the second+ occurrence
    final_names: list[str] = []
    seen: dict[str, int] = {}

    for name in new_names:
        if name in seen:
            count = seen[name]
            new_name = f"{name}_{count}"
            while new_name in seen or new_name in new_names:
                count += 1
                new_name = f"{name}_{count}"
            seen[name] = count + 1
            seen[new_name] = 1
            final_names.append(new_name)
            original_idx = len(final_names) - 1
            if original_names[original_idx] not in changes:
                changes[original_names[original_idx]] = new_name
        else:
            seen[name] = 1
            final_names.append(name)

    result.columns = pd.Index(final_names)

    logger.info("Column rename: {} changes made", len(changes))

    record = RepairRecord(
        operation="rename_columns",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        details={"changes": changes, "total_renamed": len(changes)},
    )

    return result, record
