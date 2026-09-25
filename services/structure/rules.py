"""Structure detection rules.

Each rule evaluates a DataFrame and returns a RuleEvidence with a score,
evidence list, and reason. The detector picks the strongest-scoring rule.

Rules use multiple signals — never simplistic thresholds like 'columns > 10'.
"""

import pandas as pd
import numpy as np

from models.structure import StructureType, RuleEvidence
from services.structure.utils import (
    column_type_summary,
    unique_ratio,
    numeric_ratio,
    header_likelihood,
    column_name_similarity,
    uniform_dtype_ratio,
    repeated_value_pattern,
)
from utils.dataframe import safe_numeric_columns, safe_datetime_columns


def evaluate_all_rules(df: pd.DataFrame) -> list[RuleEvidence]:
    """Evaluate all structure rules against a DataFrame.

    Args:
        df: Source DataFrame.

    Returns:
        List of RuleEvidence, one per rule.
    """
    rules = [
        _rule_normal_table,
        _rule_wide_table,
        _rule_long_table,
        _rule_pivot_table,
        _rule_crosstab,
        _rule_multi_header,
        _rule_repeated_category_blocks,
        _rule_spacer_artifacts,
        _rule_subtotal_columns_rows,
        _rule_metadata_banners,
        _rule_time_series,
        _rule_survey,
        _rule_transactional,
        _rule_embedded_structured_records,
    ]

    results: list[RuleEvidence] = []
    for rule_fn in rules:
        try:
            result = rule_fn(df)
            results.append(result)
        except Exception:
            pass

    return results


def _rule_normal_table(df: pd.DataFrame) -> RuleEvidence:
    """Detect a normal/standard table structure.

    Signals: mixed dtypes, reasonable column count, column names look like
    independent variables, no repeated-category patterns.
    """
    evidence: list[str] = []
    score = 0.0
    n_cols = len(df.columns)
    n_rows = len(df)
    type_summary = column_type_summary(df)

    # Mixed column types suggest independent variables
    type_count = sum(1 for v in type_summary.values() if v > 0)
    if type_count >= 2:
        score += 0.25
        evidence.append(f"Contains {type_count} different column type categories")

    # Reasonable column count (not excessively wide)
    if 2 <= n_cols <= 30:
        score += 0.2
        evidence.append(f"Reasonable column count: {n_cols}")

    # Low column-name similarity (names represent different things)
    col_sim = column_name_similarity([str(c) for c in df.columns])
    if col_sim < 0.2:
        score += 0.2
        evidence.append("Column names represent distinct variables")

    # Rows substantially outnumber columns
    if n_rows > 0 and n_cols > 0 and n_rows / n_cols > 3:
        score += 0.15
        evidence.append(f"Row-to-column ratio: {n_rows / n_cols:.1f}")

    # No multi-header signals
    h_likelihood = header_likelihood(df)
    if h_likelihood < 0.3:
        score += 0.1
        evidence.append("No multi-header signals detected")

    # If there's an obvious regular time series, slightly dampen normal table score
    dt_cols = safe_datetime_columns(df)
    if len(dt_cols) == 1 and len(safe_numeric_columns(df)) >= 1 and len(df.columns) <= 5:
        score -= 0.15

    score = max(0.0, min(1.0, score))

    return RuleEvidence(
        rule_name="normal_table",

        structure_type=StructureType.NORMAL_TABLE,
        score=score,
        evidence=evidence,
        reason="Dataset appears to be a standard table with independent variable columns",
    )


def _rule_wide_table(df: pd.DataFrame) -> RuleEvidence:
    """Detect a wide table structure.

    Key signals:
    - Many columns with similar names (category patterns)
    - Column names resemble category labels
    - Many columns have the same dtype (especially numeric)
    - Values spread horizontally suggest categories-as-columns
    - First column(s) act as identifiers, rest are value columns
    """
    evidence: list[str] = []
    score = 0.0
    n_cols = len(df.columns)
    col_names = [str(c) for c in df.columns]

    # Column count signal (wide tables have many columns)
    if n_cols >= 8:
        score += 0.1
        evidence.append(f"High column count: {n_cols}")
    if n_cols >= 20:
        score += 0.1
        evidence.append(f"Very high column count: {n_cols}")

    # Column name similarity (categories spread across columns)
    col_sim = column_name_similarity(col_names)
    if col_sim > 0.3:
        score += 0.25
        evidence.append(f"Column names show category patterns (similarity: {col_sim:.2f})")

    # Uniform dtypes across many columns
    dtype_ratio = uniform_dtype_ratio(df)
    if dtype_ratio > 0.7 and n_cols >= 6:
        score += 0.2
        evidence.append(f"High dtype uniformity: {dtype_ratio:.0%} of columns share the same type")

    # Many numeric columns with similar ranges
    num_cols = safe_numeric_columns(df)
    if len(num_cols) >= 5 and len(num_cols) / n_cols > 0.6:
        # Check if value ranges are similar
        ranges: list[tuple[float, float]] = []
        for col in num_cols[:20]:
            series = df[col].dropna()
            if len(series) > 0:
                ranges.append((float(series.min()), float(series.max())))

        if len(ranges) >= 3:
            min_vals = [r[0] for r in ranges]
            max_vals = [r[1] for r in ranges]
            min_range = max(min_vals) - min(min_vals) if min_vals else 0
            max_range = max(max_vals) - min(max_vals) if max_vals else 0
            overall_range = max(max_vals) - min(min_vals) if max_vals and min_vals else 1

            if overall_range > 0 and (min_range + max_range) / (2 * overall_range) < 0.5:
                score += 0.15
                evidence.append("Numeric columns have similar value ranges")

        score += 0.1
        evidence.append(f"High proportion of numeric columns: {len(num_cols)}/{n_cols}")

    # First column(s) look like identifiers, rest look like values
    if n_cols >= 5:
        first_col_unique = unique_ratio(df, str(df.columns[0]))
        if first_col_unique > 0.5:
            remaining_types = set()
            for col in df.columns[1:]:
                remaining_types.add(str(df[col].dtype))
            if len(remaining_types) == 1:
                score += 0.15
                evidence.append("First column appears to be identifier; remaining columns share same type")

    score = min(1.0, score)

    return RuleEvidence(
        rule_name="wide_table",
        structure_type=StructureType.WIDE_TABLE,
        score=score,
        evidence=evidence,
        reason="Dataset may have category values spread across columns (wide format)",
    )


def _rule_long_table(df: pd.DataFrame) -> RuleEvidence:
    """Detect a long/tidy table structure.

    Signals: few columns, many rows, one or more identifier columns,
    a 'variable' or 'key' column, a 'value' column.
    """
    evidence: list[str] = []
    score = 0.0
    n_cols = len(df.columns)
    n_rows = len(df)
    col_names_lower = [str(c).lower() for c in df.columns]

    # Few columns, many rows
    if n_cols <= 5 and n_rows > 50:
        score += 0.2
        evidence.append(f"Narrow shape: {n_cols} columns, {n_rows} rows")

    # Look for key-value column naming patterns
    kv_keywords = ["variable", "key", "metric", "measure", "attribute", "indicator"]
    val_keywords = ["value", "amount", "count", "score", "result"]

    has_key_col = any(kw in name for name in col_names_lower for kw in kv_keywords)
    has_val_col = any(kw in name for name in col_names_lower for kw in val_keywords)

    if has_key_col:
        score += 0.25
        evidence.append("Contains a key/variable column")
    if has_val_col:
        score += 0.2
        evidence.append("Contains a value column")

    # Repeated identifier values (same ID appears many times)
    if n_cols >= 2:
        first_col_ratio = unique_ratio(df, str(df.columns[0]))
        if first_col_ratio < 0.3 and n_rows > 10:
            score += 0.2
            evidence.append("First column has many repeated values (possible ID/group)")

    score = min(1.0, score)

    return RuleEvidence(
        rule_name="long_table",
        structure_type=StructureType.LONG_TABLE,
        score=score,
        evidence=evidence,
        reason="Dataset appears to be in long/tidy format with key-value structure",
    )


def _rule_pivot_table(df: pd.DataFrame) -> RuleEvidence:
    """Detect a pivot table structure.

    Signals: first column has category labels, column headers are categories,
    body contains aggregated numeric values, matrix-like layout.
    """
    evidence: list[str] = []
    score = 0.0
    n_cols = len(df.columns)
    n_rows = len(df)

    if n_cols < 3 or n_rows < 2:
        return RuleEvidence(
            rule_name="pivot_table",
            structure_type=StructureType.PIVOT_TABLE,
            score=0.0,
            evidence=["Too few columns or rows for pivot table"],
            reason="",
        )

    # First column is categorical, rest are numeric
    first_col = df.iloc[:, 0]
    is_first_categorical = not pd.api.types.is_numeric_dtype(first_col)

    remaining_numeric = 0
    for i in range(1, n_cols):
        if pd.api.types.is_numeric_dtype(df.iloc[:, i]):
            remaining_numeric += 1

    remaining_total = n_cols - 1
    numeric_pct = remaining_numeric / remaining_total if remaining_total > 0 else 0

    if is_first_categorical and numeric_pct > 0.8:
        score += 0.3
        evidence.append("First column is categorical, remaining columns are mostly numeric")

    # Row categories are unique
    if is_first_categorical:
        row_unique_ratio = unique_ratio(df, str(df.columns[0]))
        if row_unique_ratio > 0.7:
            score += 0.15
            evidence.append("Row categories are mostly unique")

    # Column headers look like category labels (short strings)
    col_names = [str(c) for c in df.columns[1:]]
    avg_name_len = sum(len(n) for n in col_names) / max(len(col_names), 1)
    if 1 < avg_name_len < 25:
        score += 0.1
        evidence.append(f"Column headers have label-like names (avg length: {avg_name_len:.0f})")

    # Moderate size (pivot tables are typically not huge)
    if n_rows < 100 and n_cols > 3:
        score += 0.1
        evidence.append(f"Compact size ({n_rows}×{n_cols}) typical of pivot tables")

    # Uniform numeric dtype
    dtype_ratio = uniform_dtype_ratio(df.iloc[:, 1:])
    if dtype_ratio > 0.9:
        score += 0.1
        evidence.append("Value columns have uniform dtype")

    score = min(1.0, score)

    return RuleEvidence(
        rule_name="pivot_table",
        structure_type=StructureType.PIVOT_TABLE,
        score=score,
        evidence=evidence,
        reason="Dataset appears to be a pivot table with categories as row/column headers",
    )


def _rule_crosstab(df: pd.DataFrame) -> RuleEvidence:
    """Detect a crosstab/contingency table.

    Similar to pivot but typically contains counts/frequencies.
    """
    evidence: list[str] = []
    score = 0.0
    n_cols = len(df.columns)
    n_rows = len(df)

    if n_cols < 3 or n_rows < 2:
        return RuleEvidence(
            rule_name="crosstab",
            structure_type=StructureType.CROSSTAB,
            score=0.0,
            evidence=["Too small for crosstab"],
            reason="",
        )

    # Check if body values are all integers (counts)
    num_cols = safe_numeric_columns(df)
    value_cols = [c for c in num_cols if c != str(df.columns[0])]

    if len(value_cols) >= 2:
        all_integers = True
        for col in value_cols:
            series = df[col].dropna()
            if len(series) > 0:
                # Check if all values are integer-like
                try:
                    if not all(float(v) == int(float(v)) for v in series.head(50)):
                        all_integers = False
                        break
                except (ValueError, TypeError):
                    all_integers = False
                    break

        if all_integers:
            score += 0.3
            evidence.append("Value columns contain integer counts")

    # First column is categorical
    first_col = df.iloc[:, 0]
    if not pd.api.types.is_numeric_dtype(first_col):
        score += 0.15
        evidence.append("First column is categorical (row labels)")

    # Small/medium compact shape
    if n_rows <= 50 and 3 <= n_cols <= 30:
        score += 0.1
        evidence.append(f"Compact shape ({n_rows}×{n_cols}) typical of crosstabs")

    # Column names look like categories
    col_sim = column_name_similarity([str(c) for c in df.columns[1:]])
    if col_sim > 0.2:
        score += 0.15
        evidence.append("Column headers show category patterns")

    score = min(1.0, score)

    return RuleEvidence(
        rule_name="crosstab",
        structure_type=StructureType.CROSSTAB,
        score=score,
        evidence=evidence,
        reason="Dataset appears to be a crosstab/contingency table with counts",
    )


def _rule_multi_header(df: pd.DataFrame) -> RuleEvidence:
    """Detect multi-header structure.

    Signals: first row(s) contain label/category information rather than data,
    type mismatch between row 0 and subsequent rows.
    """
    evidence: list[str] = []
    score = 0.0

    if len(df) < 3:
        return RuleEvidence(
            rule_name="multi_header",
            structure_type=StructureType.MULTI_HEADER,
            score=0.0,
            evidence=["Too few rows to detect multi-header"],
            reason="",
        )

    # Check header likelihood
    h_score = header_likelihood(df)
    if h_score > 0.5:
        score += 0.35
        evidence.append(f"First row looks like a header/label row (likelihood: {h_score:.2f})")
    elif h_score > 0.3:
        score += 0.15
        evidence.append(f"First row has some header characteristics (likelihood: {h_score:.2f})")

    # Check if column names are generic (Unnamed, numbers, etc.)
    unnamed_count = sum(1 for c in df.columns if str(c).startswith("Unnamed:") or str(c).strip() == "")
    if unnamed_count > 0:
        score += 0.2
        evidence.append(f"{unnamed_count} unnamed/generic column names")

    # Check if first row values could be column names
    first_row = df.iloc[0]
    string_count = 0
    for val in first_row:
        if isinstance(val, str) and len(val) < 40:
            string_count += 1

    string_ratio = string_count / max(len(df.columns), 1)
    if string_ratio > 0.7:
        score += 0.2
        evidence.append(f"First row contains {string_ratio:.0%} short strings (possible headers)")

    # Check if there are completely numeric rows after the first
    if len(df) > 2:
        row2 = df.iloc[1]
        numeric_in_row2 = 0
        for val in row2:
            try:
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    float(val)
                    numeric_in_row2 += 1
            except (ValueError, TypeError):
                pass

        if numeric_in_row2 / max(len(df.columns), 1) > 0.5 and string_ratio > 0.5:
            score += 0.15
            evidence.append("Type mismatch between first and second rows")

    score = min(1.0, score)

    return RuleEvidence(
        rule_name="multi_header",
        structure_type=StructureType.MULTI_HEADER,
        score=score,
        evidence=evidence,
        reason="Dataset may contain multiple header rows or metadata in the first row(s)",
    )


def _rule_time_series(df: pd.DataFrame) -> RuleEvidence:
    """Detect time series data.

    Signals: actual datetime column(s), regular time intervals,
    numeric value columns alongside datetime.
    """
    evidence: list[str] = []
    score = 0.0
    n_cols = len(df.columns)

    # Check for datetime columns
    dt_cols = safe_datetime_columns(df)

    if len(dt_cols) >= 1:
        # Check if the datetime column has distinct timestamps and regular intervals
        is_distinct_time = False
        for dt_col in dt_cols[:2]:
            u_ratio = unique_ratio(df, dt_col)
            if u_ratio > 0.6:
                is_distinct_time = True
                score += 0.3
                evidence.append(f"Contains distinct timestamp column '{dt_col}' ({u_ratio:.0%} unique)")

                series = df[dt_col].dropna().sort_values()
                if len(series) >= 3:
                    diffs = series.diff().dropna()
                    # Filter out zero diffs
                    pos_diffs = diffs[diffs > pd.Timedelta(0)]
                    if len(pos_diffs) >= len(diffs) * 0.8:
                        try:
                            unique_diffs = pos_diffs.nunique()
                            if unique_diffs <= 3:
                                score += 0.25
                                evidence.append(f"Column '{dt_col}' has regular time intervals")
                            elif unique_diffs <= 10:
                                score += 0.15
                                evidence.append(f"Column '{dt_col}' has semi-regular time intervals")
                        except Exception:
                            pass
                break
        if not is_distinct_time:
            # Datetime column exists but is mostly constant or non-sequential attribute
            score += 0.05
            evidence.append(f"Contains datetime column(s) but values are repeated/non-sequential")
    else:
        # Check if any object column might be datetime
        for col in df.columns:
            if df[col].dtype == object:
                from services.structure.utils import datetime_ratio
                dt_ratio = datetime_ratio(df, str(col))
                if dt_ratio > 0.8 and unique_ratio(df, str(col)) > 0.6:
                    score += 0.25
                    evidence.append(f"Column '{col}' appears to contain sequential dates ({dt_ratio:.0%} parseable)")
                    break

    # Check for numeric value columns alongside datetime
    num_cols = safe_numeric_columns(df)
    if score >= 0.3 and len(num_cols) >= 1:
        score += 0.2
        evidence.append(f"Contains {len(num_cols)} numeric metric column(s) alongside time index")

    # Time-related column names
    time_keywords = ["date", "time", "timestamp", "year", "month", "day", "period"]
    time_name_cols = [c for c in df.columns if any(kw in str(c).lower() for kw in time_keywords)]
    if time_name_cols and score >= 0.3:
        score += 0.15
        evidence.append(f"Time-related column names: {', '.join(str(c) for c in time_name_cols)}")

    score = min(1.0, score)

    return RuleEvidence(
        rule_name="time_series",
        structure_type=StructureType.TIME_SERIES,
        score=score,
        evidence=evidence,
        reason="Dataset appears to contain time series data",
    )



def _rule_survey(df: pd.DataFrame) -> RuleEvidence:
    """Detect survey-like data.

    Signals: many categorical columns, Likert-scale values,
    question-like column names, consistent value ranges.
    """
    evidence: list[str] = []
    score = 0.0
    n_cols = len(df.columns)

    type_summary = column_type_summary(df)
    cat_count = type_summary.get("categorical", 0)
    num_count = type_summary.get("numeric", 0)

    # Many columns (surveys have many questions)
    if n_cols >= 10:
        score += 0.1
        evidence.append(f"Many columns: {n_cols}")

    # High proportion of categorical columns
    if cat_count > 0 and cat_count / n_cols > 0.5:
        score += 0.15
        evidence.append(f"High categorical column ratio: {cat_count}/{n_cols}")

    # Check for Likert-scale patterns in numeric columns
    likert_count = 0
    num_cols_list = safe_numeric_columns(df)
    for col in num_cols_list[:20]:
        series = df[col].dropna()
        if len(series) > 0:
            unique_vals = sorted(series.unique())
            # Typical Likert: 1-5 or 1-7 or 0-10
            if len(unique_vals) <= 10:
                try:
                    min_v = float(unique_vals[0])
                    max_v = float(unique_vals[-1])
                    if min_v >= 0 and max_v <= 10:
                        likert_count += 1
                except (ValueError, TypeError):
                    pass

    if likert_count >= 3:
        score += 0.25
        evidence.append(f"{likert_count} columns have Likert-scale-like values")

    # Question-like column names
    q_keywords = ["q", "question", "q_", "q.", "response", "answer", "rating", "satisfaction"]
    q_cols = [c for c in df.columns if any(str(c).lower().startswith(kw) for kw in q_keywords)]
    if len(q_cols) >= 3:
        score += 0.2
        evidence.append(f"Question-like column names: {len(q_cols)} columns")

    score = min(1.0, score)

    return RuleEvidence(
        rule_name="survey",
        structure_type=StructureType.SURVEY,
        score=score,
        evidence=evidence,
        reason="Dataset appears to contain survey/questionnaire data",
    )


def _rule_transactional(df: pd.DataFrame) -> RuleEvidence:
    """Detect transactional data.

    Signals: ID column, date/timestamp column, amount/value column,
    typical transaction column names.
    """
    evidence: list[str] = []
    score = 0.0
    col_names_lower = [str(c).lower() for c in df.columns]

    # Transaction-related column names
    id_keywords = ["id", "transaction", "order", "invoice", "receipt"]
    amount_keywords = ["amount", "total", "price", "cost", "revenue", "sales", "quantity"]
    entity_keywords = ["customer", "client", "user", "buyer", "seller", "vendor", "product"]

    has_id = any(any(kw in name for kw in id_keywords) for name in col_names_lower)
    has_amount = any(any(kw in name for kw in amount_keywords) for name in col_names_lower)
    has_entity = any(any(kw in name for kw in entity_keywords) for name in col_names_lower)

    if has_id:
        score += 0.2
        evidence.append("Contains transaction ID-like column")
    if has_amount:
        score += 0.2
        evidence.append("Contains amount/value column")
    if has_entity:
        score += 0.15
        evidence.append("Contains entity column (customer, product, etc.)")

    # Has datetime column
    dt_cols = safe_datetime_columns(df)
    if dt_cols:
        score += 0.15
        evidence.append(f"Contains datetime column(s): {', '.join(dt_cols[:3])}")

    # Many rows (transactions accumulate)
    if len(df) > 100:
        score += 0.1
        evidence.append(f"Large row count: {len(df)}")

    # Mixed column types (IDs + dates + amounts + categories)
    type_summary = column_type_summary(df)
    type_count = sum(1 for v in type_summary.values() if v > 0)
    if type_count >= 3:
        score += 0.1
        evidence.append(f"Diverse column types: {type_count} categories")

    score = min(1.0, score)

    return RuleEvidence(
        rule_name="transactional",
        structure_type=StructureType.TRANSACTIONAL,
        score=score,
        evidence=evidence,
        reason="Dataset appears to contain transactional/sales data",
    )


def _rule_repeated_category_blocks(df: pd.DataFrame) -> RuleEvidence:
    """Detect horizontally distributed repeated category blocks.

    Signals:
    - Top headers or row 0 contain category labels repeated alongside sub-metrics
    - Column names contain patterns like 'Category Total', 'Category - Metric', or repeating sub-header names
    - Unnamed columns stemming from merged cells above repeating sub-blocks
    """
    evidence: list[str] = []
    score = 0.0
    cols = [str(c) for c in df.columns]

    # Check for recurring "Total" or category headers in column names
    total_cols = [c for c in cols if any(kw in c.lower() for kw in ["total", "subtotal", "sum", "grand total"])]
    unnamed_cols = [c for c in cols if c.startswith("Unnamed:") or c.strip() == ""]

    if len(total_cols) >= 2:
        score += 0.35
        evidence.append(f"Contains {len(total_cols)} repeated category total/subtotal columns")

    if len(unnamed_cols) >= 2 and len(total_cols) >= 1:
        score += 0.25
        evidence.append(f"Contains {len(unnamed_cols)} unnamed artifact columns from merged top-level headers")

    # Check if row 0 contains repeating sub-headers across blocks
    if len(df) > 0:
        row0_vals = [str(v).strip().lower() for v in df.iloc[0] if pd.notna(v)]
        if len(row0_vals) > 0:
            val_counts = pd.Series(row0_vals).value_counts()
            repeats = val_counts[val_counts >= 2]
            if len(repeats) >= 1:
                score += 0.3
                evidence.append(f"Row 0 contains repeating metric sub-headers: {', '.join(repeats.index[:3])}")

    score = min(1.0, score)

    return RuleEvidence(
        rule_name="repeated_category_blocks",
        structure_type=StructureType.REPEATED_CATEGORY_BLOCKS,
        score=score,
        evidence=evidence,
        reason="Dataset appears to contain horizontally repeated category blocks with subtotal columns",
    )


def _rule_spacer_artifacts(df: pd.DataFrame) -> RuleEvidence:
    """Detect empty spacer rows and columns.

    Signals:
    - Completely empty columns or rows
    - Columns with >90% nulls acting as visual separators
    """
    evidence: list[str] = []
    score = 0.0

    empty_cols = [str(c) for c in df.columns if df[c].isna().all()]
    sparse_cols = [str(c) for c in df.columns if df[c].isna().mean() > 0.90 and str(c).startswith("Unnamed:")]

    if empty_cols:
        score += 0.4
        evidence.append(f"{len(empty_cols)} completely empty spacer column(s)")

    if sparse_cols:
        score += 0.3
        evidence.append(f"{len(sparse_cols)} sparse spacer column(s) (>90% empty)")

    empty_rows = df.isna().all(axis=1).sum()
    if empty_rows > 0:
        score += 0.25
        evidence.append(f"{empty_rows} empty spacer row(s)")

    score = min(1.0, score)

    return RuleEvidence(
        rule_name="spacer_artifacts",
        structure_type=StructureType.SPACER_ARTIFACTS,
        score=score,
        evidence=evidence,
        reason="Dataset contains blank spacer rows or columns",
    )


def _rule_subtotal_columns_rows(df: pd.DataFrame) -> RuleEvidence:
    """Detect total/subtotal columns or summary rows mixed into data.

    Signals:
    - Columns or rows explicitly labeled Total, Subtotal, Summary, Average
    """
    evidence: list[str] = []
    score = 0.0

    kw_list = ["total", "subtotal", "summary", "average", "grand total"]

    subtotal_cols = [str(c) for c in df.columns if any(kw in str(c).lower() for kw in kw_list)]
    if subtotal_cols:
        score += 0.4
        evidence.append(f"Contains {len(subtotal_cols)} subtotal/total column(s): {', '.join(subtotal_cols[:3])}")

    # Check rows
    if len(df) > 0 and len(df.columns) > 0:
        first_col_str = df.iloc[:, 0].dropna().astype(str).str.lower()
        subtotal_rows = first_col_str[first_col_str.str.contains("|".join(kw_list), regex=True)].count()
        if subtotal_rows > 0:
            score += 0.4
            evidence.append(f"Contains {subtotal_rows} summary/subtotal row(s) in first column")

    score = min(1.0, score)

    return RuleEvidence(
        rule_name="subtotal_columns_rows",
        structure_type=StructureType.SUBTOTAL_COLUMNS_ROWS,
        score=score,
        evidence=evidence,
        reason="Dataset contains summary, total, or subtotal columns/rows",
    )


def _rule_metadata_banners(df: pd.DataFrame) -> RuleEvidence:
    """Detect metadata/report header banners at dataset boundaries.

    Signals:
    - First 1-3 rows have single non-null text value while rest of cells in row are null
    - Report title, timestamp, author notes, or disclaimer text
    """
    evidence: list[str] = []
    score = 0.0

    if len(df.columns) >= 2 and len(df) > 2:
        for r_idx in range(min(3, len(df))):
            row = df.iloc[r_idx]
            non_nulls = row.dropna()
            if len(non_nulls) == 1 and isinstance(non_nulls.iloc[0], str):
                text = str(non_nulls.iloc[0]).strip()
                if len(text) > 5 and not text.isdigit():
                    score += 0.35
                    evidence.append(f"Row {r_idx} contains a single text banner cell: '{text[:30]}...'")

    score = min(1.0, score)

    return RuleEvidence(
        rule_name="metadata_banners",
        structure_type=StructureType.METADATA_BANNERS,
        score=score,
        evidence=evidence,
        reason="Dataset contains top/bottom metadata banner rows",
    )


def _rule_embedded_structured_records(df: pd.DataFrame) -> RuleEvidence:
    """Detect embedded structured records inside text cells (single-column or narrow layout).

    Key signals:
    - Single-column or narrow table layout (1-2 columns)
    - High average string length
    - Cross-row statistical evidence of repeating field labels and key-value patterns
    - Distinction from natural language prose / free-form text comments
    """
    from services.structure.embedded_records import EmbeddedRecordAnalyzer

    evidence: list[str] = []
    score = 0.0

    profile = EmbeddedRecordAnalyzer.profile_dataset(df)

    labels = profile.get("candidate_field_labels", [])
    confidence = profile.get("confidence", 0.0)
    consistency = profile.get("consistency_of_inferred_structure", 0.0)
    ambiguity_flags = profile.get("ambiguity_flags", [])

    if labels and len(labels) >= 2 and confidence >= 0.50:
        score = confidence
        evidence.append(f"Discovered {len(labels)} candidate field label(s): {', '.join(labels[:5])}")
        evidence.append(f"Cross-row structural consistency: {consistency:.0%}")
        if ambiguity_flags:
            evidence.extend(ambiguity_flags)

    return RuleEvidence(
        rule_name="embedded_structured_records",
        structure_type=StructureType.EMBEDDED_STRUCTURED_RECORDS,
        score=score,
        evidence=evidence,
        reason=f"Dataset contains embedded structured records with discovered fields: {', '.join(labels[:4]) if labels else 'None'}",
    )


