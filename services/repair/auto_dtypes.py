"""General-purpose automatic semantic dtype detection and safe conversion engine.

Features:
1. Semantic Datatype Inference:
   - Identifies: integer, float, decimal/numeric, boolean, datetime, categorical, string, identifier, percentage, currency, text.
   - Computes calibrated confidence and actionable evidence for every column.
2. Safe Conversion Protocol:
   - Makes a copy, attempts conversion, validates success ratio.
   - Detects conversion failures and verifies row count and null count.
   - Rejects conversion if unexpected data loss occurs (never silently drops values to NaN).
3. Robust Numeric Normalization:
   - Currency symbols: $, €, £, ¥, ₹, ₩, ₺, ₪, ฿, ₫, Rs, PKR, USD, EUR, etc.
   - Percentages: % suffix with safe magnitude extraction.
   - Negative representations: (1,000) -> -1000, 50- -> -50, -$50 -> -50.
   - Locale-agnostic decimal point detection (distinguishes 1,000.50 vs 1.000,50 dynamically).
4. Identifier Protection:
   - Detects zero-padded numeric codes ('00123'), alphanumeric keys ('INV-001'), postal codes, SKUs.
   - Strictly prevents destructive casting to float/int.
5. Ambiguity Guard:
   - Rejects auto-conversion of ambiguous dates without explicit user approval.
"""

import re
from datetime import datetime
from typing import Any, Optional, Union
import numpy as np
import pandas as pd

from core.logging import logger
from models.repair import (
    ConfidenceLevel,
    IssueTaxonomy,
    RepairRecord,
    SemanticType,
    TypeInferenceResult,
)
from services.repair.dates import analyze_date_column
from utils.number_parser import parse_written_number, parse_magnitude_abbreviation


PARSE_THRESHOLD = 0.85

# Boolean-like value sets (case-insensitive)
BOOL_TRUE_VALUES = {"true", "yes", "y", "1", "t", "on"}
BOOL_FALSE_VALUES = {"false", "no", "n", "0", "f", "off"}
ALL_BOOL_VALUES = BOOL_TRUE_VALUES | BOOL_FALSE_VALUES

# Currency markers across locales
CURRENCY_SYMBOLS = ["$", "€", "£", "¥", "₹", "₩", "₺", "₪", "฿", "₫", "rs.", "rs", "pkr", "usd", "eur", "gbp", "cad", "aud"]


def _is_identifier_column(series: pd.Series) -> bool:
    """Check if a series is an identifier/code and should NOT be converted to numeric.

    100% Data-Driven & Column-Name Agnostic:
    1. Zero-padded numeric strings (e.g. '00123', '00045', zip codes '01234')
    2. Alphanumeric / hyphenated patterns (e.g. 'INV-001', 'CA-2011-167199', 'CUST_99')
    3. High uniqueness (>85%) with structured discrete string keys
    """
    non_null = series.dropna().astype(str).str.strip()
    if len(non_null) == 0:
        return False

    # Check if this series is date-like; if so, it is handled by date parser, not ID guard
    sample_10 = non_null.head(10).tolist()
    date_like = any(
        re.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", v)
        or re.match(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}", v)
        for v in sample_10
    )
    if date_like:
        return False

    # 1. Check for leading zero strings that are purely digits and length > 1 (e.g. '00123', '09')
    sample_100 = non_null.head(100).tolist()
    has_leading_zeros = any(
        v.startswith("0") and v.isdigit() and len(v) > 1 and v != "0"
        for v in sample_100
    )
    if has_leading_zeros:
        return True

    # 2. Check for alphanumeric hyphenated/underscore codes (e.g. 'INV-001', 'CA-2011-167199') with at least one letter
    has_code_pattern = any(
        re.match(r"^[A-Za-z0-9]+[-_][A-Za-z0-9-_]+$", v) and any(c.isalpha() for c in v)
        for v in sample_100[:50]
    )
    if has_code_pattern:
        return True

    # 3. High uniqueness (>85%) with non-numeric structured string tokens (excluding currency values & multi-word prose)
    unique_ratio = series.nunique() / max(len(series), 1)
    if unique_ratio >= 0.85:
        avg_spaces = sum(v.count(" ") for v in sample_100[:20]) / max(len(sample_100[:20]), 1)
        if avg_spaces >= 2:
            return False  # Multi-word prose, not a short discrete identifier key
        has_currency = any(any(sym in v.lower() for sym in CURRENCY_SYMBOLS) for v in sample_100[:20])
        if not has_currency:
            has_alpha = any(any(c.isalpha() for c in v) for v in sample_100[:20])
            if has_alpha:
                return True

    return False


def _is_boolean_column(series: pd.Series) -> bool:
    """Check if a series contains strictly boolean-like values based on empirical value set."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    lowered = non_null.astype(str).str.strip().str.lower()
    return bool(lowered.isin(ALL_BOOL_VALUES).all())


def _convert_bool_series(series: pd.Series) -> pd.Series:
    """Convert a boolean-like string series to actual boolean dtype."""
    lowered = series.astype(str).str.strip().str.lower()
    result = lowered.map(
        lambda x: True if x in BOOL_TRUE_VALUES else (False if x in BOOL_FALSE_VALUES else pd.NA)
    )
    return result.astype("boolean")


def parse_numeric_value(raw: Any) -> Optional[float]:
    """Parse a single value to float handling parentheses negatives, currencies, percentages, and locale separators."""
    if pd.isna(raw):
        return None
    if isinstance(raw, (int, float, np.integer, np.floating)):
        if np.isinf(raw) or np.isnan(raw):
            return None
        return float(raw)

    s = str(raw).strip()
    if not s:
        return None

    # 1. Try magnitude abbreviation parsing (e.g. 50K, 1.5M, $50K, 2B PKR)
    mag_val = parse_magnitude_abbreviation(s)
    if mag_val is not None:
        return mag_val

    # 2. Try written natural language number parsing (e.g. "one hundred", "twenty five thousand", "two million")
    written_val = parse_written_number(s)
    if written_val is not None:
        return written_val

    is_negative = False

    # Check parentheses negative e.g. (1,000) or ($50.25)
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1].strip()
    elif s.endswith("-"):
        is_negative = True
        s = s[:-1].strip()
    elif s.startswith("-"):
        is_negative = True
        s = s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()

    # Strip currency symbols and names
    lowered_for_curr = s.lower()
    for curr in CURRENCY_SYMBOLS:
        if curr in lowered_for_curr:
            pattern = re.compile(re.escape(curr), re.IGNORECASE)
            s = pattern.sub("", s).strip()

    # Check trailing percentage
    if s.endswith("%"):
        s = s[:-1].strip()

    # Strip non-numeric tokens except digits, comma, dot
    cleaned_num = re.sub(r"[^\d.,]", "", s)
    if not cleaned_num:
        return None

    # Handle comma and dot locale separators
    if "," in cleaned_num and "." in cleaned_num:
        first_comma = cleaned_num.find(",")
        first_dot = cleaned_num.find(".")
        if first_comma < first_dot:
            # 1,234.56 -> comma is thousands, dot is decimal
            cleaned_num = cleaned_num.replace(",", "")
        else:
            # 1.234,56 -> dot is thousands, comma is decimal
            cleaned_num = cleaned_num.replace(".", "").replace(",", ".")
    elif "," in cleaned_num:
        parts = cleaned_num.split(",")
        if len(parts) == 2 and len(parts[1]) != 3:
            cleaned_num = cleaned_num.replace(",", ".")
        else:
            cleaned_num = cleaned_num.replace(",", "")

    try:
        val = float(cleaned_num)
        return -val if is_negative else val
    except (ValueError, TypeError):
        return None


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    """Vectorized cleaning and float conversion of numeric strings."""
    return series.map(parse_numeric_value)


def infer_semantic_datatype(series: pd.Series) -> TypeInferenceResult:
    """Infer the true semantic datatype of a Series with confidence and evidence.

    Supported types:
    - integer, float, decimal, boolean, datetime, categorical, string, identifier, percentage, currency, text.
    """
    col_name = str(series.name) if series.name else "column"
    non_null = series.dropna()
    total_non_null = len(non_null)

    if total_non_null == 0:
        return TypeInferenceResult(
            column=col_name,
            detected_type=SemanticType.STRING.value,
            confidence=0.5,
            evidence=["All values in column are null or missing"],
            original_dtype=str(series.dtype),
        )

    str_sample = non_null.astype(str).str.strip()
    raw_sample = non_null.head(5).tolist()
    sample_values = [
        None if (isinstance(v, (float, np.floating)) and (np.isinf(v) or np.isnan(v))) else v
        for v in raw_sample
    ]

    # 1. Check Identifier
    if _is_identifier_column(series):
        return TypeInferenceResult(
            column=col_name,
            detected_type=SemanticType.IDENTIFIER.value,
            confidence=0.98,
            evidence=[
                "Protected code/identifier pattern or leading-zero strings detected",
                "Column name hints or high uniqueness on structured keys",
            ],
            original_dtype=str(series.dtype),
            sample_values=sample_values,
            format_patterns=["[A-Z0-9_-]+"],
            suggested_action="preserve_as_identifier",
        )

    # 2. Check Boolean
    if _is_boolean_column(series):
        return TypeInferenceResult(
            column=col_name,
            detected_type=SemanticType.BOOLEAN.value,
            confidence=0.99,
            evidence=["Values strictly match boolean domains (true/false/yes/no)"],
            original_dtype=str(series.dtype),
            sample_values=sample_values,
            suggested_action="convert_to_boolean",
        )

    # 3. Check Date / Datetime
    date_diag = analyze_date_column(series)
    if date_diag.parse_ratio >= PARSE_THRESHOLD and date_diag.parsed_count > 0:
        evidence = [
            f"Parsed {date_diag.parsed_count}/{total_non_null} values as valid dates ({date_diag.parse_ratio:.0%})",
            f"Detected date convention: {date_diag.inferred_convention}",
        ]
        if date_diag.has_time:
            evidence.append("Timestamp/time components detected and preserved")
        if date_diag.is_ambiguous:
            evidence.append("AMBIGUOUS DATE: day and month numbers are <= 12 across all rows without definitive evidence")

        detected_type = SemanticType.DATETIME.value
        return TypeInferenceResult(
            column=col_name,
            detected_type=detected_type,
            confidence=date_diag.confidence,
            evidence=evidence,
            original_dtype=str(series.dtype),
            sample_values=sample_values,
            suspicious_values=date_diag.invalid_samples,
            is_ambiguous=date_diag.is_ambiguous,
            suggested_action="flag_ambiguous" if date_diag.is_ambiguous else "convert_to_datetime",
        )

    # 4. Check Currency & Percentage & Numeric
    parsed_nums = _clean_numeric_series(non_null)
    valid_num_count = parsed_nums.notna().sum()
    num_ratio = valid_num_count / total_non_null

    if num_ratio >= PARSE_THRESHOLD:
        evidence = [f"{num_ratio:.1%} of non-null values parsed cleanly as numbers"]

        curr_count = sum(any(curr in s.lower() for curr in CURRENCY_SYMBOLS) for s in str_sample)
        pct_count = sum(s.endswith("%") for s in str_sample)

        if curr_count >= (total_non_null * 0.4):
            evidence.append(f"Currency symbols or codes detected in {curr_count} rows")
            return TypeInferenceResult(
                column=col_name,
                detected_type=SemanticType.CURRENCY.value,
                confidence=round(min(0.99, 0.90 + (num_ratio * 0.09)), 2),
                evidence=evidence,
                original_dtype=str(series.dtype),
                sample_values=sample_values,
                format_patterns=["$#,##0.00"],
                suggested_action="convert_to_numeric",
            )

        if pct_count >= (total_non_null * 0.4):
            evidence.append(f"Percentage symbol (%) detected in {pct_count} rows")
            return TypeInferenceResult(
                column=col_name,
                detected_type=SemanticType.PERCENTAGE.value,
                confidence=round(min(0.99, 0.90 + (num_ratio * 0.09)), 2),
                evidence=evidence,
                original_dtype=str(series.dtype),
                sample_values=sample_values,
                format_patterns=["00.0%"],
                suggested_action="convert_to_numeric",
            )

        # Distinguish integer vs float
        valid_nums_series = parsed_nums.dropna()
        all_integers = (valid_nums_series % 1 == 0).all()
        detected_type = SemanticType.INTEGER.value if all_integers else SemanticType.FLOAT.value
        evidence.append("All values represent whole numbers" if all_integers else "Decimal values detected")

        return TypeInferenceResult(
            column=col_name,
            detected_type=detected_type,
            confidence=round(min(0.99, 0.88 + (num_ratio * 0.10)), 2),
            evidence=evidence,
            original_dtype=str(series.dtype),
            sample_values=sample_values,
            suggested_action="convert_to_numeric",
        )

    # 5. Check Categorical vs Free Text
    unique_count = non_null.nunique()
    cardinality_ratio = unique_count / max(total_non_null, 1)
    avg_len = str_sample.str.len().mean()

    if avg_len > 80 or str_sample.str.split().str.len().mean() > 10:
        return TypeInferenceResult(
            column=col_name,
            detected_type=SemanticType.TEXT.value,
            confidence=0.95,
            evidence=["High average word count and string length indicates free-form text or notes"],
            original_dtype=str(series.dtype),
            sample_values=sample_values,
            suggested_action="preserve_as_text",
        )

    if cardinality_ratio <= 0.60 or unique_count <= 50:
        return TypeInferenceResult(
            column=col_name,
            detected_type=SemanticType.CATEGORICAL.value,
            confidence=0.92,
            evidence=[f"Low cardinality ({unique_count} unique values in {total_non_null} rows) indicates categorical data"],
            original_dtype=str(series.dtype),
            sample_values=sample_values,
            suggested_action="preserve_as_categorical",
        )

    return TypeInferenceResult(
        column=col_name,
        detected_type=SemanticType.STRING.value,
        confidence=0.80,
        evidence=["Standard string representation without detected specialized format"],
        original_dtype=str(series.dtype),
        sample_values=sample_values,
        suggested_action="preserve",
    )


def safe_convert_series(
    series: pd.Series,
    target_type: str,
    threshold: float = PARSE_THRESHOLD,
) -> tuple[bool, pd.Series, dict[str, Any]]:
    """Safely convert a series with strict data-loss protection and rollback on failure."""
    copy_series = series.copy()
    original_non_null = copy_series.notna().sum()

    if original_non_null == 0:
        return False, copy_series, {"reason": "Series is completely empty"}

    if target_type == SemanticType.BOOLEAN.value or target_type == "boolean":
        converted = _convert_bool_series(copy_series)
        valid_count = converted.notna().sum()
        loss = original_non_null - valid_count
        if loss == 0 and valid_count > 0:
            return True, converted, {"values_lost": 0, "target_type": "boolean"}
        return False, copy_series, {"reason": f"Boolean conversion would drop {loss} value(s)", "loss": loss}

    elif target_type in (SemanticType.INTEGER.value, SemanticType.FLOAT.value, SemanticType.CURRENCY.value, SemanticType.PERCENTAGE.value, "numeric"):
        cleaned = _clean_numeric_series(copy_series)
        valid_count = cleaned.notna().sum()
        loss = original_non_null - valid_count
        success_ratio = valid_count / original_non_null if original_non_null > 0 else 0.0

        if success_ratio >= threshold and loss == 0:
            valid_subset = cleaned.dropna()
            if len(valid_subset) > 0 and (valid_subset % 1 == 0).all():
                final_series = cleaned.round().astype("Int64")
            else:
                final_series = cleaned.astype(float)
            return True, final_series, {"values_lost": 0, "target_type": str(final_series.dtype), "success_ratio": success_ratio}
        elif success_ratio >= threshold:
            return False, copy_series, {
                "reason": f"Conversion would silently coerce {loss} meaningful value(s) to NaN",
                "loss": loss,
                "success_ratio": success_ratio,
            }
        return False, copy_series, {"reason": "Parse ratio below threshold", "success_ratio": success_ratio}

    elif target_type in (SemanticType.DATETIME.value, "datetime"):
        diag = analyze_date_column(copy_series)
        if diag.is_ambiguous:
            return False, copy_series, {
                "reason": "Ambiguous date convention across column (cannot distinguish day vs month with certainty)",
                "is_ambiguous": True,
            }
        if diag.parse_ratio >= threshold and diag.invalid_count == 0:
            try:
                parse_utc = diag.has_timezone
                converted = pd.to_datetime(copy_series, errors="coerce", dayfirst=diag.dayfirst, format="mixed", utc=parse_utc)
                if converted.notna().sum() == original_non_null:
                    return True, converted, {"values_lost": 0, "target_type": "datetime64[ns]", "dayfirst": diag.dayfirst}
            except Exception as e:
                return False, copy_series, {"reason": f"Datetime parse error: {str(e)}"}

        return False, copy_series, {
            "reason": f"Datetime parse would drop {diag.invalid_count} value(s)",
            "invalid_count": diag.invalid_count,
        }

    return False, copy_series, {"reason": f"Unsupported target type '{target_type}'"}


def auto_detect_dtypes(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    threshold: float = PARSE_THRESHOLD,
) -> tuple[pd.DataFrame, RepairRecord]:
    """Auto-detect and safely convert object columns to their correct semantic dtypes.

    Never converts identifiers.
    Never auto-converts ambiguous dates.
    Never silently turns meaningful values into NaN.
    """
    result = df.copy()
    target_cols = columns if columns else [
        c for c in result.columns if result[c].dtype == "object" or pd.api.types.is_string_dtype(result[c])
    ]

    conversions: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    total_converted = 0

    for col in target_cols:
        if col not in result.columns:
            continue

        non_null_count = result[col].notna().sum()
        if non_null_count == 0:
            continue

        inference = infer_semantic_datatype(result[col])

        if inference.detected_type == SemanticType.IDENTIFIER.value:
            continue

        if inference.is_ambiguous:
            warnings.append(f"Column '{col}' flagged: AMBIGUOUS DATE — REVIEW REQUIRED (no automatic conversion).")
            continue

        if inference.detected_type in (SemanticType.INTEGER.value, SemanticType.FLOAT.value, SemanticType.CURRENCY.value, SemanticType.PERCENTAGE.value):
            success, converted_s, details = safe_convert_series(result[col], "numeric", threshold=threshold)
            if success:
                result[col] = converted_s
                conversions[col] = {
                    "from": "object",
                    "to": str(converted_s.dtype),
                    "semantic_type": inference.detected_type,
                    "confidence": inference.confidence,
                    "evidence": inference.evidence,
                }
                total_converted += 1
            elif details.get("loss", 0) > 0:
                warnings.append(f"Column '{col}' conversion to numeric skipped: would lose {details['loss']} values.")

        elif inference.detected_type == SemanticType.BOOLEAN.value:
            success, converted_s, details = safe_convert_series(result[col], "boolean", threshold=threshold)
            if success:
                result[col] = converted_s
                conversions[col] = {
                    "from": "object",
                    "to": "boolean",
                    "semantic_type": SemanticType.BOOLEAN.value,
                    "confidence": inference.confidence,
                }
                total_converted += 1

        elif inference.detected_type == SemanticType.DATETIME.value and not inference.is_ambiguous:
            success, converted_s, details = safe_convert_series(result[col], "datetime", threshold=threshold)
            if success:
                result[col] = converted_s
                conversions[col] = {
                    "from": "object",
                    "to": "datetime64[ns]",
                    "semantic_type": SemanticType.DATETIME.value,
                    "confidence": inference.confidence,
                    "dayfirst": details.get("dayfirst", False),
                }
                total_converted += 1

    logger.info(
        "Safe semantic dtype auto-detection: converted {} column(s) - {}",
        total_converted,
        ", ".join(f"{c}->{info['to']}" for c, info in conversions.items()) or "none",
    )

    record = RepairRecord(
        operation="auto_dtypes",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        column=", ".join(conversions.keys()) if conversions else None,
        issue_type=IssueTaxonomy.DATATYPE.value,
        original_value="object",
        new_value=", ".join(f"{c}:{v['to']}" for c, v in conversions.items()) if conversions else "none",
        method="deterministic_semantic_engine",
        confidence=0.98 if total_converted > 0 else 1.0,
        reason=f"Safely converted {total_converted} column(s) based on semantic datatype inference without data loss",
        status="applied",
        risk_level="safe",
        warnings=warnings,
        details={
            "total_converted": total_converted,
            "conversions": conversions,
            "warnings": warnings,
        },
    )

    return result, record


def detect_mistyped_columns(
    df: pd.DataFrame,
    threshold: float = PARSE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Detect columns that are likely mistyped using semantic inference."""
    candidates: list[dict[str, Any]] = []

    for col in df.columns:
        if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
            continue
        non_null = df[col].notna().sum()
        if non_null == 0:
            continue

        inference = infer_semantic_datatype(df[col])

        if inference.detected_type in (
            SemanticType.INTEGER.value,
            SemanticType.FLOAT.value,
            SemanticType.CURRENCY.value,
            SemanticType.PERCENTAGE.value,
        ) and inference.confidence >= threshold:
            candidates.append({
                "column": col,
                "suggested_type": "numeric",
                "semantic_type": inference.detected_type,
                "confidence": inference.confidence,
                "evidence": inference.evidence,
                "parse_ratio": 1.0,
            })
        elif inference.detected_type == SemanticType.BOOLEAN.value and inference.confidence >= threshold:
            candidates.append({
                "column": col,
                "suggested_type": "boolean",
                "semantic_type": SemanticType.BOOLEAN.value,
                "confidence": inference.confidence,
                "evidence": inference.evidence,
                "parse_ratio": 1.0,
            })
        elif inference.detected_type == SemanticType.DATETIME.value and inference.confidence >= threshold:
            candidates.append({
                "column": col,
                "suggested_type": "datetime",
                "semantic_type": SemanticType.DATETIME.value,
                "confidence": inference.confidence,
                "evidence": inference.evidence,
                "is_ambiguous": inference.is_ambiguous,
                "parse_ratio": 1.0,
            })

    return candidates


# Alias for backward and external compatibility
auto_detect_and_convert_dtypes = auto_detect_dtypes

