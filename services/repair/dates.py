"""General purpose date parsing, normalization, and ambiguity detection engine.

Features:
- Generalized multi-format parsing (ISO, slash/dash/dot, textual months, timestamps)
- Row-aware date classification: UNAMBIGUOUS, AMBIGUOUS, INVALID, MISSING
- Mixed and conflicting date conventions detection (e.g. MM-DD-YYYY mixed with DD-MM-YYYY)
- Time and timezone preservation (HH:MM:SS, sub-seconds, UTC/offsets)
- Ambiguity guard: Strictly flags ambiguous dates (day and month <= 12) for review without guessing
- Comprehensive audit logging and row diagnostics
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import numpy as np
import pandas as pd

from core.logging import logger
from models.repair import DateNormalizationResult, IssueTaxonomy, RepairRecord
from core.exceptions import RepairValidationError


class DateConvention(str, Enum):
    """Enumeration of recognized date conventions."""
    DMY = "DMY"
    MDY = "MDY"
    YMD = "YMD"
    ISO = "ISO"
    TEXT_MONTH = "TEXT_MONTH"
    MIXED = "MIXED"
    AMBIGUOUS = "AMBIGUOUS"
    EMPTY = "EMPTY"


class RowDateStatus(str, Enum):
    """Row-level date classification status."""
    UNAMBIGUOUS = "UNAMBIGUOUS"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"
    MISSING = "MISSING"


# ---------------------------------------------------------------------------
# Display format constant
# ---------------------------------------------------------------------------
# Final standard output format for all date columns: DD-MM-YY  (%d-%m-%y)
# Internal representation stays as datetime64[ns]; this is only applied for
# display, preview, and CSV/Excel export.
DEFAULT_DISPLAY_FORMAT: str = "%d-%m-%y"


def safe_format_timestamp(ts: Any, fmt: str = DEFAULT_DISPLAY_FORMAT) -> str:
    """Format a Timestamp / datetime safely across all platforms (including Windows year < 1900).

    On Windows, the C-runtime implementation of strftime raises ValueError for
    dates with year < 1900 when using %y or certain format directives.
    This helper catches that and performs platform-safe component formatting.
    """
    if ts is None or pd.isna(ts):
        return ""
    try:
        return ts.strftime(fmt)
    except (ValueError, TypeError, AttributeError, OSError):
        pass

    try:
        if hasattr(ts, "day") and hasattr(ts, "month") and hasattr(ts, "year"):
            if fmt in ("%d-%m-%y", DEFAULT_DISPLAY_FORMAT):
                yr_2digit = abs(int(ts.year)) % 100
                return f"{int(ts.day):02d}-{int(ts.month):02d}-{yr_2digit:02d}"
            # Generalized manual replacement
            y4 = f"{abs(int(ts.year)):04d}"
            y2 = f"{abs(int(ts.year)) % 100:02d}"
            res = fmt.replace("%d", f"{int(ts.day):02d}")
            res = res.replace("%m", f"{int(ts.month):02d}")
            res = res.replace("%Y", y4)
            res = res.replace("%y", y2)
            res = res.replace("%H", f"{int(getattr(ts, 'hour', 0)):02d}")
            res = res.replace("%M", f"{int(getattr(ts, 'minute', 0)):02d}")
            res = res.replace("%S", f"{int(getattr(ts, 'second', 0)):02d}")
            return res
    except Exception:
        pass
    return str(ts) if ts is not None else ""


def format_date_series(
    series: pd.Series,
    fmt: str = DEFAULT_DISPLAY_FORMAT,
    date_format: Optional[str] = None,
    na_value: str = "",
) -> pd.Series:
    """Format a datetime64 Series to strings using *fmt* (default DD-MM-YY).

    Rows that are NaT / null become *na_value* (empty string by default).
    If *series* is not a datetime dtype the original series is returned as-is.

    Args:
        series: A pandas Series, ideally datetime64[ns].
        fmt:    strftime format string. Defaults to DEFAULT_DISPLAY_FORMAT.
        date_format: Optional alias for fmt.
        na_value: Replacement for NaT / null cells.

    Returns:
        String Series where every valid date is formatted with *fmt*.

    Example::

        formatted = format_date_series(df["order_date"])  # "01-09-24", "28-01-26", ...
    """
    effective_fmt = date_format or fmt
    if pd.api.types.is_datetime64_any_dtype(series):
        try:
            valid_dt = pd.to_datetime(series.dropna())
            if not valid_dt.empty and (valid_dt.dt.year < 1900).any():
                return series.apply(lambda x: safe_format_timestamp(x, effective_fmt) if pd.notna(x) else na_value)
            dt_series = pd.to_datetime(series)
            return dt_series.dt.strftime(effective_fmt).where(series.notna(), na_value)
        except Exception:
            return series.apply(lambda x: safe_format_timestamp(x, effective_fmt) if pd.notna(x) else na_value)
    elif pd.api.types.is_object_dtype(series):
        # Handle object series containing Timestamp / datetime objects
        return series.apply(lambda x: safe_format_timestamp(x, effective_fmt) if (pd.notna(x) and hasattr(x, "year")) else (x if pd.notna(x) else na_value))
    # Already strings — cannot reformat; return unchanged
    return series


# Common month names pattern across languages/formats
MONTH_NAMES_PATTERN = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)"

# Standard regexes for date candidates
ISO_DATE_REGEX = re.compile(
    r"^(\d{4})[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])(?:[T\s](\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?))?$"
)
TEXT_DATE_PREFIX_REGEX = re.compile(
    rf"^({MONTH_NAMES_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{2,4}})(?:[T\s](\d{{1,2}}:\d{{2}}(?::\d{{2}})?(?:\s*[APap][Mm])?))?",
    re.IGNORECASE,
)
TEXT_DATE_SUFFIX_REGEX = re.compile(
    rf"^(\d{{1,2}})[-/\s]({MONTH_NAMES_PATTERN})[-/\s](\d{{2,4}})(?:[T\s](\d{{1,2}}:\d{{2}}(?::\d{{2}})?(?:\s*[APap][Mm])?))?",
    re.IGNORECASE,
)
SEP_DATE_REGEX = re.compile(
    r"^(\d{1,2})([/.\-])(\d{1,2})\2(\d{2,4})(?:[T\s](\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?))?$"
)


def classify_and_parse_single_date(
    val: Any,
    column_dayfirst_hint: Optional[bool] = None,
) -> tuple[RowDateStatus, Optional[pd.Timestamp], dict[str, Any]]:
    """Classify and parse a single raw value into a Timestamp with ambiguity diagnosis.

    Returns:
        Tuple of (RowDateStatus, parsed_timestamp, details_dict).
    """
    if pd.isna(val):
        return RowDateStatus.MISSING, None, {"format": "missing", "reason": "Null or NA value"}

    s = str(val).strip()
    if not s or s.lower() in {"null", "none", "nan", "na", "n/a", "--", "?", "missing", "unknown"}:
        return RowDateStatus.MISSING, None, {"format": "missing", "reason": "Missing sentinel"}

    # Normalize whitespace & strip ordinal suffixes (e.g. 1st January 2026 -> 1 January 2026)
    s = re.sub(r"(\b\d{1,2})(?:st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()

    has_time = bool(re.search(r"(?:[T\s]\d{1,2}:\d{2}|\b\d{1,2}:\d{2}:\d{2}\b)", s))
    has_tz = bool(re.search(r"[T\s]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*(?:Z|[+-]\d{2}:?\d{2})$", s)) or bool(re.search(r"\b(?:UTC|GMT)[+-]?\d{0,4}\b", s))

    # 1. Check Compact Numeric Formats (8 digits YYYYMMDD or 6 digits YYMMDD)
    if re.match(r"^\d{8}$", s):
        y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
        if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
            try:
                ts = pd.Timestamp(year=y, month=m, day=d)
                return RowDateStatus.UNAMBIGUOUS, ts, {
                    "format": "COMPACT_YYYYMMDD",
                    "convention": "ISO",
                    "is_ambiguous": False,
                    "has_time": False,
                    "has_timezone": False,
                }
            except Exception:
                pass

    # 2. Check ISO / YMD (YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD)
    iso_m = ISO_DATE_REGEX.match(s)
    if iso_m:
        try:
            ts = pd.to_datetime(s, utc=has_tz)
            return RowDateStatus.UNAMBIGUOUS, ts, {
                "format": "ISO_YMD",
                "convention": "ISO",
                "is_ambiguous": False,
                "has_time": has_time,
                "has_timezone": has_tz,
            }
        except Exception:
            pass

    # 2. Check Textual Month (January 28, 2026 or 21-Aug-2025 or Aug 05, 2025)
    if TEXT_DATE_PREFIX_REGEX.search(s) or TEXT_DATE_SUFFIX_REGEX.search(s):
        try:
            ts = pd.to_datetime(s, utc=has_tz)
            return RowDateStatus.UNAMBIGUOUS, ts, {
                "format": "TEXT_MONTH",
                "convention": "TEXT_MONTH",
                "is_ambiguous": False,
                "has_time": has_time,
                "has_timezone": has_tz,
            }
        except Exception:
            pass

    # 3. Check Separator-based (N1 / N2 / N3)
    sep_m = SEP_DATE_REGEX.match(s)
    if sep_m:
        n1 = int(sep_m.group(1))
        sep = sep_m.group(2)
        n2 = int(sep_m.group(3))
        n3 = int(sep_m.group(4))

        # Check for unambiguous day vs month
        if n1 > 12 and 1 <= n2 <= 12:
            # Must be DD-MM-YYYY (DMY)
            try:
                ts = pd.to_datetime(s, dayfirst=True, utc=has_tz)
                return RowDateStatus.UNAMBIGUOUS, ts, {
                    "format": f"DD{sep}MM{sep}YYYY",
                    "convention": "DMY",
                    "is_ambiguous": False,
                    "evidence": f"Day {n1} > 12",
                    "has_time": has_time,
                    "has_timezone": has_tz,
                }
            except Exception:
                return RowDateStatus.INVALID, None, {"format": "INVALID_DATE", "reason": f"Invalid date components: {s}"}

        elif 1 <= n1 <= 12 and n2 > 12:
            # Must be MM-DD-YYYY (MDY)
            try:
                ts = pd.to_datetime(s, dayfirst=False, utc=has_tz)
                return RowDateStatus.UNAMBIGUOUS, ts, {
                    "format": f"MM{sep}DD{sep}YYYY",
                    "convention": "MDY",
                    "is_ambiguous": False,
                    "evidence": f"Day {n2} > 12",
                    "has_time": has_time,
                    "has_timezone": has_tz,
                }
            except Exception:
                return RowDateStatus.INVALID, None, {"format": "INVALID_DATE", "reason": f"Invalid date components: {s}"}

        elif 1 <= n1 <= 12 and 1 <= n2 <= 12:
            if n1 == n2:
                # Same day and month (e.g. 05/05/2025)
                try:
                    ts = pd.to_datetime(s, dayfirst=False, utc=has_tz)
                    return RowDateStatus.UNAMBIGUOUS, ts, {
                        "format": f"DD{sep}MM{sep}YYYY",
                        "convention": "IDENTICAL_DM",
                        "is_ambiguous": False,
                        "has_time": has_time,
                        "has_timezone": has_tz,
                    }
                except Exception:
                    pass

            # Strictly Ambiguous: both n1 and n2 are <= 12 and n1 != n2
            # E.g. 12/10/2025 or 06.07.2025
            df_opt = column_dayfirst_hint if column_dayfirst_hint is not None else False
            try:
                tentative_ts = pd.to_datetime(s, dayfirst=df_opt, utc=has_tz)
            except Exception:
                tentative_ts = None

            return RowDateStatus.AMBIGUOUS, tentative_ts, {
                "format": f"AMBIGUOUS_SEPARATOR ({sep})",
                "convention": "AMBIGUOUS",
                "is_ambiguous": True,
                "evidence": f"Both tokens ({n1}, {n2}) <= 12",
                "possible_interpretations": [
                    f"{n1:02d}/{n2:02d}/{n3} (DD/MM/YYYY)",
                    f"{n2:02d}/{n1:02d}/{n3} (MM/DD/YYYY)",
                ],
                "has_time": has_time,
                "has_timezone": has_tz,
            }

        else:
            # Impossible month or day (e.g. 13/35/2025)
            return RowDateStatus.INVALID, None, {"format": "INVALID_DATE", "reason": f"Impossible date numbers: {s}"}

    # 4. Fallback generic parser attempt
    try:
        ts = pd.to_datetime(s, errors="raise", utc=has_tz)
        return RowDateStatus.UNAMBIGUOUS, ts, {
            "format": "GENERIC_PARSED",
            "convention": "UNKNOWN",
            "is_ambiguous": False,
            "has_time": has_time,
            "has_timezone": has_tz,
        }
    except Exception:
        return RowDateStatus.INVALID, None, {"format": "UNPARSEABLE", "reason": f"Could not parse '{s}' as date"}


def analyze_date_column(series: pd.Series) -> DateNormalizationResult:
    """Analyze an entire column for date formats, global convention evidence, and ambiguity.

    Args:
        series: Pandas Series to inspect.

    Returns:
        DateNormalizationResult with full audit diagnostic info.
    """
    col_name = str(series.name) if series.name else "date_column"
    non_null = series.dropna()
    total_non_null = len(non_null)

    if total_non_null == 0:
        return DateNormalizationResult(
            column=col_name,
            inferred_convention="EMPTY",
            parse_ratio=0.0,
            confidence=0.0,
        )

    # First pass: Gather global column-wide evidence
    dmy_evidence_count = 0
    mdy_evidence_count = 0
    iso_count = 0
    text_count = 0
    ambiguous_count = 0
    unambiguous_count = 0
    invalid_count = 0
    has_time_count = 0
    has_tz_count = 0
    invalid_samples: list[str] = []

    for idx, raw_val in non_null.items():
        status, ts, diag = classify_and_parse_single_date(raw_val)
        if diag.get("has_time"):
            has_time_count += 1
        if diag.get("has_timezone"):
            has_tz_count += 1

        if status == RowDateStatus.UNAMBIGUOUS:
            unambiguous_count += 1
            conv = diag.get("convention")
            if conv == "DMY":
                dmy_evidence_count += 1
            elif conv == "MDY":
                mdy_evidence_count += 1
            elif conv == "ISO":
                iso_count += 1
            elif conv == "TEXT_MONTH":
                text_count += 1

        elif status == RowDateStatus.AMBIGUOUS:
            ambiguous_count += 1

        elif status == RowDateStatus.INVALID:
            invalid_count += 1
            if len(invalid_samples) < 5:
                invalid_samples.append(str(raw_val))

    # Check column name hints
    col_lower = col_name.lower()
    col_hints_dmy = any(h in col_lower for h in ["dmy", "ddmmyy", "dd_mm", "dayfirst"])
    col_hints_mdy = any(h in col_lower for h in ["mdy", "mmddyy", "mm_dd", "monthfirst"])

    # Determine Column Convention & Ambiguity
    mixed_conventions = (dmy_evidence_count > 0 and mdy_evidence_count > 0)
    dayfirst = False

    if mixed_conventions:
        convention = "MIXED"
        dayfirst = False
        is_ambiguous = True
        confidence = 0.65
    elif dmy_evidence_count > 0 and mdy_evidence_count == 0:
        convention = "DMY"
        dayfirst = True
        is_ambiguous = False
        confidence = 0.98
    elif mdy_evidence_count > 0 and dmy_evidence_count == 0:
        convention = "MDY"
        dayfirst = False
        is_ambiguous = False
        confidence = 0.98
    elif iso_count >= (total_non_null * 0.5):
        convention = "ISO"
        dayfirst = False
        is_ambiguous = False
        confidence = 0.99
    elif text_count >= (total_non_null * 0.5):
        convention = "TEXT_MONTH"
        dayfirst = False
        is_ambiguous = False
        confidence = 0.98
    elif col_hints_dmy and not col_hints_mdy:
        convention = "DMY"
        dayfirst = True
        is_ambiguous = False
        confidence = 0.90
    elif col_hints_mdy and not col_hints_dmy:
        convention = "MDY"
        dayfirst = False
        is_ambiguous = False
        confidence = 0.90
    elif ambiguous_count > 0 and unambiguous_count == 0:
        convention = "AMBIGUOUS"
        dayfirst = False
        is_ambiguous = True
        confidence = 0.70
    else:
        convention = "ISO" if iso_count > 0 else ("DMY" if dmy_evidence_count > 0 else "MDY")
        is_ambiguous = ambiguous_count > 0
        confidence = 0.80

    # Second pass: Generate row diagnostics using the established global convention
    row_diagnostics: list[dict[str, Any]] = []
    for idx, raw_val in non_null.items():
        status, ts, diag = classify_and_parse_single_date(raw_val, column_dayfirst_hint=dayfirst)
        diag["index"] = idx
        diag["raw_value"] = str(raw_val)
        diag["status"] = status.value
        diag["parsed_iso"] = ts.isoformat() if ts is not None else None
        # Pre-format the cleaned value as DD-MM-YY for UI previews
        diag["formatted_ddmmyy"] = safe_format_timestamp(ts, DEFAULT_DISPLAY_FORMAT)
        if status == RowDateStatus.AMBIGUOUS and not is_ambiguous:
            # Resolved by global column convention
            diag["status"] = RowDateStatus.UNAMBIGUOUS.value
            diag["reason"] = f"Resolved via global column {convention} convention"
        row_diagnostics.append(diag)

    parse_ratio = (unambiguous_count + (ambiguous_count if not is_ambiguous else 0)) / total_non_null if total_non_null > 0 else 0.0

    return DateNormalizationResult(
        column=col_name,
        inferred_convention=convention,
        dayfirst=dayfirst,
        is_ambiguous=is_ambiguous,
        has_time=has_time_count > 0,
        has_timezone=has_tz_count > 0,
        total_non_null=total_non_null,
        parsed_count=unambiguous_count + (ambiguous_count if not is_ambiguous else 0),
        unambiguous_count=unambiguous_count,
        ambiguous_count=ambiguous_count,
        invalid_count=invalid_count,
        mixed_conventions_detected=mixed_conventions,
        invalid_samples=invalid_samples,
        row_diagnostics=row_diagnostics[:100],
        parse_ratio=round(parse_ratio, 3),
        confidence=confidence,
    )


def normalize_date_column(
    df: pd.DataFrame,
    column: str,
    dayfirst: Optional[bool] = None,
    display_format: Optional[str] = DEFAULT_DISPLAY_FORMAT,
) -> tuple[pd.DataFrame, RepairRecord]:
    """Normalize a date column using row-aware parsing.

    Behaviour
    ---------
    * Parses every cell individually using global column-evidence for ambiguity.
    * Internally stores the column as datetime64[ns].
    * When *display_format* is set (default ``"%d-%m-%y"``), also writes a
      ``{column}_display`` string column formatted as DD-MM-YY.  The original
      datetime column is kept so downstream operations can still do date maths.
    * Ambiguous / invalid cells become NaT in the datetime column and empty
      string in the display column.

    Args:
        df:             Source DataFrame (not mutated).
        column:         Column name to normalize.
        dayfirst:       Explicit dayfirst override.  If None, inferred.
        display_format: strftime format for the output display column.
                        Defaults to DEFAULT_DISPLAY_FORMAT = "%d-%m-%y".

    Returns:
        Tuple of (repaired DataFrame, RepairRecord).
    """
    if column not in df.columns:
        raise RepairValidationError(f"Column '{column}' does not exist in dataset.")

    result = df.copy()
    diag = analyze_date_column(result[column])
    effective_dayfirst = dayfirst if dayfirst is not None else diag.dayfirst
    effective_display_fmt = display_format if display_format is not None else DEFAULT_DISPLAY_FORMAT

    # Row-by-row safe transformation
    parsed_values: list[Any] = []
    ambiguous_rows_indices: list[Any] = []

    for idx, val in result[column].items():
        status, ts, row_info = classify_and_parse_single_date(val, column_dayfirst_hint=effective_dayfirst)
        if status == RowDateStatus.MISSING:
            parsed_values.append(pd.NaT)
        elif status == RowDateStatus.UNAMBIGUOUS:
            parsed_values.append(ts)
        elif status == RowDateStatus.AMBIGUOUS:
            ambiguous_rows_indices.append(idx)
            # Column has a clearly resolved single convention → use it
            if not diag.mixed_conventions_detected and diag.confidence >= 0.85:
                parsed_values.append(ts)
            else:
                parsed_values.append(ts if ts is not None else pd.NaT)
        else:  # INVALID
            parsed_values.append(pd.NaT)

    try:
        parse_utc = diag.has_timezone
        converted_series = pd.Series(
            pd.to_datetime(parsed_values, utc=parse_utc, errors="coerce"),
            index=result.index,
        )
    except Exception:
        converted_series = pd.Series(parsed_values, index=result.index)

    valid_count = converted_series.notna().sum()
    original_non_null = result[column].notna().sum()
    lost_count = max(0, original_non_null - valid_count)

    # ── Store datetime64 internally ──
    result[column] = converted_series

    # ── Build display-formatted string column (DD-MM-YY by default) ──
    # Rows that are NaT become empty string so the review table shows them cleanly.
    display_series = format_date_series(converted_series, fmt=effective_display_fmt, na_value="")
    display_col = f"{column}_display"
    result[display_col] = display_series

    # ── Determine audit status ──
    if diag.mixed_conventions_detected:
        op_status = "flagged"
        reason = (
            f"MIXED DATE CONVENTIONS DETECTED — "
            f"{diag.ambiguous_count} ambiguous row(s) flagged for review"
        )
    elif diag.is_ambiguous:
        op_status = "flagged"
        reason = (
            f"AMBIGUOUS DATE CONVENTION — "
            f"{diag.ambiguous_count} row(s) have day/month ≤ 12 without definitive evidence"
        )
    else:
        op_status = "applied"
        reason = (
            f"Normalized {valid_count} dates → datetime64 ({diag.inferred_convention}); "
            f"display column '{display_col}' formatted as {effective_display_fmt}"
        )

    # Sample of formatted display values for audit trail
    display_sample = display_series.dropna().head(10).tolist()

    record = RepairRecord(
        operation="normalize_dates",
        timestamp=datetime.now(),
        rows_before=len(df),
        rows_after=len(result),
        columns_before=len(df.columns),
        columns_after=len(result.columns),
        success=True,
        column=column,
        issue_type=IssueTaxonomy.DATE.value,
        original_value=f"{original_non_null} values ({diag.inferred_convention})",
        new_value=f"datetime64 + {display_col} ({effective_display_fmt})",
        method="deterministic_row_aware_dates_engine",
        confidence=diag.confidence,
        reason=reason,
        status=op_status,
        risk_level="medium" if (diag.is_ambiguous or diag.mixed_conventions_detected) else "safe",
        details={
            "repair_type": "date_normalization",
            "column": column,
            "display_column": display_col,
            "display_format": effective_display_fmt,
            "inferred_convention": diag.inferred_convention,
            "dayfirst": effective_dayfirst,
            "is_ambiguous": diag.is_ambiguous,
            "mixed_conventions_detected": diag.mixed_conventions_detected,
            "has_time": diag.has_time,
            "has_timezone": diag.has_timezone,
            "original_non_null": original_non_null,
            "parsed_count": valid_count,
            "unambiguous_count": diag.unambiguous_count,
            "ambiguous_count": diag.ambiguous_count,
            "invalid_count": lost_count,
            "invalid_samples": diag.invalid_samples,
            "ambiguous_rows_indices": ambiguous_rows_indices[:50],
            "display_sample": display_sample,
            "confidence": diag.confidence,
            "status": op_status,
            "issue_type": IssueTaxonomy.DATE.value,
        },
    )

    logger.info(
        "Date normalization '{}': {} parsed ({:.1%}), display='{}', unambiguous={}, ambiguous={}, mixed={}",
        column,
        valid_count,
        (valid_count / max(original_non_null, 1)),
        effective_display_fmt,
        diag.unambiguous_count,
        diag.ambiguous_count,
        diag.mixed_conventions_detected,
    )
    return result, record


def normalize_dates(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    dayfirst: Optional[bool] = None,
    display_format: Optional[str] = DEFAULT_DISPLAY_FORMAT,
) -> tuple[pd.DataFrame, RepairRecord]:
    """Normalize multiple date columns to datetime64 + DD-MM-YY display columns."""
    target_cols = columns if columns is not None else [c for c in df.columns if df[c].dtype == "object"]
    res = df.copy()
    last_rec = None
    for col in target_cols:
        if col in res.columns:
            res, last_rec = normalize_date_column(res, col, dayfirst=dayfirst, display_format=display_format)
    if last_rec is None:
        last_rec = RepairRecord(
            operation="normalize_dates",
            timestamp=datetime.now(),
            rows_before=len(df),
            rows_after=len(res),
            columns_before=len(df.columns),
            columns_after=len(res.columns),
            success=True,
        )
    return res, last_rec


