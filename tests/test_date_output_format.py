"""Tests for DD-MM-YY final output format standardization.

Requirement: All valid dates, regardless of input format, must produce
output formatted as DD-MM-YY (%d-%m-%y).

Test cases are taken directly from the user specification.
Zero hardcoded mappings - uses generalized datetime parsing with global convention detection.
"""
import pytest
import pandas as pd
import numpy as np

from services.repair.dates import (
    DEFAULT_DISPLAY_FORMAT,
    DateConvention,
    RowDateStatus,
    analyze_date_column,
    classify_and_parse_single_date,
    format_date_series,
    normalize_date_column,
    normalize_dates,
)


def _fmt(ts: pd.Timestamp) -> str:
    """Helper to format a single timestamp as DD-MM-YY."""
    return ts.strftime(DEFAULT_DISPLAY_FORMAT) if (ts is not None and not pd.isna(ts)) else ""


# ---------------------------------------------------------------------------
# Spec Section 1 & 2: DEFAULT_DISPLAY_FORMAT constant and helper function
# ---------------------------------------------------------------------------

class TestDefaultDisplayFormat:
    """Verify that DEFAULT_DISPLAY_FORMAT is '%d-%m-%y' and helper works."""

    def test_constant_value(self):
        assert DEFAULT_DISPLAY_FORMAT == "%d-%m-%y"

    def test_format_date_series_returns_ddmmyy(self):
        s = pd.Series(pd.to_datetime(["2024-09-01", "2024-02-05", "2026-03-04"]))
        formatted = format_date_series(s)
        assert list(formatted) == ["01-09-24", "05-02-24", "04-03-26"]

    def test_format_date_series_nat_becomes_empty_string(self):
        s = pd.Series([pd.Timestamp("2025-07-15"), pd.NaT, np.nan])
        formatted = format_date_series(s)
        assert formatted.iloc[0] == "15-07-25"
        assert formatted.iloc[1] == ""
        assert formatted.iloc[2] == ""

    def test_format_date_series_non_datetime_passthrough(self):
        s = pd.Series(["not", "dates"])
        formatted = format_date_series(s)
        assert list(formatted) == ["not", "dates"]


# ---------------------------------------------------------------------------
# Spec Section 11: Multiple formats for 2024-09-01 -> "01-09-24"
# ---------------------------------------------------------------------------

class TestSingleDateMultipleInputFormats:
    """Each of these represents the date 1st September 2024."""

    @pytest.mark.parametrize("raw", [
        "2024-09-01",       # ISO YYYY-MM-DD
        "01/09/2024",       # DD/MM/YYYY (DMY)
        "01.09.2024",       # DD.MM.YYYY (DMY)
        "01-Sep-2024",      # DD-MMM-YYYY
        "Sep 01, 2024",     # MMM DD, YYYY
        "01/09/24",         # DD/MM/YY
        "2024/09/01",       # YYYY/MM/DD
    ])
    def test_all_formats_parse_to_01_09_24(self, raw: str):
        """Every listed format must parse to 1st September 2024 when parsed in DMY context."""
        s, ts, d = classify_and_parse_single_date(raw, column_dayfirst_hint=True)
        assert s in (RowDateStatus.UNAMBIGUOUS, RowDateStatus.AMBIGUOUS), f"'{raw}' unexpected status: {s}"
        assert ts is not None, f"'{raw}' should produce a valid timestamp"
        assert ts.year == 2024 and ts.month == 9 and ts.day == 1, \
            f"'{raw}' -> expected 2024-09-01, got {ts}"
        assert _fmt(ts) == "01-09-24", \
            f"'{raw}' formatted output should be '01-09-24', got '{_fmt(ts)}'"

    def test_column_of_all_sep_formats_outputs_01_09_24(self):
        """Whole column with all formats should produce identical DD-MM-YY output."""
        df = pd.DataFrame({"date": [
            "2024-09-01",
            "01-Sep-2024",
            "Sep 01, 2024",
            "2024/09/01",
        ]})
        diag = analyze_date_column(df["date"])
        assert diag.is_ambiguous is False
        res_df, record = normalize_date_column(df, "date", display_format=DEFAULT_DISPLAY_FORMAT)
        display = res_df["date_display"]
        assert all(v == "01-09-24" for v in display), \
            f"All rows should be '01-09-24', got: {list(display)}"


# ---------------------------------------------------------------------------
# Spec Section 11: Multiple formats for 2026-03-04 -> "04-03-26"
# ---------------------------------------------------------------------------

class TestSingleDateMultipleInputFormats2:
    """Each of these represents the date 4th March 2026."""

    @pytest.mark.parametrize("raw", [
        "2026-03-04",       # ISO
        "04/03/2026",       # DD/MM/YYYY
        "04-Mar-2026",      # DD-MMM-YYYY
        "March 4, 2026",    # MMMM DD, YYYY
    ])
    def test_all_formats_parse_to_04_03_26(self, raw: str):
        s, ts, d = classify_and_parse_single_date(raw, column_dayfirst_hint=True)
        assert s in (RowDateStatus.UNAMBIGUOUS, RowDateStatus.AMBIGUOUS), f"'{raw}' unexpected status: {s}"
        assert ts is not None
        assert ts.year == 2026 and ts.month == 3 and ts.day == 4, \
            f"'{raw}' -> expected 2026-03-04, got {ts}"
        assert _fmt(ts) == "04-03-26", \
            f"'{raw}' formatted output should be '04-03-26', got '{_fmt(ts)}'"


# ---------------------------------------------------------------------------
# Spec Section 1: Example conversions
# ---------------------------------------------------------------------------

class TestSpecExampleConversions:
    """All exact examples from the user prompt mapped to expected DD-MM-YY."""

    def test_iso_2024_09_01_to_01_09_24(self):
        df = pd.DataFrame({"d": ["2024-09-01"]})
        res_df, _ = normalize_date_column(df, "d")
        assert res_df["d_display"].iloc[0] == "01-09-24"

    def test_21_aug_2025_to_21_08_25(self):
        df = pd.DataFrame({"d": ["21-Aug-2025"]})
        res_df, _ = normalize_date_column(df, "d")
        assert res_df["d_display"].iloc[0] == "21-08-25"

    def test_january_28_2026_to_28_01_26(self):
        df = pd.DataFrame({"d": ["January 28, 2026"]})
        res_df, _ = normalize_date_column(df, "d")
        assert res_df["d_display"].iloc[0] == "28-01-26"

    def test_2025_slash_07_slash_15_to_15_07_25(self):
        df = pd.DataFrame({"d": ["2025/07/15"]})
        res_df, _ = normalize_date_column(df, "d")
        assert res_df["d_display"].iloc[0] == "15-07-25"

    def test_aug_05_2025_to_05_08_25(self):
        df = pd.DataFrame({"d": ["Aug 05, 2025"]})
        res_df, _ = normalize_date_column(df, "d")
        assert res_df["d_display"].iloc[0] == "05-08-25"


# ---------------------------------------------------------------------------
# Spec Section 7: Mixed column with many different formats -> uniform DD-MM-YY
# ---------------------------------------------------------------------------

class TestMixedColumnOutputFormat:
    """Mixed column containing 10 diverse formats must all produce DD-MM-YY."""

    @pytest.fixture
    def mixed_df(self):
        return pd.DataFrame({
            "order_date": [
                "2024-09-01",              # ISO -> 01-09-24
                "21-Aug-2025",             # DD-MMM-YYYY -> 21-08-25
                "January 28, 2026",        # MMMM DD, YYYY -> 28-01-26
                "2024/12/28 20:45:24",     # ISO datetime -> 28-12-24
                "Aug 05, 2025",            # MMM DD, YYYY -> 05-08-25
                "2025-04-04",              # ISO -> 04-04-25
                "28/03/25",                # DD/MM/YY (unambiguous 28>12) -> 28-03-25
                "15/07/2025",              # DD/MM/YYYY (unambiguous 15>12) -> 15-07-25
            ]
        })

    def test_unambiguous_rows_all_formatted_ddmmyy(self, mixed_df):
        res_df, record = normalize_date_column(mixed_df, "order_date", display_format=DEFAULT_DISPLAY_FORMAT)
        display = res_df["order_date_display"]

        expected = [
            "01-09-24",
            "21-08-25",
            "28-01-26",
            "28-12-24",
            "05-08-25",
            "04-04-25",
            "28-03-25",
            "15-07-25",
        ]
        assert list(display) == expected

    def test_display_column_created(self, mixed_df):
        res_df, _ = normalize_date_column(mixed_df, "order_date")
        assert "order_date_display" in res_df.columns

    def test_datetime_column_remains_datetime64(self, mixed_df):
        res_df, _ = normalize_date_column(mixed_df, "order_date")
        assert pd.api.types.is_datetime64_any_dtype(res_df["order_date"])


# ---------------------------------------------------------------------------
# Spec Section 3: Ambiguous dates resolved via column convention
# ---------------------------------------------------------------------------

class TestAmbiguousDateConvention:
    """Ambiguous rows resolve correctly according to the detected column convention."""

    def test_dmy_column_02_05_formats_as_02_05_24(self):
        """In a DMY column (evidenced by 21/08/2024), 02/05/2024 -> 2nd May -> '02-05-24'."""
        df = pd.DataFrame({"date": [
            "21/08/2024",   # DMY proof (21 > 12)
            "02/05/2024",   # Ambiguous: 2nd May 2024
            "15/07/2024",   # DMY proof (15 > 12)
        ]})
        res_df, _ = normalize_date_column(df, "date", display_format=DEFAULT_DISPLAY_FORMAT)
        display = res_df["date_display"]
        assert display.iloc[1] == "02-05-24", \
            f"Expected '02-05-24' (2nd May), got '{display.iloc[1]}'"

    def test_mdy_column_02_05_formats_as_05_02_24(self):
        """In an MDY column (evidenced by 08/21/2024), 02/05/2024 -> Feb 5th -> '05-02-24'."""
        df = pd.DataFrame({"date": [
            "08/21/2024",   # MDY proof (21 > 12 in 2nd token)
            "02/05/2024",   # Ambiguous: Feb 5th 2024 -> DD-MM-YY is 05-02-24
            "07/15/2024",   # MDY proof (15 > 12 in 2nd token)
        ]})
        res_df, _ = normalize_date_column(df, "date", display_format=DEFAULT_DISPLAY_FORMAT)
        display = res_df["date_display"]
        assert display.iloc[1] == "05-02-24", \
            f"Expected '05-02-24' (5th Feb), got '{display.iloc[1]}'"

    def test_genuinely_ambiguous_column_display_flagged(self):
        """When convention cannot be determined, ambiguous rows retain placeholder/flag."""
        df = pd.DataFrame({"date": ["02/05/2024", "06/07/2024"]})
        diag = analyze_date_column(df["date"])
        assert diag.is_ambiguous is True


# ---------------------------------------------------------------------------
# Spec Section 4: Invalid dates -> NaT, empty display
# ---------------------------------------------------------------------------

class TestInvalidDatesProduceNaT:
    """Invalid calendar dates must not raise errors; they produce NaT and empty display."""

    def test_32_01_2025_produces_nat_and_empty_display(self):
        df = pd.DataFrame({"date": ["32/01/2025"]})
        res_df, record = normalize_date_column(df, "date", display_format=DEFAULT_DISPLAY_FORMAT)
        assert pd.isna(res_df["date"].iloc[0])
        assert res_df["date_display"].iloc[0] == ""

    def test_31_02_2025_is_invalid(self):
        status, ts, diag = classify_and_parse_single_date("31/02/2025")
        assert status == RowDateStatus.INVALID
        assert ts is None

    def test_valid_rows_around_invalid_still_format(self):
        df = pd.DataFrame({"date": [
            "2024-09-01",
            "99/99/9999",   # Invalid
            "21-Aug-2025",
        ]})
        res_df, _ = normalize_date_column(df, "date", display_format=DEFAULT_DISPLAY_FORMAT)
        assert res_df["date_display"].iloc[0] == "01-09-24"
        assert res_df["date_display"].iloc[1] == ""
        assert res_df["date_display"].iloc[2] == "21-08-25"


# ---------------------------------------------------------------------------
# Spec Section 5: Timestamps -> Date part extracted for DD-MM-YY display
# ---------------------------------------------------------------------------

class TestTimestampDateFormatted:
    """Timestamps must have their date component correctly formatted as DD-MM-YY."""

    def test_timestamp_date_portion_formatted(self):
        status, ts, diag = classify_and_parse_single_date("2024/12/28 20:45:24")
        assert ts is not None
        assert _fmt(ts) == "28-12-24"

    def test_timestamp_column_display_is_date_only(self):
        df = pd.DataFrame({"ts": ["2024-09-01 14:30:00", "2025-07-15 08:00:00"]})
        res_df, _ = normalize_date_column(df, "ts", display_format=DEFAULT_DISPLAY_FORMAT)
        assert res_df["ts_display"].iloc[0] == "01-09-24"
        assert res_df["ts_display"].iloc[1] == "15-07-25"

    def test_time_preserved_in_datetime64_column(self):
        df = pd.DataFrame({"ts": ["2024-09-01 14:30:00"]})
        res_df, _ = normalize_date_column(df, "ts")
        val = res_df["ts"].iloc[0]
        assert val.hour == 14 and val.minute == 30


# ---------------------------------------------------------------------------
# Spec Section 6: Row diagnostics include formatted_ddmmyy field
# ---------------------------------------------------------------------------

class TestRowDiagnosticsContainFormattedValue:
    """Row-level diagnostics returned by analyze_date_column must include formatted_ddmmyy."""

    def test_row_diagnostics_have_formatted_ddmmyy(self):
        s = pd.Series(["2024-09-01", "21-Aug-2025", "invalid"])
        diag = analyze_date_column(s)
        for r in diag.row_diagnostics:
            assert "formatted_ddmmyy" in r, f"Key 'formatted_ddmmyy' missing from row diag: {r}"

    def test_row_diagnostics_ddmmyy_correct(self):
        s = pd.Series(["2024-09-01", "21-Aug-2025"])
        diag = analyze_date_column(s)
        assert diag.row_diagnostics[0]["formatted_ddmmyy"] == "01-09-24"
        assert diag.row_diagnostics[1]["formatted_ddmmyy"] == "21-08-25"

    def test_invalid_row_has_empty_formatted(self):
        s = pd.Series(["not_a_date"])
        diag = analyze_date_column(s)
        assert diag.row_diagnostics[0]["formatted_ddmmyy"] == ""


# ---------------------------------------------------------------------------
# Spec Section 7: format_date_series edge cases
# ---------------------------------------------------------------------------

class TestFormatDateSeriesEdgeCases:

    def test_custom_format(self):
        s = pd.Series(pd.to_datetime(["2024-09-01"]))
        formatted = format_date_series(s, date_format="%Y/%m/%d")
        assert formatted.iloc[0] == "2024/09/01"

    def test_all_nat_produces_all_empty(self):
        s = pd.Series([pd.NaT, pd.NaT, np.nan])
        formatted = format_date_series(s)
        assert (formatted == "").all()

    def test_custom_na_value(self):
        s = pd.Series([pd.Timestamp("2024-09-01"), pd.NaT])
        formatted = format_date_series(s, na_value="MISSING")
        assert formatted.iloc[0] == "01-09-24"
        assert formatted.iloc[1] == "MISSING"


class TestPre1900WindowsSafeFormatting:
    """Test that pre-1900 dates never raise Windows strftime ValueError."""

    def test_pre_1900_safe_format_timestamp(self):
        from services.repair.dates import safe_format_timestamp
        ts = pd.Timestamp("1850-05-12")
        assert safe_format_timestamp(ts) == "12-05-50"

    def test_pre_1900_series_format(self):
        s = pd.Series([pd.Timestamp("1850-05-12"), pd.Timestamp("2024-09-01")])
        formatted = format_date_series(s)
        assert list(formatted) == ["12-05-50", "01-09-24"]

    def test_pre_1900_analyze_date_column(self):
        s = pd.Series(["1850-05-12", "2024-09-01"])
        diag = analyze_date_column(s)
        assert diag.row_diagnostics[0]["formatted_ddmmyy"] == "12-05-50"
        assert diag.row_diagnostics[1]["formatted_ddmmyy"] == "01-09-24"
