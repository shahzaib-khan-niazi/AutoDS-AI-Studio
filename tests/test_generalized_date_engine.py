"""Generalized Date Engine test suite with synthetic random date generation.

Covers all 16 specified requirement categories:
1. ISO dates
2. Slash-separated dates
3. Dash-separated dates
4. Dot-separated dates
5. Month-name dates (short)
6. Full month-name dates
7. Two-digit years
8. Four-digit years
9. Datetime values with timestamps
10. Mixed-format columns
11. Ambiguous dates (resolved via column evidence vs flagged)
12. Invalid dates
13. Null values
14. Empty strings
15. Whitespace around dates
16. Timezone-aware datetime values

Verification rule: Every successfully parsed date MUST produce DD-MM-YY string format.
"""
import random
from datetime import datetime, timezone
import pytest
import pandas as pd
import numpy as np

from services.repair.dates import (
    DEFAULT_DISPLAY_FORMAT,
    RowDateStatus,
    analyze_date_column,
    classify_and_parse_single_date,
    format_date_series,
    normalize_date_column,
)


def _to_ddmmyy(dt: datetime) -> str:
    """Expected DD-MM-YY string representation."""
    return f"{dt.day:02d}-{dt.month:02d}-{dt.year % 100:02d}"


# ---------------------------------------------------------------------------
# 1-9: Synthetic Format Parsers
# ---------------------------------------------------------------------------

class TestSyntheticFormatParsers:
    """Test 100+ randomly generated dates across 9 format categories."""

    @pytest.mark.parametrize("idx", range(30))
    def test_randomized_iso_dates(self, idx: int):
        year = random.randint(1970, 2035)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        dt = datetime(year, month, day)
        expected = _to_ddmmyy(dt)

        for sep in ["-", "/", "."]:
            raw = f"{year}{sep}{month:02d}{sep}{day:02d}"
            _status, ts, _diag = classify_and_parse_single_date(raw)
            assert ts is not None, f"Failed to parse ISO: {raw}"
            assert ts.strftime(DEFAULT_DISPLAY_FORMAT) == expected

    @pytest.mark.parametrize("idx", range(20))
    def test_randomized_month_name_dates(self, idx: int):
        year = random.randint(1970, 2035)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        dt = datetime(year, month, day)
        expected = _to_ddmmyy(dt)

        short_m = dt.strftime("%b")
        long_m = dt.strftime("%B")

        patterns = [
            f"{day:02d}-{short_m}-{year}",
            f"{day:02d}-{long_m}-{year}",
            f"{short_m} {day:02d}, {year}",
            f"{long_m} {day:02d}, {year}",
            f"{day:02d} {short_m} {year}",
            f"{day:02d} {long_m} {year}",
        ]

        for p in patterns:
            _status, ts, _diag = classify_and_parse_single_date(p)
            assert ts is not None, f"Failed to parse month name pattern: '{p}'"
            assert ts.strftime(DEFAULT_DISPLAY_FORMAT) == expected, f"Pattern '{p}' -> got {ts.strftime(DEFAULT_DISPLAY_FORMAT)}, expected {expected}"

    @pytest.mark.parametrize("idx", range(20))
    def test_randomized_timestamps_and_timezones(self, idx: int):
        year = random.randint(1970, 2035)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        dt = datetime(year, month, day, random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))
        expected = _to_ddmmyy(dt)

        ts_patterns = [
            f"{year}-{month:02d}-{day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}",
            f"{year}/{month:02d}/{day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}",
            f"{year}-{month:02d}-{day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}Z",
            f"{year}-{month:02d}-{day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}+00:00",
        ]

        for p in ts_patterns:
            _status, ts, _diag = classify_and_parse_single_date(p)
            assert ts is not None, f"Failed to parse timestamp: '{p}'"
            assert ts.strftime(DEFAULT_DISPLAY_FORMAT) == expected, f"Timestamp '{p}' -> got {ts.strftime(DEFAULT_DISPLAY_FORMAT)}, expected {expected}"

    def test_ordinal_date_parsing(self):
        ordinals = [
            ("1st January 2026", "01-01-26"),
            ("January 2nd, 2026", "02-01-26"),
            ("3rd March 2026", "03-03-26"),
            ("4th April 2026", "04-04-26"),
            ("21st May 2026", "21-05-26"),
            ("22nd June 2026", "22-06-26"),
            ("23rd July 2026", "23-07-26"),
            ("31st August 2026", "31-08-26"),
        ]
        for raw, expected in ordinals:
            status, ts, _ = classify_and_parse_single_date(raw)
            assert ts is not None, f"Failed ordinal parse: '{raw}'"
            assert ts.strftime(DEFAULT_DISPLAY_FORMAT) == expected, f"'{raw}' -> got '{ts.strftime(DEFAULT_DISPLAY_FORMAT)}', expected '{expected}'"

    def test_compact_numeric_dates(self):
        compacts = [
            ("20260731", "31-07-26"),
            ("20251217", "17-12-25"),
        ]
        for raw, expected in compacts:
            status, ts, _ = classify_and_parse_single_date(raw)
            assert ts is not None, f"Failed compact parse: '{raw}'"
            assert ts.strftime(DEFAULT_DISPLAY_FORMAT) == expected, f"'{raw}' -> got '{ts.strftime(DEFAULT_DISPLAY_FORMAT)}', expected '{expected}'"


# ---------------------------------------------------------------------------
# 10: Mixed-Format Columns
# ---------------------------------------------------------------------------

class TestMixedFormatColumnNormalization:
    """Column containing multiple distinct date formats in the SAME column."""

    def test_mixed_formats_single_column(self):
        df = pd.DataFrame({
            "order_date": [
                "2025-04-04",               # ISO
                "04/05/2025",               # Slash (DMY proof 15>12 later)
                "May 06, 2025",             # Month name
                "07-Jun-2025",              # Dash month name
                "2025.08.09",               # Dot ISO
                "10/09/25",                 # 2-digit year
                "2025/10/11 14:30:00",      # Timestamp
                "15/07/2025",               # Unambiguous DMY proof (15>12)
            ]
        })

        res_df, record = normalize_date_column(df, "order_date", display_format=DEFAULT_DISPLAY_FORMAT)
        display = res_df["order_date_display"]

        expected_ddmmyy = [
            "04-04-25",
            "04-05-25",
            "06-05-25",
            "07-06-25",
            "09-08-25",
            "10-09-25",
            "11-10-25",
            "15-07-25",
        ]

        assert list(display) == expected_ddmmyy
        assert pd.api.types.is_datetime64_any_dtype(res_df["order_date"])


# ---------------------------------------------------------------------------
# 11: Ambiguous Dates (Resolved via Column Evidence vs Flagged)
# ---------------------------------------------------------------------------

class TestAmbiguousDateHandling:

    def test_dmy_proof_resolves_ambiguous_rows(self):
        df = pd.DataFrame({
            "dt": [
                "25/08/2024",  # Proof: DMY (25 > 12)
                "03/04/2024",  # Ambiguous: 3rd April 2024 -> 03-04-24
                "18/11/2024",  # Proof: DMY (18 > 12)
            ]
        })
        res_df, _ = normalize_date_column(df, "dt", display_format=DEFAULT_DISPLAY_FORMAT)
        assert res_df["dt_display"].iloc[1] == "03-04-24"

    def test_mdy_proof_resolves_ambiguous_rows(self):
        df = pd.DataFrame({
            "dt": [
                "08/25/2024",  # Proof: MDY (25 > 12 in token 2)
                "03/04/2024",  # Ambiguous: March 4th 2024 -> DD-MM-YY = 04-03-24
                "11/18/2024",  # Proof: MDY (18 > 12 in token 2)
            ]
        })
        res_df, _ = normalize_date_column(df, "dt", display_format=DEFAULT_DISPLAY_FORMAT)
        assert res_df["dt_display"].iloc[1] == "04-03-24"

    def test_genuinely_ambiguous_column_flagged(self):
        df = pd.DataFrame({
            "dt": ["01/02/2024", "03/04/2024", "05/06/2024"]
        })
        diag = analyze_date_column(df["dt"])
        assert diag.is_ambiguous is True


# ---------------------------------------------------------------------------
# 12: Invalid Dates
# ---------------------------------------------------------------------------

class TestInvalidDateHandling:

    @pytest.mark.parametrize("invalid_raw", [
        "31/02/2025",
        "32/01/2025",
        "99/99/2025",
        "2025-15-40",
    ])
    def test_invalid_dates_flagged_as_invalid(self, invalid_raw: str):
        status, ts, diag = classify_and_parse_single_date(invalid_raw)
        assert status == RowDateStatus.INVALID
        assert ts is None

    def test_invalid_dates_in_column_become_nat_and_empty_display(self):
        df = pd.DataFrame({
            "dates": ["2025-04-04", "31/02/2025", "Aug 05, 2025"]
        })
        res_df, record = normalize_date_column(df, "dates")
        assert res_df["dates_display"].iloc[0] == "04-04-25"
        assert res_df["dates_display"].iloc[1] == ""
        assert res_df["dates_display"].iloc[2] == "05-08-25"
        assert pd.isna(res_df["dates"].iloc[1])


# ---------------------------------------------------------------------------
# 13, 14, 15: Nulls, Empty Strings, Whitespace Handling
# ---------------------------------------------------------------------------

class TestNullsEmptyAndWhitespace:

    def test_whitespace_trimmed_before_parsing(self):
        raw_values = [
            "  2025-04-04  ",
            "\tAug 05, 2025\n",
            " 21-Aug-2025 ",
        ]
        for raw in raw_values:
            status, ts, _ = classify_and_parse_single_date(raw)
            assert status == RowDateStatus.UNAMBIGUOUS, f"Failed for whitespace string: '{raw}'"
            assert ts is not None

    def test_nulls_and_empty_strings_handled_safely(self):
        s = pd.Series(["2025-04-04", "", None, np.nan, "   "])
        diag = analyze_date_column(s)
        assert diag.parsed_count == 1
        assert diag.total_non_null == 3
        assert len(s) == 5
