"""Comprehensive date engine test suite for Task 1 requirements.

Tests every stated requirement:
- ISO dates
- DD/MM/YYYY and MM/DD/YYYY
- DD-MM-YYYY and MM-DD-YYYY
- Dot-separated dates
- Textual months
- Timestamps with time preservation
- Mixed date formats
- Global evidence resolving ambiguous rows (the KEY requirement)
- Genuinely ambiguous columns -> AMBIGUOUS (never guessed)
- Impossible dates -> INVALID
- Two-digit years
- Timezone preservation

All tests use synthetic data -- zero domain hardcoding.
"""

import pytest
import pandas as pd
import numpy as np

from services.repair.dates import (
    analyze_date_column,
    classify_and_parse_single_date,
    normalize_date_column,
    RowDateStatus,
)


# =============================================================================
# 1. ISO Dates (YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD)
# =============================================================================

class TestISODates:
    def test_iso_dash(self):
        s, ts, d = classify_and_parse_single_date("2025-04-15")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.year == 2025 and ts.month == 4 and ts.day == 15
        assert d["convention"] == "ISO"

    def test_iso_slash(self):
        s, ts, d = classify_and_parse_single_date("2026/01/28")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.year == 2026 and ts.month == 1 and ts.day == 28
        assert d["convention"] == "ISO"

    def test_iso_with_time(self):
        s, ts, d = classify_and_parse_single_date("2024-07-20 14:30:00")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.hour == 14 and ts.minute == 30 and ts.second == 0
        assert d["has_time"] is True

    def test_iso_column_detection(self):
        df = pd.DataFrame({"dt": ["2025-01-01", "2025-06-15", "2026-12-31", "2024-02-28"]})
        diag = analyze_date_column(df["dt"])
        assert diag.inferred_convention == "ISO"
        assert diag.confidence >= 0.95


# =============================================================================
# 2. DD/MM/YYYY
# =============================================================================

class TestDDMMYYYY:
    def test_unambiguous_dmy_day_gt_12(self):
        """Day > 12 proves DMY."""
        s, ts, d = classify_and_parse_single_date("25/01/2026")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.day == 25 and ts.month == 1 and ts.year == 2026
        assert d["convention"] == "DMY"

    def test_dmy_column_with_proof(self):
        """Column with at least one day > 12 must be detected as DMY."""
        df = pd.DataFrame({"date": ["25/01/2026", "31/05/2026", "14/08/2026", "02/03/2026"]})
        diag = analyze_date_column(df["date"])
        assert diag.inferred_convention == "DMY"
        assert diag.dayfirst is True

    def test_dmy_ambiguous_row_resolved_by_column(self):
        """02/03/2026 with DMY column proof -> day=2, month=3."""
        df = pd.DataFrame({"date": ["25/01/2026", "31/05/2026", "02/03/2026"]})
        res_df, record = normalize_date_column(df, "date")
        assert record.status == "applied"
        assert res_df["date"].iloc[2].day == 2
        assert res_df["date"].iloc[2].month == 3

    def test_dd_slash_mm_slash_yy(self):
        """DD/MM/YY with day > 12."""
        s, ts, d = classify_and_parse_single_date("28/03/25")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.day == 28 and ts.month == 3


# =============================================================================
# 3. MM/DD/YYYY
# =============================================================================

class TestMMDDYYYY:
    def test_unambiguous_mdy_day_gt_12(self):
        """Second component > 12 proves MDY."""
        s, ts, d = classify_and_parse_single_date("10/15/2025")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.month == 10 and ts.day == 15 and ts.year == 2025
        assert d["convention"] == "MDY"

    def test_mdy_column_with_proof(self):
        df = pd.DataFrame({"date": ["07/15/2025", "10/15/2025", "02/05/2024"]})
        diag = analyze_date_column(df["date"])
        assert diag.inferred_convention == "MDY"
        assert diag.dayfirst is False

    def test_mdy_ambiguous_row_resolved_by_column_proof(self):
        """KEY TEST -- the primary stated requirement.

        Column:
            07-15-2025  (MDY proof: day=15 > 12)
            02/05/2024  (ambiguous: both <= 12)
            10-15-2025  (MDY proof: day=15 > 12)

        Expected:
            07-15-2025 -> 2025-07-15
            02/05/2024 -> 2024-02-05  (February 5, using MDY convention)
            10-15-2025 -> 2025-10-15
        """
        df = pd.DataFrame({"date": ["07-15-2025", "02/05/2024", "10-15-2025"]})
        diag = analyze_date_column(df["date"])
        assert diag.inferred_convention == "MDY", f"Expected MDY but got {diag.inferred_convention}"
        assert diag.dayfirst is False
        assert diag.is_ambiguous is False

        res_df, record = normalize_date_column(df, "date")
        assert record.status == "applied"

        row0 = res_df["date"].iloc[0]
        assert row0.month == 7 and row0.day == 15 and row0.year == 2025

        # 02/05/2024 with MDY convention -> February 5, 2024 (NOT May 2!)
        row1 = res_df["date"].iloc[1]
        assert row1.month == 2 and row1.day == 5 and row1.year == 2024, \
            f"02/05/2024 with MDY proof must be Feb 5 2024, got {row1}"

        row2 = res_df["date"].iloc[2]
        assert row2.month == 10 and row2.day == 15 and row2.year == 2025


# =============================================================================
# 4. DD-MM-YYYY (Dash-separated DMY)
# =============================================================================

class TestDDMMYYYYDash:
    def test_unambiguous_dmy_dash_day_gt_12(self):
        s, ts, d = classify_and_parse_single_date("21-08-2025")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.day == 21 and ts.month == 8 and ts.year == 2025
        assert d["convention"] == "DMY"

    def test_dmy_dash_column_resolves_ambiguous(self):
        """Clean DMY column: 15/07/2025, 02/05/2024, 14/11/2025."""
        df = pd.DataFrame({"date": ["15/07/2025", "02/05/2024", "14/11/2025"]})
        diag = analyze_date_column(df["date"])
        assert diag.inferred_convention == "DMY", f"Expected DMY got {diag.inferred_convention}"
        assert diag.dayfirst is True

        res_df, record = normalize_date_column(df, "date")
        assert record.status == "applied"

        # 15/07/2025 -> July 15 (day=15, month=7)
        assert res_df["date"].iloc[0].day == 15
        assert res_df["date"].iloc[0].month == 7

        # 02/05/2024 with DMY convention -> May 2 (day=2, month=5)
        assert res_df["date"].iloc[1].day == 2
        assert res_df["date"].iloc[1].month == 5

        # 14/11/2025 -> November 14 (day=14, month=11)
        assert res_df["date"].iloc[2].day == 14
        assert res_df["date"].iloc[2].month == 11


# =============================================================================
# 5. MM-DD-YYYY (Dash-separated MDY)
# =============================================================================

class TestMMDDYYYYDash:
    def test_mdy_dash_day_gt_12(self):
        s, ts, d = classify_and_parse_single_date("10-15-2025")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.month == 10 and ts.day == 15 and ts.year == 2025
        assert d["convention"] == "MDY"

    def test_mdy_dash_column(self):
        df = pd.DataFrame({"date": ["10-15-2025", "06-20-2024", "03-25-2026"]})
        diag = analyze_date_column(df["date"])
        assert diag.inferred_convention == "MDY"


# =============================================================================
# 6. Dot-Separated Dates
# =============================================================================

class TestDotSeparatedDates:
    def test_dot_ambiguous(self):
        """06.07.2025 -- ambiguous (both <= 12)."""
        s, ts, d = classify_and_parse_single_date("06.07.2025")
        assert s == RowDateStatus.AMBIGUOUS
        assert d["is_ambiguous"] is True

    def test_dot_unambiguous_dmy(self):
        """31.12.2025 -- day=31 > 12 -> DMY."""
        s, ts, d = classify_and_parse_single_date("31.12.2025")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.day == 31 and ts.month == 12

    def test_dot_column_with_proof(self):
        df = pd.DataFrame({"date": ["31.12.2025", "06.07.2025", "15.04.2026"]})
        diag = analyze_date_column(df["date"])
        assert diag.inferred_convention == "DMY"
        res_df, _ = normalize_date_column(df, "date")
        # 06.07.2025 with DMY -> day=6, month=7
        assert res_df["date"].iloc[1].day == 6
        assert res_df["date"].iloc[1].month == 7


# =============================================================================
# 7. Textual Month Dates
# =============================================================================

class TestTextualMonthDates:
    def test_month_name_prefix(self):
        """Aug 05, 2025."""
        s, ts, d = classify_and_parse_single_date("Aug 05, 2025")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.month == 8 and ts.day == 5 and ts.year == 2025
        assert d["convention"] == "TEXT_MONTH"

    def test_month_name_full(self):
        """January 28, 2026."""
        s, ts, d = classify_and_parse_single_date("January 28, 2026")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.month == 1 and ts.day == 28 and ts.year == 2026

    def test_month_name_suffix_format(self):
        """21-Aug-2025 (DD-Mon-YYYY)."""
        s, ts, d = classify_and_parse_single_date("21-Aug-2025")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.day == 21 and ts.month == 8 and ts.year == 2025

    def test_textual_column(self):
        df = pd.DataFrame({
            "date": ["Jan 15, 2026", "February 28, 2026", "March 1, 2025"]
        })
        diag = analyze_date_column(df["date"])
        assert diag.is_ambiguous is False
        res_df, record = normalize_date_column(df, "date")
        assert record.success is True
        assert pd.api.types.is_datetime64_any_dtype(res_df["date"])


# =============================================================================
# 8. Timestamps / Time Preservation
# =============================================================================

class TestTimestamps:
    def test_iso_timestamp_preserved(self):
        s, ts, d = classify_and_parse_single_date("2024/12/28 20:45:24")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.hour == 20 and ts.minute == 45 and ts.second == 24
        assert d["has_time"] is True

    def test_timestamp_column_preserves_time(self):
        df = pd.DataFrame({
            "event_time": [
                "2026-01-15 08:30:00",
                "2026-01-15 14:45:22",
                "2026-01-16 23:59:59",
            ]
        })
        res_df, record = normalize_date_column(df, "event_time")
        assert record.success is True
        assert pd.api.types.is_datetime64_any_dtype(res_df["event_time"])
        assert res_df["event_time"].iloc[0].hour == 8
        assert res_df["event_time"].iloc[0].minute == 30
        assert res_df["event_time"].iloc[1].hour == 14
        assert res_df["event_time"].iloc[1].minute == 45
        assert res_df["event_time"].iloc[2].second == 59

    def test_time_not_stripped_from_mixed_column(self):
        """A date with timestamp must not lose its time."""
        df = pd.DataFrame({
            "date": [
                "2025-04-04",
                "2025/04/07 14:30:00",
            ]
        })
        res_df, record = normalize_date_column(df, "date")
        assert res_df["date"].iloc[1].hour == 14
        assert res_df["date"].iloc[1].minute == 30

    def test_midnight_not_lost(self):
        """Midnight (00:00:00) must not cause issues."""
        s, ts, d = classify_and_parse_single_date("2026-03-01 00:00:00")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.hour == 0 and ts.minute == 0 and ts.second == 0


# =============================================================================
# 9. Mixed Date Formats in Same Column
# =============================================================================

class TestMixedFormats:
    def test_mixed_representations_all_convert(self):
        """Multiple unambiguous formats in same column should all convert to datetime64."""
        df = pd.DataFrame({
            "date": [
                "2025-04-04",
                "Aug 05, 2025",
                "21-Aug-2025",
                "April 6, 2025",
                "2025/04/07 14:30:00",
            ]
        })
        res_df, record = normalize_date_column(df, "date")
        assert record.success is True
        assert pd.api.types.is_datetime64_any_dtype(res_df["date"])
        # All values should be non-null
        assert res_df["date"].notna().sum() == 5

    def test_mixed_column_row_count_preserved(self):
        df = pd.DataFrame({
            "date": [
                "2025-04-04",
                "10-15-2025",
                "08/05/2025",
                "Jan 12, 2026",
            ]
        })
        res_df, _ = normalize_date_column(df, "date")
        assert len(res_df) == 4


# =============================================================================
# 10. Genuinely Ambiguous Columns -- MUST NOT BE GUESSED
# =============================================================================

class TestAmbiguousColumns:
    def test_all_ambiguous_no_proof(self):
        """All dates with both components <= 12 -- flagged as AMBIGUOUS."""
        df = pd.DataFrame({"date": ["01/02/2025", "03/04/2025", "05/06/2025"]})
        diag = analyze_date_column(df["date"])
        assert diag.inferred_convention == "AMBIGUOUS"
        assert diag.is_ambiguous is True

    def test_ambiguous_column_status_is_flagged(self):
        """normalize_date_column on an ambiguous column must NOT return status='applied'."""
        df = pd.DataFrame({"date": ["01/02/2025", "03/04/2025", "05/06/2025"]})
        res_df, record = normalize_date_column(df, "date")
        assert record.status in ("flagged", "review_required", "ambiguous"), \
            f"Ambiguous column must be flagged, got '{record.status}'"

    def test_mixed_convention_detected(self):
        """Both DMY and MDY proof in same column -> MIXED."""
        df = pd.DataFrame({
            "date": [
                "10-15-2025",
                "28/03/25",
                "06/07/2025",
            ]
        })
        diag = analyze_date_column(df["date"])
        assert diag.mixed_conventions_detected is True
        assert diag.inferred_convention == "MIXED"

    def test_ambiguous_single_value(self):
        """Single ambiguous value without column context."""
        s, ts, d = classify_and_parse_single_date("05/06/2025")
        assert s == RowDateStatus.AMBIGUOUS
        assert d["is_ambiguous"] is True
        assert "possible_interpretations" in d


# =============================================================================
# 11. Impossible / Invalid Dates
# =============================================================================

class TestImpossibleDates:
    def test_impossible_day_32(self):
        """Day=32 is always invalid."""
        s, ts, d = classify_and_parse_single_date("32/01/2025")
        assert s == RowDateStatus.INVALID
        assert ts is None

    def test_february_31(self):
        """31 Feb never exists."""
        s, ts, d = classify_and_parse_single_date("31/02/2025")
        assert s == RowDateStatus.INVALID or ts is None

    def test_impossible_month_13(self):
        """Month=13 is always invalid."""
        s, ts, d = classify_and_parse_single_date("2025-13-40")
        assert s == RowDateStatus.INVALID
        assert ts is None

    def test_all_nines(self):
        """99/99/9999 is invalid."""
        s, ts, d = classify_and_parse_single_date("99/99/9999")
        assert s == RowDateStatus.INVALID
        assert ts is None

    def test_column_with_invalid_dates_reports_count(self):
        """Invalid dates in a column must be counted."""
        df = pd.DataFrame({
            "date": [
                "2025-01-15",
                "2025-13-01",
                "2026-04-20",
                "32/01/2025",
            ]
        })
        diag = analyze_date_column(df["date"])
        assert diag.invalid_count >= 2

    def test_invalid_dates_become_nat(self):
        """Invalid dates must become NaT in the output."""
        df = pd.DataFrame({
            "date": [
                "2025-06-15",
                "32/01/2025",
                "2025-07-20",
            ]
        })
        res_df, record = normalize_date_column(df, "date")
        assert record.success is True
        assert pd.isna(res_df["date"].iloc[1])
        assert not pd.isna(res_df["date"].iloc[0])
        assert not pd.isna(res_df["date"].iloc[2])


# =============================================================================
# 12. Two-Digit Years -- Deterministic Policy
# =============================================================================

class TestTwoDigitYears:
    def test_two_digit_year_25(self):
        """28/03/25 -> 2025-03-28 (day=28 proves DMY, year 25 -> 2025)."""
        s, ts, d = classify_and_parse_single_date("28/03/25")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert ts.day == 28 and ts.month == 3
        assert ts.year == 2025

    def test_two_digit_year_column_dmy(self):
        """DMY column with two-digit years: 28/03/25 and 15/07/24."""
        df = pd.DataFrame({"date": ["28/03/25", "15/07/24", "31/12/23"]})
        diag = analyze_date_column(df["date"])
        assert diag.inferred_convention == "DMY"
        res_df, record = normalize_date_column(df, "date")
        assert record.status == "applied"
        assert res_df["date"].iloc[0].year == 2025
        assert res_df["date"].iloc[0].month == 3
        assert res_df["date"].iloc[0].day == 28


# =============================================================================
# 13. Timezone Preservation
# =============================================================================

class TestTimezonePreservation:
    def test_utc_z_suffix_preserved(self):
        s, ts, d = classify_and_parse_single_date("2026-01-15T14:30:00Z")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert d["has_timezone"] is True
        assert ts is not None

    def test_offset_preserved(self):
        s, ts, d = classify_and_parse_single_date("2026-03-20T09:00:00+05:30")
        assert s == RowDateStatus.UNAMBIGUOUS
        assert d["has_timezone"] is True
        assert ts is not None


# =============================================================================
# 14. Unknown / Novel Date Formats -- Generic Fallback
# =============================================================================

class TestUnknownFormats:
    def test_generic_fallback_does_not_crash(self):
        """Unusual but parseable date strings must not crash."""
        test_cases = [
            "15 March 2026",
            "March 15 2026",
        ]
        for val in test_cases:
            s, ts, d = classify_and_parse_single_date(val)
            assert s in (RowDateStatus.UNAMBIGUOUS, RowDateStatus.INVALID, RowDateStatus.AMBIGUOUS)


# =============================================================================
# 15. Global Column Evidence Determines Convention (KEY STATED TESTS)
# =============================================================================

class TestGlobalEvidenceDeterminesConvention:
    def test_mdy_column_02_05_resolves_to_feb_5(self):
        """THE PRIMARY STATED TEST CASE.

        Column:
            07-15-2025  (MDY proof: 15 > 12 in position 2)
            02/05/2024  (ambiguous: both <= 12)
            10-15-2025  (MDY proof: 15 > 12 in position 2)

        Expected outputs:
            07-15-2025 -> 2025-07-15  (July 15, 2025)
            02/05/2024 -> 2024-02-05  (February 5, 2024)  <- must NOT be May 2!
            10-15-2025 -> 2025-10-15  (October 15, 2025)
        """
        df = pd.DataFrame({"date": ["07-15-2025", "02/05/2024", "10-15-2025"]})

        diag = analyze_date_column(df["date"])
        assert diag.inferred_convention == "MDY", \
            f"Column should be MDY due to proof rows, got: {diag.inferred_convention}"
        assert diag.is_ambiguous is False

        res_df, record = normalize_date_column(df, "date")
        assert record.status == "applied"

        row0 = res_df["date"].iloc[0]
        assert row0.month == 7 and row0.day == 15 and row0.year == 2025, \
            f"07-15-2025 -> expected July 15 2025, got {row0}"

        row1 = res_df["date"].iloc[1]
        assert row1.month == 2 and row1.day == 5 and row1.year == 2024, \
            f"02/05/2024 with MDY proof -> expected Feb 5 2024, got {row1}"

        row2 = res_df["date"].iloc[2]
        assert row2.month == 10 and row2.day == 15 and row2.year == 2025, \
            f"10-15-2025 -> expected Oct 15 2025, got {row2}"

    def test_dmy_column_02_05_resolves_to_may_2(self):
        """OPPOSITE TEST CASE with DMY proof.

        Column:
            15/07/2025  (DMY proof: first=15 > 12)
            02/05/2024  (ambiguous)
            14/11/2025  (DMY proof: first=14 > 12)

        Expected:
            15/07/2025 -> day=15, month=7, year=2025
            02/05/2024 -> day=2, month=5, year=2024  (May 2, not Feb 5!)
            14/11/2025 -> day=14, month=11, year=2025
        """
        df = pd.DataFrame({"date": ["15/07/2025", "02/05/2024", "14/11/2025"]})

        diag = analyze_date_column(df["date"])
        assert diag.inferred_convention == "DMY", \
            f"Column should be DMY due to proof rows, got: {diag.inferred_convention}"
        assert diag.dayfirst is True
        assert diag.is_ambiguous is False

        res_df, record = normalize_date_column(df, "date")
        assert record.status == "applied"

        row0 = res_df["date"].iloc[0]
        assert row0.day == 15 and row0.month == 7 and row0.year == 2025, \
            f"15/07/2025 -> expected day=15 month=7, got {row0}"

        # 02/05/2024 with DMY -> May 2, 2024 (day=2, month=5)
        row1 = res_df["date"].iloc[1]
        assert row1.day == 2 and row1.month == 5 and row1.year == 2024, \
            f"02/05/2024 with DMY proof -> expected day=2 month=5, got {row1}"

        row2 = res_df["date"].iloc[2]
        assert row2.day == 14 and row2.month == 11 and row2.year == 2025, \
            f"14/11/2025 -> expected day=14 month=11, got {row2}"

    def test_genuinely_ambiguous_column_never_guessed(self):
        """ALL dates <= 12 in both positions -- must be AMBIGUOUS, never guessed.

        01/02/2025, 03/04/2025, 05/06/2025 -> AMBIGUOUS -- REVIEW REQUIRED
        """
        df = pd.DataFrame({"date": ["01/02/2025", "03/04/2025", "05/06/2025"]})
        diag = analyze_date_column(df["date"])
        assert diag.inferred_convention == "AMBIGUOUS", \
            f"Genuinely ambiguous column must be AMBIGUOUS, got: {diag.inferred_convention}"
        assert diag.is_ambiguous is True

        res_df, record = normalize_date_column(df, "date")
        assert record.status in ("flagged", "review_required", "ambiguous"), \
            f"Ambiguous column must be flagged, not '{record.status}'"

    def test_single_proof_row_resolves_entire_column(self):
        """Just ONE proof row is sufficient to resolve all other ambiguous rows."""
        df = pd.DataFrame({
            "date": [
                "01/02/2025",
                "03/04/2025",
                "17/08/2025",   # DMY proof (17 > 12)
                "05/06/2025",
            ]
        })
        diag = analyze_date_column(df["date"])
        assert diag.inferred_convention == "DMY"
        assert diag.is_ambiguous is False

        res_df, record = normalize_date_column(df, "date")
        assert record.status == "applied"
        # 01/02/2025 with DMY -> day=1, month=2
        assert res_df["date"].iloc[0].day == 1
        assert res_df["date"].iloc[0].month == 2


# =============================================================================
# 16. Row Count and Data Integrity
# =============================================================================

class TestDataIntegrity:
    def test_row_count_unchanged(self):
        """Normalization must never change the row count."""
        df = pd.DataFrame({
            "date": ["2025-01-01", "2025-06-15", "not-a-date", "2026-12-31", None]
        })
        res_df, record = normalize_date_column(df, "date")
        assert len(res_df) == len(df)

    def test_other_columns_unchanged(self):
        """Repairing one date column must not modify other columns."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Carol"],
            "date": ["2025-01-01", "2025-06-15", "2026-12-31"],
        })
        res_df, _ = normalize_date_column(df, "date")
        assert list(res_df["id"]) == [1, 2, 3]
        assert list(res_df["name"]) == ["Alice", "Bob", "Carol"]

    def test_null_handling(self):
        """NaN and None input values must become NaT (not crash)."""
        df = pd.DataFrame({
            "date": ["2025-01-15", None, "2026-06-20", np.nan]
        })
        res_df, record = normalize_date_column(df, "date")
        assert record.success is True
        assert pd.isna(res_df["date"].iloc[1])
        assert pd.isna(res_df["date"].iloc[3])

    def test_sentinel_strings_become_nat(self):
        """Sentinel strings like 'NULL', 'N/A', '--' must become NaT."""
        df = pd.DataFrame({
            "date": ["2025-05-20", "NULL", "N/A", "--", "unknown", "2026-01-01"]
        })
        res_df, record = normalize_date_column(df, "date")
        assert record.success is True
        for i in [1, 2, 3, 4]:
            assert pd.isna(res_df["date"].iloc[i]), f"Row {i} should be NaT"


# =============================================================================
# 17. Date Repair Report fields (via DateNormalizationResult)
# =============================================================================

class TestDateRepairReport:
    def test_report_contains_required_fields(self):
        """DateNormalizationResult must expose all required report fields."""
        df = pd.DataFrame({"date": ["2025-01-01", "2025-06-15", "2026-12-31"]})
        diag = analyze_date_column(df["date"])
        for field in [
            "column", "inferred_convention", "confidence", "parsed_count",
            "total_non_null", "invalid_count", "ambiguous_count", "has_time",
            "mixed_conventions_detected", "row_diagnostics", "is_ambiguous",
            "unambiguous_count",
        ]:
            assert hasattr(diag, field), f"Missing field: {field}"

    def test_report_parse_ratio_reasonable(self):
        """Parse ratio for a clean ISO column should be > 0.9."""
        df = pd.DataFrame({
            "date": ["2025-01-01", "2025-06-15", "2026-12-31", "2024-03-20"]
        })
        diag = analyze_date_column(df["date"])
        assert diag.parse_ratio > 0.9

    def test_report_row_diagnostics_present(self):
        """Row diagnostics should include an entry for each non-null row."""
        df = pd.DataFrame({"date": ["2025-01-01", "2025-06-15", "2026-12-31"]})
        diag = analyze_date_column(df["date"])
        assert len(diag.row_diagnostics) == 3
        for rd in diag.row_diagnostics:
            assert "status" in rd
            assert "raw_value" in rd
