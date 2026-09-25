import pytest
import pandas as pd
import numpy as np
from services.repair.dates import (
    analyze_date_column,
    normalize_date_column,
    format_datetime_series_for_display,
)
from services.repair.auto_dtypes import infer_semantic_datatype
from services.repair.types import convert_to_datetime


def test_mixed_date_formats_and_semantic_detection():
    """Requirement 1, 2, 6, 7, 8, 10, 16: Mixed date formats in object dtype column with generic name."""
    df = pd.DataFrame(
        {
            "abc123_var": [
                "02/27/2021",
                "15-01-2022",
                "11/05/2021",
                "2021/10/29",
                "05/16/2021",
                "Dec 05 2021",
                "25-04-2022",
                "May 19 2021",
                "2021.08.23",
                "Jul 05 2021",
                "2021.11.18",
                "2021/09/28",
                "2021-12-01 14:30:00",
            ]
        }
    )

    # 1. Semantic detection before categorical
    type_res = infer_semantic_datatype(df["abc123_var"])
    assert type_res.detected_type == "datetime"

    # 2. Normalize date column
    df_clean, summary = normalize_date_column(df, "abc123_var")

    # 3. Check internal dtype is datetime64[ns]
    assert pd.api.types.is_datetime64_any_dtype(df_clean["abc123_var"])
    assert df_clean["abc123_var"].isna().sum() == 0


def test_category_dtype_date_column():
    """Requirement 3 & 20: Category dtype date column conversion."""
    df = pd.DataFrame(
        {"cat_dates": pd.Series(["2021-01-15", "2021-02-20", "2021-03-25"]).astype("category")}
    )

    type_res = infer_semantic_datatype(df["cat_dates"])
    assert type_res.detected_type == "datetime"

    df_clean, _ = normalize_date_column(df, "cat_dates")
    assert pd.api.types.is_datetime64_any_dtype(df_clean["cat_dates"])
    assert len(df_clean) == 3


def test_valid_plus_invalid_dates_and_row_preservation():
    """Requirement 4, 13, 14, 18: Valid + invalid dates become pd.NaT without dropping rows/cols."""
    df = pd.DataFrame(
        {
            "column_7": [
                "2021-05-10",
                "invalid_date_entry",
                "2021/08/15",
                "random_string_xyz",
                "25-12-2021",
            ]
        }
    )

    df_clean, summary = normalize_date_column(df, "column_7")
    s_clean = df_clean["column_7"]

    assert len(df_clean) == len(df)
    assert len(df_clean.columns) >= len(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(s_clean)

    # Invalid entries should be pd.NaT
    assert pd.isna(s_clean.iloc[1])
    assert pd.isna(s_clean.iloc[3])

    # Valid entries should be parsed
    assert s_clean.iloc[0] == pd.Timestamp("2021-05-10")
    assert s_clean.iloc[2] == pd.Timestamp("2021-08-15")
    assert summary.details["invalid_count"] == 2


def test_valid_plus_missing_dates():
    """Requirement 5: Missing values remain pd.NaT and never string labels."""
    df = pd.DataFrame({"dates": ["2021-01-01", None, "2021-02-02", np.nan, "2021-03-03"]})

    df_clean, summary = normalize_date_column(df, "dates")
    s_clean = df_clean["dates"]

    assert pd.api.types.is_datetime64_any_dtype(s_clean)
    assert pd.isna(s_clean.iloc[1])
    assert pd.isna(s_clean.iloc[3])
    assert s_clean.isna().sum() == 2


def test_ambiguous_date_resolution():
    """Requirement 9: Ambiguous date resolution using column-wide evidence."""
    # Column has strong DMY evidence (25/04/2021, 18/05/2021)
    s = pd.Series(
        [
            "01/03/2021",  # Ambiguous (1st March or 3rd Jan)
            "25/04/2021",  # Day > 12 -> DMY
            "18/05/2021",  # Day > 12 -> DMY
            "05/06/2021",  # Ambiguous (5th June or 6th May)
        ],
        name="dmy_evidence",
    )

    analysis = analyze_date_column(s)
    assert analysis.inferred_convention == "DMY"

    df = pd.DataFrame({"dmy_evidence": s})
    df_clean, _ = normalize_date_column(df, "dmy_evidence")
    s_clean = df_clean["dmy_evidence"]

    # 01/03/2021 parsed with DMY -> 1st March 2021
    assert s_clean.iloc[0] == pd.Timestamp("2021-03-01")
    # 05/06/2021 parsed with DMY -> 5th June 2021
    assert s_clean.iloc[3] == pd.Timestamp("2021-06-05")


def test_column_named_date_that_is_text():
    """Requirement 11: A column named 'date' that is text should NOT be detected as date."""
    df = pd.DataFrame(
        {
            "date": [
                "today is a beautiful day",
                "meeting scheduled with team",
                "update release notes",
                "customer call log",
            ]
        }
    )

    type_res = infer_semantic_datatype(df["date"])
    assert type_res.detected_type != "datetime"


def test_user_facing_format_display():
    """Requirement 17: User-facing display format is DD-MM-YY."""
    s_dt = pd.Series(
        [
            pd.Timestamp("2021-02-27"),
            pd.Timestamp("2022-01-15"),
        ]
    )

    formatted = format_datetime_series_for_display(s_dt)
    assert formatted.iloc[0] == "27-02-21"
    assert formatted.iloc[1] == "15-01-22"


def test_explicit_convert_to_date_override():
    """Requirement 20 & 16: Explicit user 'Convert to Date' action overrides dtype."""
    df = pd.DataFrame(
        {
            "event_record": pd.Series(
                ["2021-04-10", "2021-05-12", "invalid_entry", "2021-06-14"]
            ).astype("category")
        }
    )

    repaired_df, op_rec = convert_to_datetime(df, columns=["event_record"])

    assert pd.api.types.is_datetime64_any_dtype(repaired_df["event_record"])
    assert len(repaired_df) == 4
    assert pd.isna(repaired_df["event_record"].iloc[2])
    assert op_rec.success is True
