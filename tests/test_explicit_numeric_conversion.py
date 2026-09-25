import pandas as pd
import numpy as np
import pytest

from models.repair import RepairAction, RepairOperation
from services.repair.types import convert_series_to_numeric_explicit, convert_to_numeric
from services.repair.service import RepairService


def test_case_a_mixed_numeric_and_written_numbers():
    """Test Case A: 34, 43, 34, ten, five -> 34, 43, 34, 10, 5 (Int64)."""
    df = pd.DataFrame({
        "age": [34, 43, 34, "ten", "five"]
    })

    conv_s, metrics = convert_series_to_numeric_explicit(df["age"])

    assert str(conv_s.dtype) == "Int64"
    assert conv_s.tolist() == [34, 43, 34, 10, 5]
    assert metrics["written_numbers"] == 2
    assert metrics["already_numeric"] == 3


def test_case_b_mixed_numeric_currency_unrecoverable():
    """Test Case B: Valid numerics/currencies remain numeric, unrecoverables become pd.NA."""
    df = pd.DataFrame({
        "balance": ["$50,000", "50K", "100", "N/A", "unknown", "-"]
    })

    conv_s, metrics = convert_series_to_numeric_explicit(df["balance"])

    assert str(conv_s.dtype) == "Int64"
    assert len(conv_s) == 6
    assert conv_s.iloc[0] == 50000
    assert conv_s.iloc[1] == 50000
    assert conv_s.iloc[2] == 100
    assert pd.isna(conv_s.iloc[3])
    assert pd.isna(conv_s.iloc[4])
    assert pd.isna(conv_s.iloc[5])
    assert metrics["unavailable_values"] == 3


def test_case_c_pure_categorical_becomes_na():
    """Test Case C: Red, Blue, Green -> all unparseable text becomes pd.NA rather than inventing category codes."""
    df = pd.DataFrame({
        "color": ["Red", "Blue", "Green", "Blue"]
    })

    conv_s, metrics = convert_series_to_numeric_explicit(df["color"])

    assert conv_s.isna().all()
    assert metrics["unparseable_text_to_na"] == 4
    assert metrics["categorical_converted_to_numeric_codes"] == 0
    assert len(metrics["affected_cells"]) == 4


def test_case_d_mixed_numeric_written_and_arbitrary_text():
    """Test Case D: 100, 200, random_text, 300, bad_value, 400 -> 100, 200, NA, 300, NA, 400."""
    s = pd.Series([100, 200, "random_text", 300, "bad_value", 400], name="val")
    conv_s, metrics = convert_series_to_numeric_explicit(s)

    assert str(conv_s.dtype) == "Int64"
    assert conv_s.iloc[0] == 100
    assert conv_s.iloc[1] == 200
    assert pd.isna(conv_s.iloc[2])
    assert conv_s.iloc[3] == 300
    assert pd.isna(conv_s.iloc[4])
    assert conv_s.iloc[5] == 400
    assert metrics["unparseable_text_to_na"] == 2


def test_case_e_lineage_recorded_for_na_conversions():
    """Test Case E: Lineage records row index, original value, cleaned value, and reason for pd.NA conversions."""
    df = pd.DataFrame({"col_x": [9044, "free", 35962.04, "anything"]})
    conv_s, metrics = convert_series_to_numeric_explicit(df["col_x"])

    assert str(conv_s.dtype) == "Float64"
    assert conv_s.iloc[0] == 9044.0
    assert pd.isna(conv_s.iloc[1])
    assert conv_s.iloc[2] == 35962.04
    assert pd.isna(conv_s.iloc[3])

    cells = metrics["affected_cells"]
    assert len(cells) == 2
    assert cells[0]["row_index"] == 1
    assert cells[0]["original_value"] == "free"
    assert pd.isna(cells[0]["cleaned_value"])
    assert cells[0]["reason"] == "not reliably interpretable as numeric"


def test_case_f_renamed_columns_arbitrary_names():
    """Test Case F: Rename columns to arbitrary names; behavior is 100% value-driven."""
    df1 = pd.DataFrame({"banana_column": [100, "twenty", "Gold"]})
    df2 = pd.DataFrame({"field_17": [100, "twenty", "Gold"]})
    df3 = pd.DataFrame({"random_header": [100, "twenty", "Gold"]})

    s1, m1 = convert_series_to_numeric_explicit(df1["banana_column"])
    s2, m2 = convert_series_to_numeric_explicit(df2["field_17"])
    s3, m3 = convert_series_to_numeric_explicit(df3["random_header"])

    # 100 -> 100, "twenty" -> 20, "Gold" -> pd.NA
    assert s1.iloc[0] == s2.iloc[0] == s3.iloc[0] == 100
    assert s1.iloc[1] == s2.iloc[1] == s3.iloc[1] == 20
    assert pd.isna(s1.iloc[2]) and pd.isna(s2.iloc[2]) and pd.isna(s3.iloc[2])


def test_case_g_row_and_column_count_strictly_preserved():
    """Test Case G: Row count and column count remain exactly unchanged."""
    df = pd.DataFrame({
        "col_a": ["10", "twenty", "30,000", "$40K", "Special", None],
        "col_b": [1, 2, 3, 4, 5, 6],
    })

    orig_rows, orig_cols = len(df), len(df.columns)
    action = RepairAction(
        operation=RepairOperation.CONVERT_TYPES,
        target=["col_a"],
        parameters={"target_type": "numeric"},
        reason="Explicit user numeric conversion",
    )

    repaired_df, result = RepairService.repair(df, [action])

    assert result.success is True
    assert len(repaired_df) == orig_rows
    assert len(repaired_df.columns) == orig_cols
    assert list(repaired_df.columns) == list(df.columns)
    assert str(repaired_df["col_a"].dtype) == "Int64"
    assert repaired_df["col_b"].tolist() == [1, 2, 3, 4, 5, 6]
    assert pd.isna(repaired_df["col_a"].iloc[4])  # "Special" -> pd.NA
