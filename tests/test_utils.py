"""Tests for utility modules."""

import datetime
import pandas as pd
import numpy as np
import pytest

from utils.dataframe import (
    safe_copy,
    make_arrow_safe_preview,
    safe_numeric_columns,
    safe_datetime_columns,
    safe_categorical_columns,
    is_numeric_column,
    is_datetime_column,
    is_categorical_column,
)
from utils.formatting import (
    format_bytes,
    format_number,
    format_percentage,
    truncate_string,
)
from utils.files import (
    get_file_extension,
    is_supported_extension,
    generate_timestamped_filename,
)


def test_dataframe_safe_copy() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    copy_df = safe_copy(df)
    copy_df.loc[0, "a"] = 999
    assert df.loc[0, "a"] == 1


def test_make_arrow_safe_preview() -> None:
    # Mixed type object column: string, datetime, int
    mixed_data = ["test", datetime.datetime(2026, 1, 1), 42, None, np.nan]
    df = pd.DataFrame({"mixed_col": mixed_data, "num_col": [1, 2, 3, 4, 5]})

    preview = make_arrow_safe_preview(df)
    assert len(preview) == 5
    # The original DataFrame should be untouched
    assert isinstance(df["mixed_col"].iloc[1], datetime.datetime)
    # The preview strings should be safe
    assert preview["mixed_col"].dtype == object


def test_safe_column_dtype_detectors() -> None:
    df = pd.DataFrame({
        "num_int": [1, 2, 3],
        "num_float": [1.1, 2.2, 3.3],
        "cat_str": ["a", "b", "c"],
        "dt_col": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "cat_col": pd.Categorical(["x", "y", "z"]),
    })

    assert set(safe_numeric_columns(df)) == {"num_int", "num_float"}
    assert set(safe_datetime_columns(df)) == {"dt_col"}
    assert set(safe_categorical_columns(df)) == {"cat_str", "cat_col"}

    assert is_numeric_column(df, "num_int") is True
    assert is_numeric_column(df, "cat_str") is False
    assert is_datetime_column(df, "dt_col") is True
    assert is_categorical_column(df, "cat_col") is True


def test_formatting_utils() -> None:
    assert format_bytes(500) == "500 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(1024 * 1024 * 5) == "5.00 MB"

    assert format_number(1234567) == "1,234,567"
    assert format_percentage(0.8543) == "85.4%"
    assert truncate_string("Hello World", max_length=8) == "Hello..."


def test_file_utils() -> None:
    assert get_file_extension("data.csv") == ".csv"
    assert get_file_extension("SHEET.XLSX") == ".xlsx"
    assert is_supported_extension("file.csv") is True
    assert is_supported_extension("file.parquet") is True
    assert is_supported_extension("file.txt") is False

    ts_name = generate_timestamped_filename("dataset.csv")
    assert ts_name.endswith("_dataset.csv")
