"""Unit tests for Enhanced Dataset Repair capabilities:

- Whitespace stripping
- Categorical value standardization & abbreviation expansion (Payment Methods, Cities, Status)
- Automatic dtype detection & conversion (mixed date formats, numeric currencies, booleans)
"""

import pandas as pd
import pytest

from models.repair import RepairAction, RepairOperation
from services.repair.whitespace import strip_whitespace
from services.repair.standardize import standardize_values, detect_inconsistent_columns, get_proposed_standardizations
from services.repair.auto_dtypes import auto_detect_dtypes, detect_mistyped_columns
from services.repair.service import RepairService
from services.repair.dispatcher import dispatch


def test_strip_whitespace():
    df = pd.DataFrame({
        "city": [" islamabad ", "Karachi  ", " Lahore", "Quetta\t", "Rawalpindi"],
        "num": [1, 2, 3, 4, 5],
    })

    cleaned_df, record = strip_whitespace(df)

    assert cleaned_df["city"].tolist() == ["islamabad", "Karachi", "Lahore", "Quetta", "Rawalpindi"]
    assert record.details["total_cells_modified"] == 4
    assert record.success is True


def test_standardize_payment_method_inconsistencies():
    df = pd.DataFrame({
        "payment_method": [
            "Credit Card",
            "CC",
            "cash",
            "Cash",
            "Bank Transfer",
            "CreditCard",
            "BANK TRANSFER",
            "COD",
            "CREDIT CARD",
            "BT",
        ]
    })

    # Standardize values
    df_std, record = standardize_values(df)
    pm_list = df_std["payment_method"].tolist()

    # Credit Card variants
    assert pm_list[0] == "Credit Card"
    assert pm_list[1] == "Credit Card"
    assert pm_list[5] == "Credit Card"
    assert pm_list[8] == "Credit Card"

    # Cash variants
    assert pm_list[2] == "Cash"
    assert pm_list[3] == "Cash"

    # Bank Transfer variants
    assert pm_list[4] == "Bank Transfer"
    assert pm_list[6] == "Bank Transfer"
    assert pm_list[9] == "Bank Transfer"

    # COD
    assert pm_list[7] == "COD"


def test_standardize_values_city_abbreviations():
    df = pd.DataFrame({
        "city": [
            "islamabad",
            "Quetta",
            "Karachi ",
            "Lahore  ",
            "Islam-Abad",
            "Quetta ",
            "faisalabad",
            "Karachi ",
            "lahore",
            "Karachi",
            "FAISALABAD",
        ]
    })

    # First strip whitespace
    df_ws, _ = strip_whitespace(df)
    # Then standardize
    df_std, record = standardize_values(df_ws)

    cities = df_std["city"].tolist()

    assert cities[0] == "Islam-Abad"
    assert cities[1] == "Quetta"
    assert cities[2] == "Karachi"
    assert cities[3] == "Lahore"
    assert cities[4] == "Islam-Abad"
    assert cities[5] == "Quetta"
    assert cities[6] == "FAISALABAD" or cities[6] == "faisalabad"
    assert cities[7] == "Karachi"
    assert cities[8] == "Lahore"
    assert cities[9] == "Karachi"
    assert cities[10] == "FAISALABAD" or cities[10] == "faisalabad"

    assert record.success is True


def test_detect_inconsistent_columns():
    df = pd.DataFrame({
        "city": ["islamabad", "ISB", "Islam Abad", "Karachi"],
        "gender": ["Male", "Male", "Female", "Female"],
    })

    inconsistent = detect_inconsistent_columns(df)
    assert "city" in inconsistent
    assert "gender" not in inconsistent


def test_auto_detect_dtypes_numeric_and_datetime():
    df = pd.DataFrame({
        "price_str": ["$100.50", "Rs. 200", "300.75", "PKR 400", "500"],
        "date_str": ["2026-01-15", "15/01/2026", "15-Jan-2026", "Jan 15, 2026", "2026/01/15"],
        "is_active_str": ["True", "False", "Yes", "No", "True"],
        "text_col": ["Hello", "World", "Foo", "Bar", "Baz"],
    })

    converted_df, record = auto_detect_dtypes(df)

    assert pd.api.types.is_numeric_dtype(converted_df["price_str"])
    assert pd.api.types.is_datetime64_any_dtype(converted_df["date_str"])
    assert str(converted_df["is_active_str"].dtype) in ("boolean", "bool")
    assert converted_df["text_col"].dtype == "object"
    assert record.details["total_converted"] == 3


def test_detect_mistyped_columns():
    df = pd.DataFrame({
        "num_text": ["10", "20", "30", "40", "50"],
        "regular_text": ["apple", "banana", "cherry", "date", "elderberry"],
    })

    mistyped = detect_mistyped_columns(df)
    assert len(mistyped) == 1
    assert mistyped[0]["column"] == "num_text"
    assert mistyped[0]["suggested_type"] == "numeric"


def test_dispatcher_new_operations():
    df = pd.DataFrame({
        "city": [" Karachi ", " karachi ", " KARACHI "]
    })

    act_ws = RepairAction(operation=RepairOperation.STRIP_WHITESPACE)
    res_df, rec_ws = dispatch(df, act_ws)
    assert res_df["city"].tolist() == ["Karachi", "karachi", "KARACHI"]

    act_std = RepairAction(operation=RepairOperation.STANDARDIZE_VALUES)
    res_df2, rec_std = dispatch(res_df, act_std)
    assert res_df2["city"].tolist() == ["Karachi", "Karachi", "Karachi"]


def test_auto_detect_repairs_includes_new_features():
    df = pd.DataFrame({
        "city": [" ISB ", "Islamabad ", " islamabad"],
        "price": ["$10", "$20", "$30"],
    })

    suggestions = RepairService.auto_detect_repairs(df)
    ops = [s.operation for s in suggestions]

    assert RepairOperation.STRIP_WHITESPACE in ops
    assert RepairOperation.STANDARDIZE_VALUES in ops
    assert RepairOperation.AUTO_DTYPES in ops
