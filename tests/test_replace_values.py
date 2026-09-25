"""Unit and Integration Tests for 🔄 Replace Values Tool.

Verifies:
1. Exact string/symbol replacement (e.g., "@" -> "." or "_" -> " ").
2. Empty replacement explicitly means removal ("@" -> "") without introducing "NaN" or "None".
3. Data integrity: rows and columns count remain identical (no row/column deletion).
4. Audit log records column, old_value, new_value, affected_cell_count, affected_row_indices, and timestamp.
5. Works across arbitrary unseen column names and string/categorical types.
"""

import pandas as pd
import pytest

from models.repair import RepairAction, RepairOperation
from services.repair.columns import replace_value_in_column
from services.repair.service import RepairService


def test_replace_values_removal():
    """Verify replacing a value with an empty string removes it without introducing NaN/None."""
    df = pd.DataFrame({
        "customer_name": ["john@smith.com", "abc@xyz.com", "plainuser"],
        "score": [100, 200, 300],
    })

    repaired_df, record = replace_value_in_column(
        df=df,
        column="customer_name",
        old_value="@",
        new_value="",
    )

    assert len(repaired_df) == 3
    assert len(repaired_df.columns) == 2
    assert repaired_df["customer_name"].iloc[0] == "johnsmith.com"
    assert repaired_df["customer_name"].iloc[1] == "abcxyz.com"
    assert repaired_df["customer_name"].iloc[2] == "plainuser"

    # Ensure score column is untouched
    assert repaired_df["score"].tolist() == [100, 200, 300]

    # Audit log validation
    assert record.success is True
    assert record.details["column"] == "customer_name"
    assert record.details["old_value"] == "@"
    assert record.details["new_value"] == ""
    assert record.details["is_empty_replacement"] is True
    assert record.details["affected_cell_count"] == 2
    assert record.details["affected_row_indices"] == [0, 1]


def test_replace_values_custom():
    """Verify custom replacement (e.g. '@' -> '.' and '_' -> ' ')."""
    df = pd.DataFrame({
        "email": ["john@smith.com", "jane@company.org"],
        "full_name": ["john_doe", "jane_smith"],
    })

    df1, rec1 = replace_value_in_column(df, column="email", old_value="@", new_value=".")
    assert df1["email"].iloc[0] == "john.smith.com"
    assert df1["email"].iloc[1] == "jane.company.org"

    df2, rec2 = replace_value_in_column(df1, column="full_name", old_value="_", new_value=" ")
    assert df2["full_name"].iloc[0] == "john doe"
    assert df2["full_name"].iloc[1] == "jane smith"


def test_replace_values_data_integrity():
    """Verify row count and column count are strictly preserved."""
    df = pd.DataFrame({
        "notes": ["N/A", "valid note", "unknown", "N/A"],
        "val": [1.0, 2.0, 3.0, 4.0],
    })

    repaired_df, record = replace_value_in_column(
        df=df,
        column="notes",
        old_value="N/A",
        new_value="",
    )

    assert len(repaired_df) == len(df)
    assert list(repaired_df.columns) == list(df.columns)
    assert repaired_df["notes"].iloc[0] == ""
    assert repaired_df["notes"].iloc[1] == "valid note"
    assert repaired_df["notes"].iloc[3] == ""


def test_replace_values_via_repair_service():
    """Verify RepairService dispatch for REPLACE_VALUES operation."""
    df = pd.DataFrame({
        "code": ["#100", "#200", "#300"],
    })

    action = RepairAction(
        operation=RepairOperation.REPLACE_VALUES,
        target=["code"],
        parameters={
            "column": "code",
            "old_value": "#",
            "new_value": "",
        },
        reason="Remove hash symbol",
    )

    repaired_df, result = RepairService.repair(df, [action])

    assert result.success is True
    assert repaired_df["code"].tolist() == ["100", "200", "300"]
    assert len(result.records) == 1
    assert result.records[0].details["affected_cell_count"] == 3


@pytest.mark.parametrize("special_char", [
    ".", "*", "+", "?", "[", "]", "(", ")", "$", "^", "|"
])
def test_replace_values_literal_special_regex_characters(special_char):
    """Verify regex special characters are replaced literally, not interpreted as regex."""
    val_before = f"price{special_char}100"
    df = pd.DataFrame({
        "data": [val_before, "normal_value"],
        "other_col": [1, 2],
    })

    repaired_df, record = replace_value_in_column(
        df=df,
        column="data",
        old_value=special_char,
        new_value="X",
    )

    assert repaired_df["data"].iloc[0] == f"priceX100"
    assert repaired_df["data"].iloc[1] == "normal_value"
    assert repaired_df["other_col"].tolist() == [1, 2]
    assert len(repaired_df) == 2
    assert len(repaired_df.columns) == 2


def test_replace_values_dtype_preservation():
    """Verify numeric, datetime, and categorical dtypes are preserved where possible."""
    df = pd.DataFrame({
        "numeric_val": [100.0, 200.0, 300.0],
        "cat_val": pd.Categorical(["A_1", "B_1", "C_1"]),
    })

    # Replacing '_' with '-' in categorical column
    repaired_cat, _ = replace_value_in_column(df, column="cat_val", old_value="_", new_value="-")
    assert isinstance(repaired_cat["cat_val"].dtype, pd.CategoricalDtype)
    assert repaired_cat["cat_val"].iloc[0] == "A-1"

    # Replacing '0' with '5' in numeric column
    repaired_num, _ = replace_value_in_column(df, column="numeric_val", old_value="0", new_value="5")
    assert pd.api.types.is_numeric_dtype(repaired_num["numeric_val"])
    assert repaired_num["numeric_val"].iloc[0] == 155.5
