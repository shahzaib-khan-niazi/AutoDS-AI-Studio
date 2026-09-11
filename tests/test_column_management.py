"""Tests for Column Name Management suite.

Verifies:
- Manual single column rename & validation
- Column standardization (snake_case, lowercase, whitespace, symbols, underscores)
- Duplicate column detection & resolution
- Unnamed column classification (accidental index, empty column, legitimate data)
- AI column name suggestions & heuristic fallback
- Data preservation, type safety, and audit record logging
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from services.repair.columns import (
    validate_column_rename,
    rename_single_column,
    standardize_name,
    get_column_name_standardizations,
    standardize_column_names,
    detect_duplicate_columns,
    resolve_duplicate_columns,
    detect_unnamed_columns,
    remove_column,
    rename_columns,
)
from services.ai.columns import AIColumnSuggester
from core.exceptions import RepairValidationError


def test_validate_column_rename() -> None:
    df = pd.DataFrame({"Customer Name": [1, 2], "age": [30, 40]})

    # Valid rename
    is_valid, err = validate_column_rename(df, "Customer Name", "customer_name")
    assert is_valid is True
    assert err is None

    # Invalid: old name not in df
    is_valid, err = validate_column_rename(df, "Nonexistent", "new_name")
    assert is_valid is False
    assert err is not None
    assert "does not exist" in err

    # Invalid: empty new name
    is_valid, err = validate_column_rename(df, "Customer Name", "   ")
    assert is_valid is False
    assert err is not None
    assert "cannot be empty" in err

    # Invalid: new name already exists
    is_valid, err = validate_column_rename(df, "Customer Name", "age")
    assert is_valid is False
    assert err is not None
    assert "already exists" in err

    # Valid: rename to itself
    is_valid, err = validate_column_rename(df, "Customer Name", "Customer Name")
    assert is_valid is True


def test_rename_single_column_preserves_data_and_types() -> None:
    original_df = pd.DataFrame({
        "Customer Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "Score": [98.5, 88.0, 92.3],
    })

    repaired_df, record = rename_single_column(
        df=original_df,
        old_name="Customer Name",
        new_name="customer_name",
        method="manual",
        confidence=1.0,
    )

    # 1. Column name updated
    assert "customer_name" in repaired_df.columns
    assert "Customer Name" not in repaired_df.columns

    # 2. Row count and column count unchanged
    assert len(repaired_df) == len(original_df)
    assert len(repaired_df.columns) == len(original_df.columns)

    # 3. Values and dtypes completely preserved
    assert list(repaired_df["customer_name"]) == ["Alice", "Bob", "Charlie"]
    assert repaired_df["customer_name"].dtype == original_df["Customer Name"].dtype
    assert list(repaired_df["Age"]) == [25, 30, 35]

    # 4. Original dataframe was not mutated
    assert "Customer Name" in original_df.columns

    # 5. Audit record verified
    assert record.operation == "column_rename"
    assert record.success is True
    assert record.details["repair_type"] == "column_rename"
    assert record.details["old_name"] == "Customer Name"
    assert record.details["new_name"] == "customer_name"
    assert record.details["method"] == "manual"
    assert record.details["confidence"] == 1.0
    assert record.details["status"] == "applied"


def test_rename_single_column_invalid_raises_error() -> None:
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    with pytest.raises(RepairValidationError):
        rename_single_column(df, "col_a", "col_b")


def test_standardize_name_rules() -> None:
    strategies = ["snake_case"]

    # Test specified requirements
    assert standardize_name("Customer Name", strategies) == "customer_name"
    assert standardize_name(" Customer Age ", strategies) == "customer_age"
    assert standardize_name("Order-ID", strategies) == "order_id"
    assert standardize_name("Sales Amount ($)", strategies) == "sales_amount"
    assert standardize_name("Product   Name", strategies) == "product_name"
    assert standardize_name("CustomerName", strategies) == "customer_name"
    assert standardize_name("cust_nm", strategies) == "cust_nm"


def test_standardize_column_names_dataframe() -> None:
    df = pd.DataFrame(
        [[1, 2, 3, 4, 5]],
        columns=[
            "Customer Name",
            " Customer Age ",
            "Order-ID",
            "Sales Amount ($)",
            "Product   Name",
        ],
    )

    proposals = get_column_name_standardizations(df, ["snake_case"])
    assert len(proposals) == 5
    assert all(p["changed"] for p in proposals)

    repaired_df, record = standardize_column_names(df, ["snake_case"])
    expected_cols = [
        "customer_name",
        "customer_age",
        "order_id",
        "sales_amount",
        "product_name",
    ]
    assert list(repaired_df.columns) == expected_cols
    assert record.success is True
    assert record.details["repair_type"] == "column_rename"
    assert record.details["method"] == "standardize"
    assert len(record.details["changes"]) == 5

    # Original untouched
    assert "Customer Name" in df.columns


def test_duplicate_column_detection_and_resolution() -> None:
    # Construct DataFrame with duplicate column names
    df = pd.DataFrame(
        [[1, 2, 3, 4]],
        columns=["Customer", "Customer", "Sales", "Sales"],
    )

    detection = detect_duplicate_columns(df)
    assert detection["has_duplicates"] is True
    assert detection["duplicates"] == {"Customer": 2, "Sales": 2}

    repaired_df, record = resolve_duplicate_columns(df)
    assert list(repaired_df.columns) == ["Customer", "Customer_2", "Sales", "Sales_2"]
    assert record.success is True
    assert record.details["repair_type"] == "column_rename"
    assert record.details["method"] == "duplicate_resolution"
    assert record.details["total_resolved"] == 2

    # Original untouched
    assert list(df.columns) == ["Customer", "Customer", "Sales", "Sales"]


def test_unnamed_column_detection() -> None:
    df = pd.DataFrame({
        "Unnamed: 0": [0, 1, 2, 3],  # Exported dataframe index
        "": [np.nan, np.nan, np.nan, np.nan],  # Completely empty column
        "Unnamed: 1": ["Alice", "Bob", "Charlie", "David"],  # Legitimate data
        "NormalCol": [10, 20, 30, 40],
    })

    detected = detect_unnamed_columns(df)
    assert len(detected) == 3

    # Check index column classification
    idx_col = next(d for d in detected if d["column_name"] == "Unnamed: 0")
    assert idx_col["classification"] == "accidental_index"
    assert idx_col["recommendation"] == "Remove"
    assert idx_col["confidence"] == 0.98

    # Check empty column classification
    empty_col = next(d for d in detected if d["column_name"] == "")
    assert empty_col["classification"] == "empty_column"
    assert empty_col["recommendation"] == "Remove"
    assert empty_col["confidence"] == 0.95

    # Check legitimate data classification
    data_col = next(d for d in detected if d["column_name"] == "Unnamed: 1")
    assert data_col["classification"] == "legitimate_data"
    assert "Keep" in data_col["recommendation"]

    # Test removing accidental index column
    repaired, rec = remove_column(df, "Unnamed: 0", reason=idx_col["issue"])
    assert "Unnamed: 0" not in repaired.columns
    assert len(repaired.columns) == 3
    assert len(repaired) == 4
    assert rec.details["repair_type"] == "column_drop"


def test_ai_column_suggester_heuristic_fallback() -> None:
    df = pd.DataFrame({
        "cust_nm": ["Acme Corp", "Beta LLC"],
        "amt": [100.5, 200.0],
        "Order-ID": [101, 102],
        "already_clean": ["A", "B"],
    })

    from core.llm.config import LLMConfig
    from unittest.mock import PropertyMock

    # When AI is offline, heuristic suggestions kick in
    with patch.object(LLMConfig, "is_configured", new_callable=PropertyMock, return_value=False):
        suggestions = AIColumnSuggester.suggest_column_names(df)

    sugg_map = {s["current_name"]: s for s in suggestions}

    assert sugg_map["cust_nm"]["suggested_name"] == "customer_name"
    assert sugg_map["cust_nm"]["changed"] is True
    assert sugg_map["amt"]["suggested_name"] == "amount"
    assert sugg_map["Order-ID"]["suggested_name"] == "order_id"
    assert sugg_map["already_clean"]["suggested_name"] == "already_clean"
    assert sugg_map["already_clean"]["changed"] is False


def test_ai_column_suggester_mocked_llm() -> None:
    df = pd.DataFrame({
        "c_nm": ["John Doe", "Jane Smith"],
        "ord_dt": ["2026-01-01", "2026-01-02"],
    })

    mock_llm_response = {
        "suggestions": [
            {
                "current_name": "c_nm",
                "suggested_name": "customer_name",
                "reason": "Column contains customer full names.",
                "confidence": 0.96,
            },
            {
                "current_name": "ord_dt",
                "suggested_name": "order_date",
                "reason": "Column contains ISO order timestamps.",
                "confidence": 0.98,
            },
        ]
    }

    from core.llm.config import LLMConfig
    from unittest.mock import PropertyMock

    with patch.object(LLMConfig, "is_configured", new_callable=PropertyMock, return_value=True):
        with patch("services.ai.client.AIClient.call_structured", return_value=mock_llm_response):
            suggestions = AIColumnSuggester.suggest_column_names(df)

            assert len(suggestions) == 2
            assert suggestions[0]["current_name"] == "c_nm"
            assert suggestions[0]["suggested_name"] == "customer_name"
            assert suggestions[0]["confidence"] == 0.96
            assert suggestions[0]["reason"] == "Column contains customer full names."

            # Verify safe rename execution following architecture:
            # DATA PROFILE -> AI SUGGESTION -> VALIDATION -> USER APPROVAL -> PYTHON EXECUTION -> VERIFICATION
            repaired_df, rec = rename_single_column(
                df,
                old_name=suggestions[0]["current_name"],
                new_name=suggestions[0]["suggested_name"],
                method="ai_suggestion",
                confidence=suggestions[0]["confidence"],
            )
            rec.details["status"] = "approved"

            assert "customer_name" in repaired_df.columns
            assert "c_nm" not in repaired_df.columns
            assert rec.details["repair_type"] == "column_rename"
            assert rec.details["method"] == "ai_suggestion"
            assert rec.details["confidence"] == 0.96
            assert rec.details["status"] == "approved"


def test_deterministic_rename_columns_backward_compatibility() -> None:
    df = pd.DataFrame(
        [[1, 2, 3, 4]],
        columns=["Unnamed: 0", "Revenue", "Revenue", "  TrimMe  "],
    )
    repaired, record = rename_columns(df)
    assert list(repaired.columns) == ["unnamed_0", "Revenue", "Revenue_1", "TrimMe"]
    assert record.success is True
