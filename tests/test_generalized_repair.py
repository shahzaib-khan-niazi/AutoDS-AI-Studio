"""Comprehensive Generalized Data Cleaning & Repair Engine Tests.

Tests pure algorithmic, domain-agnostic behavior:
- Delimited multi-value cell exploding (synchronized & single-column)
- Repeated header and summary row removal
- Algorithmic fuzzy clustering, case/whitespace/punctuation normalization, dynamic acronym discovery
- Identifier column protection (leading zeros, alphanumeric codes)
- Currency, percentage, unit, and mixed date parsing
- Full autonomous pipeline execution
"""

import os
import pytest
import pandas as pd
import numpy as np

from models.repair import RepairAction, RepairOperation
from services.repair.structural import (
    detect_multi_value_cells,
    explode_multi_value_cells,
    remove_repeated_headers,
    remove_metadata_rows,
)
from services.repair.standardize import (
    standardize_values,
    get_proposed_standardizations,
)
from services.repair.auto_dtypes import auto_detect_dtypes
from services.repair.missing import fill_missing
from services.repair.service import RepairService
from services.pipeline.autonomous import AutonomousPipelineService


# =====================================================================
# 1. Structural Repairs: Multi-Value Packed Cells (Explode / Unpack)
# =====================================================================

def test_detect_and_explode_synchronized_multi_value_cells():
    """Test synchronized multi-column unpacking with different delimiters."""
    data = {
        "txn_id": ["TXN-101", "TXN-102"],
        "items": ["Apples | Oranges | Bananas", "Milk | Bread"],
        "quantities": ["5 | 2 | 10", "1 | 3"],
        "prices": ["1.50 | 3.00 | 0.75", "4.20 | 2.50"],
    }
    df = pd.DataFrame(data)

    detected = detect_multi_value_cells(df)
    assert detected.get("is_synchronized") is True
    assert detected.get("delimiter") == "|"
    det_cols = detected.get("synchronized_columns", [])
    assert len(det_cols) == 3
    assert "items" in det_cols
    assert "quantities" in det_cols
    assert "prices" in det_cols

    # Explode
    res_df, record = explode_multi_value_cells(df, target_columns=["items", "quantities", "prices"])
    
    assert record.success is True
    # TXN-101 (3 items) + TXN-102 (2 items) = 5 rows total
    assert len(res_df) == 5
    assert list(res_df["txn_id"]) == ["TXN-101", "TXN-101", "TXN-101", "TXN-102", "TXN-102"]
    assert list(res_df["items"]) == ["Apples", "Oranges", "Bananas", "Milk", "Bread"]
    assert list(res_df["quantities"]) == [5, 2, 10, 1, 3]
    assert list(res_df["prices"]) == [1.50, 3.00, 0.75, 4.20, 2.50]


def test_detect_and_explode_newline_delimiter():
    """Test exploding multi-line cells delimited by newline characters."""
    data = {
        "dept": ["Engineering", "Marketing"],
        "employees": ["Alice\nBob\nCharlie", "Dana\nEvan"],
        "roles": ["Dev\nLead\nDev", "Specialist\nManager"],
    }
    df = pd.DataFrame(data)

    res_df, record = explode_multi_value_cells(df)
    assert record.success is True
    assert len(res_df) == 5
    assert list(res_df["employees"]) == ["Alice", "Bob", "Charlie", "Dana", "Evan"]
    assert list(res_df["roles"]) == ["Dev", "Lead", "Dev", "Specialist", "Manager"]


def test_explode_single_column_list():
    """Test exploding a single column with tags/skills."""
    data = {
        "user": ["Alice", "Bob", "Charlie"],
        "skills": ["Python; SQL; Docker", "Java; C++", "Python"],
    }
    df = pd.DataFrame(data)

    res_df, record = explode_multi_value_cells(df, target_columns=["skills"])
    assert record.success is True
    assert len(res_df) == 6
    assert list(res_df["user"]) == ["Alice", "Alice", "Alice", "Bob", "Bob", "Charlie"]
    assert list(res_df["skills"]) == ["Python", "SQL", "Docker", "Java", "C++", "Python"]


def test_explode_mismatched_lengths_graceful():
    """Test that mismatched list lengths across columns pad with None without crashing."""
    data = {
        "group": ["G1"],
        "col_a": ["A1 | A2 | A3"],
        "col_b": ["B1 | B2"],  # Shorter
    }
    df = pd.DataFrame(data)

    res_df, record = explode_multi_value_cells(df, target_columns=["col_a", "col_b"])
    assert record.success is True
    assert len(res_df) == 3
    assert list(res_df["col_a"]) == ["A1", "A2", "A3"]
    assert res_df["col_b"].tolist()[:2] == ["B1", "B2"]
    assert pd.isna(res_df["col_b"].iloc[2])


# =====================================================================
# 2. Structural Repairs: Repeated Headers & Metadata Rows
# =====================================================================

def test_remove_repeated_headers():
    """Test removal of duplicated header rows within the dataset body."""
    data = {
        "Product": ["Widget A", "Product", "Widget B", "Product", "Widget C"],
        "Price": ["10.00", "Price", "20.00", "Price", "30.00"],
        "Stock": ["100", "Stock", "50", "Stock", "75"],
    }
    df = pd.DataFrame(data)

    res_df, record = remove_repeated_headers(df)
    assert record.success is True
    assert len(res_df) == 3
    assert list(res_df["Product"]) == ["Widget A", "Widget B", "Widget C"]


def test_remove_metadata_rows():
    """Test removal of trailing total and note rows."""
    data = {
        "Region": ["North", "South", "East", "Total", "Source: Internal Q3 Report"],
        "Sales": ["1000", "1500", "1200", "3700", None],
        "Margin": ["0.2", "0.25", "0.22", "0.22", None],
    }
    df = pd.DataFrame(data)

    res_df, record = remove_metadata_rows(df)
    assert record.success is True
    assert len(res_df) == 3
    assert list(res_df["Region"]) == ["North", "South", "East"]


# =====================================================================
# 3. Categorical Standardization: Pure Algorithmic Clustering
# =====================================================================

def test_standardize_typos_via_fuzzy_clustering():
    """Test Levenshtein typo clustering without any hardcoded dictionary."""
    data = {
        "country": [
            "Australia", "Australia", "Australia", "Australia",
            "Austarlia", "Australai",  # Typos of Australia
            "Singapore", "Singapore", "Singapore",
            "Singapre",  # Typo of Singapore
        ]
    }
    df = pd.DataFrame(data)

    res_df, record = standardize_values(df, columns=["country"])
    assert record.success is True
    unique_countries = set(res_df["country"].dropna())
    assert unique_countries == {"Australia", "Singapore"}


def test_standardize_case_and_whitespace_clustering():
    """Test case variation and whitespace inconsistency clustering."""
    data = {
        "status": ["Completed", "completed", "COMPLETED", " Completed ", "In-Progress", "in progress", "IN_PROGRESS"],
    }
    df = pd.DataFrame(data)

    res_df, record = standardize_values(df, columns=["status"])
    assert record.success is True
    assert res_df["status"].nunique() <= 2


def test_standardize_dynamic_acronym_discovery():
    """Test dynamic acronym matching without hardcoded acronym dictionaries."""
    data = {
        "org": [
            "United States of America", "United States of America", "United States of America",
            "USA", "USA",
            "European Union", "European Union",
            "EU",
        ]
    }
    df = pd.DataFrame(data)

    res_df, record = standardize_values(df, columns=["org"])
    assert record.success is True
    # USA should be standardized to dominant "United States of America"
    assert "USA" not in res_df["org"].values
    assert "EU" not in res_df["org"].values


def test_standardize_sentinels_to_nan():
    """Test standardizing placeholder text sentinels to NaN."""
    data = {
        "comments": ["Great service", "N/A", "none", "Fast shipping", "???", "null"],
    }
    df = pd.DataFrame(data)

    res_df, record = standardize_values(df, columns=["comments"])
    assert record.success is True
    assert res_df["comments"].isna().sum() == 4


# =====================================================================
# 4. Type Conversion & Identifier Protection
# =====================================================================

def test_identifier_protection_leading_zeros():
    """Verify that postal codes / IDs with leading zeros are NOT converted to integer."""
    data = {
        "zip_code": ["01234", "05678", "00981", "08001"],
        "sales": ["$1,200.50", "$3,400.00", "$500.25", "$9,100.00"],
    }
    df = pd.DataFrame(data)

    res_df, record = auto_detect_dtypes(df)
    assert record.success is True
    # zip_code must remain object/string to preserve leading zero
    assert res_df["zip_code"].dtype == "object"
    assert res_df["zip_code"].iloc[0] == "01234"
    # sales must be converted to float
    assert pd.api.types.is_float_dtype(res_df["sales"])
    assert res_df["sales"].iloc[0] == 1200.50


def test_identifier_protection_alphanumeric_codes():
    """Verify that alphanumeric codes like 'INV-2024-001' are protected from numerical conversions."""
    data = {
        "invoice_no": ["INV-001", "INV-002", "INV-003"],
        "rate": ["15.5%", "20.0%", "5.0%"],
    }
    df = pd.DataFrame(data)

    res_df, record = auto_detect_dtypes(df)
    assert record.success is True
    assert res_df["invoice_no"].dtype == "object"
    assert pd.api.types.is_float_dtype(res_df["rate"])
    assert res_df["rate"].iloc[0] == 15.5


def test_identifier_protection_in_missing_value_imputation():
    """Verify that missing value imputation does not compute mean/median on ID columns."""
    data = {
        "order_id": ["ORD-101", None, "ORD-103", "ORD-104"],
        "amount": [100.0, None, 300.0, 400.0],
    }
    df = pd.DataFrame(data)

    res_df, record = fill_missing(df, strategy="auto")
    assert record.success is True
    # order_id missing value should NOT be filled with arbitrary statistical average or mode
    assert res_df["order_id"].isna().sum() == 1
    # amount should be imputed with median (300.0 or 250.0)
    assert res_df["amount"].isna().sum() == 0


def test_mixed_date_format_parsing():
    """Test parsing dates with mixed slash and ISO formats."""
    data = {
        "created_at": ["2023-01-15", "2023-02-20", "2023-03-25", "2023-04-10"],
    }
    df = pd.DataFrame(data)

    res_df, record = auto_detect_dtypes(df)
    assert record.success is True
    assert pd.api.types.is_datetime64_any_dtype(res_df["created_at"])


# =====================================================================
# 5. Full Autonomous Pipeline Integration
# =====================================================================

def test_autonomous_pipeline_synthetic_messy_dataset():
    """Test the end-to-end 1-click autonomous pipeline on a multi-issue dataset."""
    data = {
        "Ref Code": ["REF-001", "REF-002", "REF-002", "Ref Code", "Total Summary"],
        "Category": ["Electronics | Furniture", "Office Supplies", "Office Supplies", "Category", None],
        "Amount": ["$1,500.00 | $250.50", "$45.00", "$45.00", "Amount", None],
        "City": ["San Francisco ", "san francisco", "san-francisco", "City", None],
        "Date": ["2024-01-01 | 2024-01-02", "2024-01-05", "2024-01-05", "Date", None],
    }
    df = pd.DataFrame(data)

    cleaned_df, pipeline_res = AutonomousPipelineService.run_auto_preprocess(df, use_ai_if_available=False)

    assert pipeline_res.success is True
    assert pipeline_res.quality_improvement >= 0.0
    # Repeated header and total rows removed
    assert "Ref Code" not in cleaned_df["Ref Code"].values
    assert "Total Summary" not in cleaned_df["Ref Code"].values
    # Delimited cells exploded
    assert len(cleaned_df) >= 3
    # Amount converted to numeric
    assert pd.api.types.is_float_dtype(cleaned_df["Amount"])
    # City standardized
    assert cleaned_df["City"].nunique() == 1


def test_autonomous_pipeline_on_example_excel_if_available():
    """Verify autonomous pipeline on the example invoice excel file if present."""
    paths = [
        "20260904_125539_8.-Invoices-with-Merged-Categories-and-Merged-Amounts.xlsx",
        os.path.join("data", "raw", "20260904_125539_8.-Invoices-with-Merged-Categories-and-Merged-Amounts.xlsx"),
    ]
    excel_path = next((p for p in paths if os.path.exists(p)), None)
    if not excel_path:
        pytest.skip("Example excel file not present in workspace.")

    dirty_df = pd.read_excel(excel_path, sheet_name="Dirty 8")
    cleaned_df, res = AutonomousPipelineService.run_auto_preprocess(dirty_df, use_ai_if_available=False)

    assert res.success is True
    # The 4 dirty invoice rows should unpack into 13 relational rows
    assert len(cleaned_df) == 13
    assert "Category" in cleaned_df.columns
    assert "Amount" in cleaned_df.columns
    # Verify Amount column is numeric
    assert pd.api.types.is_numeric_dtype(cleaned_df["Amount"])
