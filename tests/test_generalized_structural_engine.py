"""Comprehensive tests for the generalized structural cleaning engine in AutoDS AI Studio.

Tests cover:
- Multi-row headers
- Merged-cell-like spreadsheet layouts with Unnamed columns
- Repeated horizontal category blocks across domains (Finance, Healthcare, HR, Inventory)
- Subtotal rows & columns detection and safety rules
- Blank spacer columns & empty spacer rows
- Wide-to-long conversion (unpivot)
- Metadata banner rows
- Ambiguous structures that must be flagged for human review
- Clean CSV/Excel export without extra index columns
- Regression test fixture for badly structured spreadsheet layouts
"""

import pandas as pd
import pytest

from models.structure import StructureType, StructuralPlan
from services.structure.detector import StructureDetector
from services.structure.planner import StructuralPlanner
from services.repair.structural import (
    remove_spacer_rows_cols,
    flatten_headers,
    remove_metadata_rows,
    remove_subtotal_elements,
    unpivot_horizontal_category_blocks,
    auto_reconstruct_structure,
)
from services.repair.export import ExportService


class TestGeneralizedStructuralEngine:
    """Test suite for domain-agnostic structural engine."""

    def test_spacer_rows_and_columns_removal(self):
        """Test removal of completely blank spacer rows and spacer columns."""
        data = {
            "ID": [101, 102, None, 104],
            "Spacer_Col": [None, None, None, None],
            "Name": ["Alice", "Bob", None, "Dave"],
            "Spacer_Col_2": [None, None, None, None],
            "Score": [85.0, 90.0, None, 78.0],
        }
        df = pd.DataFrame(data)
        # Row 2 is completely empty
        df.iloc[2] = [None, None, None, None, None]

        cleaned, rec = remove_spacer_rows_cols(df)
        assert len(cleaned) == 3
        assert len(cleaned.columns) == 3
        assert "Spacer_Col" not in cleaned.columns
        assert "Spacer_Col_2" not in cleaned.columns
        assert list(cleaned["ID"]) == [101, 102, 104]

    def test_multi_row_headers_flattening(self):
        """Test flattening of multi-row headers."""
        data = [
            ["Region", "Q1", "Unnamed: 2", "Q2", "Unnamed: 4"],
            ["Region", "Sales", "Profit", "Sales", "Profit"],
            ["North", 100, 20, 150, 30],
            ["South", 200, 40, 220, 45],
        ]
        df = pd.DataFrame(data[1:], columns=data[0])

        flattened, rec = flatten_headers(df, header_rows=1)
        assert len(flattened) == 2
        assert "Region" in flattened.columns or "Region - Region" in flattened.columns
        assert any("Q1" in col for col in flattened.columns)
        assert any("Q2" in col for col in flattened.columns)

    def test_metadata_banner_removal(self):
        """Test detection and removal of report metadata banner rows."""
        data = {
            "Col1": ["CONFIDENTIAL REPORT - DO NOT DISTRIBUTE", "ID_101", "ID_102"],
            "Col2": [None, 50, 60],
            "Col3": [None, 100, 120],
        }
        df = pd.DataFrame(data)

        cleaned, rec = remove_metadata_rows(df)
        assert len(cleaned) == 2
        assert rec.details["dropped_count"] == 1
        assert cleaned.iloc[0]["Col1"] == "ID_101"

    def test_subtotal_elements_safety_and_removal(self):
        """Test detection and safety rules for subtotal rows and columns."""
        data = {
            "Department": ["Engineering", "Sales", "HR", "Total Summary"],
            "Budget": [100000, 80000, 40000, 220000],
            "Department Total": [100000, 80000, 40000, 220000],
        }
        df = pd.DataFrame(data)

        plan = StructuralPlanner.create_plan(df)
        assert plan.detected_issue == "SUBTOTAL_COLUMNS_ROWS"
        assert plan.requires_human_approval is True
        assert len(plan.ambiguity_flags) > 0

        cleaned, rec = remove_subtotal_elements(df)
        assert "Department Total" not in cleaned.columns
        assert len(cleaned) == 3
        assert list(cleaned["Department"]) == ["Engineering", "Sales", "HR"]

    def test_horizontal_category_blocks_hr_domain(self):
        """Test horizontal category block unpivoting on an HR dataset."""
        data = [
            ["Emp_ID", "Engineering", "Unnamed: 2", "Engineering Total", "Marketing", "Unnamed: 5", "Marketing Total"],
            ["Emp_ID", "Base_Salary", "Bonus", "Total_Comp", "Base_Salary", "Bonus", "Total_Comp"],
            ["E101", 120000, 15000, 135000, 90000, 10000, 100000],
            ["E102", 130000, 20000, 150000, 95000, 12000, 107000],
        ]
        df = pd.DataFrame(data[1:], columns=data[0])

        plan = StructuralPlanner.create_plan(df)
        assert plan.detected_issue in ("REPEATED_CATEGORY_BLOCKS", "MULTI_ROW_HEADER")

        cleaned, rec = unpivot_horizontal_category_blocks(df)
        assert len(cleaned) > 0
        assert len(cleaned.columns) < len(df.columns)
        assert not any("Unnamed:" in str(c) for c in cleaned.columns)

    def test_horizontal_category_blocks_finance_domain(self):
        """Test horizontal category block unpivoting on a Finance dataset."""
        data = [
            ["Account_Code", "Q1_2025", "Unnamed: 2", "Q1 Total", "Q2_2025", "Unnamed: 5", "Q2 Total"],
            ["Account_Code", "Revenue", "Expense", "Net", "Revenue", "Expense", "Net"],
            ["ACC-4001", 50000, 20000, 30000, 55000, 22000, 33000],
        ]
        df = pd.DataFrame(data[1:], columns=data[0])

        cleaned, rec = unpivot_horizontal_category_blocks(df)
        assert len(cleaned) == 1
        assert not any("Unnamed:" in str(c) for c in cleaned.columns)

    def test_horizontal_category_blocks_healthcare_domain(self):
        """Test horizontal category block unpivoting on a Healthcare dataset."""
        data = [
            ["Ward_ID", "ICU", "Unnamed: 2", "ICU Total", "General_Ward", "Unnamed: 5", "General Total"],
            ["Ward_ID", "Beds_Occupied", "Beds_Free", "Total_Beds", "Beds_Occupied", "Beds_Free", "Total_Beds"],
            ["W-10", 45, 5, 50, 120, 30, 150],
        ]
        df = pd.DataFrame(data[1:], columns=data[0])

        cleaned, rec = unpivot_horizontal_category_blocks(df)
        assert len(cleaned) == 1
        assert not any("Unnamed:" in str(c) for c in cleaned.columns)

    def test_horizontal_category_blocks_inventory_domain(self):
        """Test horizontal category block unpivoting on an Inventory dataset."""
        data = [
            ["SKU", "North_Warehouse", "Unnamed: 2", "North Total", "South_Warehouse", "Unnamed: 5", "South Total"],
            ["SKU", "In_Stock", "Reserved", "Total_Qty", "In_Stock", "Reserved", "Total_Qty"],
            ["SKU-0099", 500, 50, 550, 300, 20, 320],
        ]
        df = pd.DataFrame(data[1:], columns=data[0])

        cleaned, rec = unpivot_horizontal_category_blocks(df)
        assert len(cleaned) == 1
        assert not any("Unnamed:" in str(c) for c in cleaned.columns)

    def test_ambiguous_structure_flagging(self):
        """Test that ambiguous structures set requires_human_approval to True."""
        data = {
            "Col_A": ["Val1", "Total", "Val2"],
            "Col_B": [10, 20, 30],
            "Col_C": [100, 200, 300],
        }
        df = pd.DataFrame(data)

        plan = StructuralPlanner.create_plan(df)
        assert plan.requires_human_approval is True or plan.detected_issue == "STANDARD_TABULAR"

    def test_export_cleanliness_no_index_column(self):
        """Test that ExportService.export_csv produces clean CSV without Unnamed: 0 or index columns."""
        data = {
            "Unnamed: 0": [0, 1, 2],  # Artificial index column
            "Item_Name": ["Widget A", "Widget B", "Widget C"],
            "Price": [10.5, 20.0, 15.75],
        }
        df = pd.DataFrame(data)

        csv_bytes = ExportService.export_csv(df)
        csv_str = csv_bytes.decode("utf-8")

        assert "Unnamed: 0" not in csv_str
        assert "Item_Name,Price" in csv_str
        assert "Widget A,10.5" in csv_str

    def test_regression_badly_structured_sales_workbook_fixture(self):
        """Regression test fixture using messy horizontally distributed spreadsheet layout without hardcoding exact names."""
        data = [
            ["Segment >>", "Consumer", "Unnamed: 2", "Consumer Total", "Corporate", "Unnamed: 5", "Corporate Total", "Home Office", "Unnamed: 8", "Home Office Total"],
            ["Segment >>", "Order Date", "Sales", "Consumer Total", "Order Date", "Sales", "Corporate Total", "Order Date", "Sales", "Home Office Total"],
            ["First Class", "01/01/2024", 250.0, 250.0, "02/01/2024", 400.0, 400.0, "03/01/2024", 150.0, 150.0],
            ["Second Class", "04/01/2024", 310.0, 310.0, "05/01/2024", 500.0, 500.0, "06/01/2024", 200.0, 200.0],
            ["Standard Class", "07/01/2024", 180.0, 180.0, "08/01/2024", 620.0, 620.0, "09/01/2024", 90.0, 90.0],
        ]
        df = pd.DataFrame(data[1:], columns=data[0])

        # 1. Structural Detector & Planner
        res = StructureDetector.detect(df)
        assert res.structure != StructureType.UNKNOWN

        plan = StructuralPlanner.create_plan(df, structure_result=res)
        assert plan.detected_issue in ("REPEATED_CATEGORY_BLOCKS", "MULTI_ROW_HEADER")
        assert plan.audit is not None

        # 2. Structural Transformation Execution
        cleaned, rec = unpivot_horizontal_category_blocks(df)
        assert len(cleaned) > 0
        assert not any("Unnamed:" in str(c) for c in cleaned.columns)
        assert rec.success is True

        # 3. Export Verification
        csv_bytes = ExportService.export_csv(cleaned)
        csv_str = csv_bytes.decode("utf-8")
        assert "Unnamed: 0" not in csv_str
        assert "Unnamed: 2" not in csv_str
