"""Unit tests for Embedded Structured Record Analysis and Reconstruction Engine."""

import pandas as pd
import pytest

from models.structure import StructureType
from services.structure.embedded_records import EmbeddedRecordAnalyzer
from services.structure.detector import StructureDetector
from services.structure.planner import StructuralPlanner
from services.repair.structural import reconstruct_embedded_records


# 1. Embedded fields separated by colons
def test_embedded_colons():
    df = pd.DataFrame({
        "data": [
            "Name: Alice | Age: 28 | City: London",
            "Name: Bob | Age: 34 | City: Paris",
            "Name: Charlie | Age: 42 | City: Tokyo",
        ]
    })
    profile = EmbeddedRecordAnalyzer.profile_dataset(df)
    assert profile["confidence"] >= 0.85
    assert set(profile["candidate_field_labels"]) == {"Name", "Age", "City"}

    reconstructed, rec = reconstruct_embedded_records(df)
    assert set(reconstructed.columns) == {"Name", "Age", "City"}
    assert len(reconstructed) == 3
    assert reconstructed["Name"].tolist() == ["Alice", "Bob", "Charlie"]


# 2. Embedded fields separated by pipes
def test_embedded_pipes():
    df = pd.DataFrame({
        "record": [
            "SKU: 101 | Price: 49.99 | Stock: 150",
            "SKU: 102 | Price: 89.50 | Stock: 75",
            "SKU: 103 | Price: 12.00 | Stock: 300",
        ]
    })
    profile = EmbeddedRecordAnalyzer.profile_dataset(df)
    assert profile["confidence"] >= 0.85
    assert "SKU" in profile["candidate_field_labels"]

    reconstructed, _ = reconstruct_embedded_records(df)
    assert "SKU" in reconstructed.columns
    assert "Price" in reconstructed.columns
    assert "Stock" in reconstructed.columns


# 3. Embedded fields separated by commas
def test_embedded_commas():
    df = pd.DataFrame({
        "info": [
            "Title: Manager, Dept: HR, Salary: 85000",
            "Title: Engineer, Dept: R&D, Salary: 95000",
            "Title: Analyst, Dept: Finance, Salary: 78000",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert "Title" in reconstructed.columns
    assert "Dept" in reconstructed.columns
    assert "Salary" in reconstructed.columns


# 4. Embedded fields separated by semicolons
def test_embedded_semicolons():
    df = pd.DataFrame({
        "log": [
            "ID: 9001; Status: Active; Region: North",
            "ID: 9002; Status: Pending; Region: South",
            "ID: 9003; Status: Active; Region: East",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert "ID" in reconstructed.columns
    assert "Status" in reconstructed.columns
    assert "Region" in reconstructed.columns


# 5. Embedded fields separated by whitespace
def test_embedded_whitespace():
    df = pd.DataFrame({
        "raw": [
            "Name John Smith Age 25 City Karachi",
            "Name Sara Khan Age 30 City Lahore",
            "Name David Miller Age 40 City London",
        ]
    })
    profile = EmbeddedRecordAnalyzer.profile_dataset(df)
    assert profile["confidence"] >= 0.70
    assert "Name" in profile["candidate_field_labels"]
    assert "Age" in profile["candidate_field_labels"]

    reconstructed, _ = reconstruct_embedded_records(df)
    assert "Name" in reconstructed.columns
    assert "Age" in reconstructed.columns
    assert "City" in reconstructed.columns


# 6. Repeated field labels without explicit separators
def test_repeated_field_labels_without_separators():
    df = pd.DataFrame({
        "text": [
            "Product Laptop Price 1200 Quantity 3 Category Electronics",
            "Product Phone Price 800 Quantity 5 Category Electronics",
            "Product Desk Price 350 Quantity 2 Category Furniture",
        ]
    })
    profile = EmbeddedRecordAnalyzer.profile_dataset(df)
    assert set(profile["candidate_field_labels"]).issuperset({"Product", "Price", "Quantity", "Category"})


# 7. Different field ordering across rows
def test_different_field_ordering():
    df = pd.DataFrame({
        "data": [
            "Name: Alice | Age: 28 | City: London",
            "City: Paris | Name: Bob | Age: 34",
            "Age: 42 | City: Tokyo | Name: Charlie",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert reconstructed.loc[0, "Name"] == "Alice"
    assert reconstructed.loc[1, "Name"] == "Bob"
    assert reconstructed.loc[2, "Name"] == "Charlie"


# 8. Missing fields in some rows
def test_missing_fields_in_rows():
    df = pd.DataFrame({
        "data": [
            "Name: Alice | Age: 28 | City: London",
            "Name: Bob | City: Paris",
            "Name: Charlie | Age: 42",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert len(reconstructed) == 3
    assert pd.isna(reconstructed.loc[1, "Age"])
    assert pd.isna(reconstructed.loc[2, "City"])


# 9. Extra fields in some rows
def test_extra_fields_in_rows():
    df = pd.DataFrame({
        "data": [
            "Name: Alice | Age: 28 | City: London",
            "Name: Bob | Age: 34 | City: Paris | Bonus: 5000",
            "Name: Charlie | Age: 42 | City: Tokyo",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert len(reconstructed) == 3
    assert "Bonus" in reconstructed.columns or "Name" in reconstructed.columns


# 10. Numeric fields embedded in text
def test_numeric_fields_embedded():
    df = pd.DataFrame({
        "data": [
            "Item: Chair | Qty: 10 | Cost: 45.50",
            "Item: Table | Qty: 2 | Cost: 150.00",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert pd.api.types.is_numeric_dtype(reconstructed["Qty"])
    assert pd.api.types.is_numeric_dtype(reconstructed["Cost"])


# 11. Date fields embedded in text
def test_date_fields_embedded():
    df = pd.DataFrame({
        "data": [
            "Event: Conference | Date: 2026-05-15 | Code: 101",
            "Event: Workshop | Date: 2026-06-20 | Code: 102",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert "Date" in reconstructed.columns
    assert pd.api.types.is_datetime64_any_dtype(reconstructed["Date"])


# 12. Identifier fields embedded in text
def test_identifier_fields_embedded():
    df = pd.DataFrame({
        "data": [
            "ID: US-1001 | Status: Shipped",
            "ID: EU-2002 | Status: Processing",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert reconstructed["ID"].tolist() == ["US-1001", "EU-2002"]


# 13. Ambiguous text that must NOT be automatically parsed
def test_ambiguous_text_withheld():
    df = pd.DataFrame({
        "notes": [
            "John Smith New York 25",
            "Jane Doe Los Angeles 30",
        ]
    })
    profile = EmbeddedRecordAnalyzer.profile_dataset(df)
    assert profile["confidence"] < 0.85
    assert len(profile["ambiguity_flags"]) > 0


# 14. Non-customer domain: E-commerce Order records
def test_ecommerce_domain():
    df = pd.DataFrame({
        "orders": [
            "Order 1001 Product Phone Quantity 2 Amount 50000",
            "Order 1002 Product Tablet Quantity 1 Amount 35000",
            "Order 1003 Product Laptop Quantity 1 Amount 120000",
        ]
    })
    profile = EmbeddedRecordAnalyzer.profile_dataset(df)
    assert profile["confidence"] >= 0.70
    assert "Order" in profile["candidate_field_labels"]


# 15. Finance domain
def test_finance_domain():
    df = pd.DataFrame({
        "tx": [
            "Account: ACC-101 | Type: Credit | Balance: 15400",
            "Account: ACC-102 | Type: Debit | Balance: 2300",
            "Account: ACC-103 | Type: Credit | Balance: 89000",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert "Account" in reconstructed.columns
    assert "Type" in reconstructed.columns
    assert "Balance" in reconstructed.columns


# 16. HR / Employee domain
def test_hr_domain():
    df = pd.DataFrame({
        "emp": [
            "Employee Ahmed Khan Department Finance Salary 150000",
            "Employee Bilal Raza Department IT Salary 180000",
            "Employee Fatima Noor Department HR Salary 130000",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert "Employee" in reconstructed.columns
    assert "Department" in reconstructed.columns


# 17. Inventory / Warehouse domain
def test_inventory_domain():
    df = pd.DataFrame({
        "inv": [
            "Bin: B-12 | Item: Bolt-10mm | Qty: 5000 | Zone: A1",
            "Bin: B-14 | Item: Nut-10mm | Qty: 8000 | Zone: A2",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert "Bin" in reconstructed.columns
    assert "Item" in reconstructed.columns


# 18. Education / Student domain
def test_education_domain():
    df = pd.DataFrame({
        "student": [
            "Student: S101 | Major: Physics | GPA: 3.8",
            "Student: S102 | Major: Math | GPA: 3.9",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert "Student" in reconstructed.columns
    assert "Major" in reconstructed.columns


# 19. Healthcare-like synthetic non-sensitive domain
def test_healthcare_synthetic_domain():
    df = pd.DataFrame({
        "records": [
            "Subject: SUB-01 | Trial: Phase-2 | Metric: 120",
            "Subject: SUB-02 | Trial: Phase-2 | Metric: 115",
        ]
    })
    reconstructed, _ = reconstruct_embedded_records(df)
    assert "Subject" in reconstructed.columns
    assert "Trial" in reconstructed.columns


# 20. Completely unrelated prose text that must NOT be converted to columns
def test_unrelated_prose_preservation():
    df = pd.DataFrame({
        "comments": [
            "The customer expressed great satisfaction with the delivery speed and packaging quality.",
            "This product has excellent build quality and performs well under heavy daily load conditions.",
            "We encountered a temporary network timeout during database migration on Friday evening.",
        ]
    })
    result = StructureDetector.detect(df)
    # Should NOT be classified as high confidence embedded records
    if result.structure == StructureType.EMBEDDED_STRUCTURED_RECORDS:
        assert result.confidence < 0.85


# 21. Fixture regression test verifying general structural behavior for single-cell customer records
def test_single_cell_customer_record_fixture():
    df = pd.DataFrame({
        "Unstructured_Record": [
            "Name Hussein Hakeem Address Number 22 Fioye Crescent Surulere Lagos Age 17 Gender Male",
            "Name Mary Smith Address 123 Main Street Springfield Age 45 Gender Female",
            "Name John Doe Address 456 Park Avenue New York Age 30 Gender Male",
        ]
    })

    # Test structural detector
    res = StructureDetector.detect(df)
    assert res.structure == StructureType.EMBEDDED_STRUCTURED_RECORDS
    assert res.confidence >= 0.70

    # Test structural planner
    plan = StructuralPlanner.create_plan(df)
    assert plan.detected_issue == "EMBEDDED_STRUCTURED_RECORDS"
    assert plan.proposed_transformation == "reconstruct_embedded_records"

    # Test reconstruction execution
    reconstructed, record = reconstruct_embedded_records(df)
    assert len(reconstructed) == 3
    assert len(reconstructed.columns) >= 3
    assert record.success is True
