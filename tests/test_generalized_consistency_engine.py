"""Comprehensive test suite for Generalized Categorical & Date Consistency Engine.

Strict invariants:
1. Zero domain hardcoding (no static city, country, product, payment, or acronym mappings).
2. Generalized 8-tier categorical normalization (whitespace, unicode, casing, typos, prefix abbreviations, consonant skeleton abbreviations, out-of-vocabulary review flagging).
3. Row-aware date parsing: unambiguous vs ambiguous vs invalid.
4. Mixed / conflicting date conventions detection (e.g. MM-DD-YYYY mixed with DD-MM-YYYY in same column).
5. Timestamp & timezone preservation.
6. Ambiguity protection: Ambiguous values and dates are strictly flagged for review rather than guessed.
"""

import pytest
import pandas as pd
import numpy as np

from services.repair.standardize import (
    get_proposed_standardizations,
    standardize_values,
)
from services.repair.dates import (
    analyze_date_column,
    classify_and_parse_single_date,
    normalize_date_column,
    RowDateStatus,
)
from services.repair.auto_dtypes import auto_detect_dtypes, infer_semantic_datatype
from services.repair.service import RepairService
from models.repair import RepairAction, RepairOperation


# =============================================================================
# 1. Regression Test on User's Inconsistent Sales / Demonstration Dataset
# =============================================================================

def test_user_observed_inconsistent_dataset_behavior():
    """Verify that the engine dynamically discovers inconsistencies in the demonstration dataset
    without hardcoding any mappings.
    """
    df = pd.DataFrame({
        "city": [
            "islamabad", "Quetta", "ISB", "Karachi", "Lahore", "Lhr", "islamabad", "UET", "Quetta"
        ],
        "product_category": [
            "APPAREL", "Clothing", "APPAREL", "Furniture", "ELECTRONICS", "GROCERIES", "App", "APPAREL", "Furn"
        ],
        "order_date": [
            "2025-04-04",
            "12/10/2025",
            "10-15-2025",
            "Aug 05, 2025",
            "2024/12/28 20:45:24",
            "21-Aug-2025",
            "06.07.2025",
            "28/03/25",
            "January 28, 2026",
        ],
    })

    # 1. Categorical proposals
    props = get_proposed_standardizations(df)
    props_by_orig = {(p["column"], p["original_value"]): p for p in props}

    # Product category: 'App' -> prefix of 'APPAREL', 'Furn' -> prefix of 'Furniture'
    assert ("product_category", "App") in props_by_orig
    assert props_by_orig[("product_category", "App")]["normalized_value"] == "APPAREL"
    assert props_by_orig[("product_category", "App")]["method"] == "prefix_abbreviation"

    assert ("product_category", "Furn") in props_by_orig
    assert props_by_orig[("product_category", "Furn")]["normalized_value"] == "Furniture"
    assert props_by_orig[("product_category", "Furn")]["method"] == "prefix_abbreviation"

    # City: 'Lhr' -> consonant skeleton of 'Lahore', 'ISB' -> consonant/subsequence of 'Islamabad'
    assert ("city", "Lhr") in props_by_orig
    assert props_by_orig[("city", "Lhr")]["normalized_value"] == "Lahore"
    assert props_by_orig[("city", "Lhr")]["method"] == "consonant_skeleton_abbreviation"

    assert ("city", "ISB") in props_by_orig
    assert props_by_orig[("city", "ISB")]["normalized_value"] == "islamabad"
    assert props_by_orig[("city", "ISB")]["method"] == "consonant_skeleton_abbreviation"

    # UET: has no canonical expansion in dataset -> flagged as unresolved abbreviation requiring review
    assert ("city", "UET") in props_by_orig
    assert props_by_orig[("city", "UET")]["is_ambiguous"] is True
    assert props_by_orig[("city", "UET")]["is_safe"] is False

    # 2. Date column analysis
    date_diag = analyze_date_column(df["order_date"])

    # Column contains 10-15-2025 (MDY proof) AND 28/03/25 (DMY proof) -> MIXED CONVENTIONS!
    assert date_diag.mixed_conventions_detected is True
    assert date_diag.is_ambiguous is True
    assert date_diag.inferred_convention == "MIXED"
    assert date_diag.has_time is True

    # Check row-level diagnosis:
    # 2025-04-04 -> unambiguous ISO
    st_iso, ts_iso, d_iso = classify_and_parse_single_date("2025-04-04")
    assert st_iso == RowDateStatus.UNAMBIGUOUS
    assert ts_iso is not None
    assert ts_iso.year == 2025 and ts_iso.month == 4 and ts_iso.day == 4

    # 10-15-2025 -> unambiguous MDY (15 > 12)
    st_mdy, ts_mdy, d_mdy = classify_and_parse_single_date("10-15-2025")
    assert st_mdy == RowDateStatus.UNAMBIGUOUS
    assert ts_mdy is not None
    assert ts_mdy.month == 10 and ts_mdy.day == 15 and ts_mdy.year == 2025

    # 28/03/25 -> unambiguous DMY (28 > 12)
    st_dmy, ts_dmy, d_dmy = classify_and_parse_single_date("28/03/25")
    assert st_dmy == RowDateStatus.UNAMBIGUOUS
    assert ts_dmy is not None
    assert ts_dmy.day == 28 and ts_dmy.month == 3

    # 2024/12/28 20:45:24 -> unambiguous with time preserved
    st_time, ts_time, d_time = classify_and_parse_single_date("2024/12/28 20:45:24")
    assert st_time == RowDateStatus.UNAMBIGUOUS
    assert ts_time is not None
    assert ts_time.hour == 20 and ts_time.minute == 45 and ts_time.second == 24

    # 12/10/2025 -> ambiguous (both 12 and 10 <= 12)
    st_amb, ts_amb, d_amb = classify_and_parse_single_date("12/10/2025")
    assert st_amb == RowDateStatus.AMBIGUOUS
    assert d_amb["is_ambiguous"] is True


# =============================================================================
# 2. Generalized Domain-Agnostic Synthetic Tests
# =============================================================================

def test_unknown_domain_prefix_and_consonant_abbreviations():
    """Verify abbreviation discovery on a completely synthetic unknown domain (e.g. biology taxonomy)."""
    df = pd.DataFrame({
        "specimen_type": [
            "Amphibian", "Amphibian", "Amph",  # Prefix abbreviation
            "Mammalia", "Mammalia", "Mml",   # Consonant skeleton
            "Reptilia", "Reptilia", "Rept",   # Prefix abbreviation
            "UNK_CODE",                      # Unresolved code
        ]
    })

    props = get_proposed_standardizations(df)
    mapping = {p["original_value"]: p["normalized_value"] for p in props if not p.get("is_ambiguous")}

    assert mapping.get("Amph") == "Amphibian"
    assert mapping.get("Rept") == "Reptilia"
    assert mapping.get("Mml") == "Mammalia"

    unresolved = [p for p in props if p["original_value"] == "UNK_CODE"]
    assert len(unresolved) == 1
    assert unresolved[0]["is_ambiguous"] is True
    assert unresolved[0]["is_safe"] is False


def test_entity_protection_legal_qualifiers():
    """Verify distinct business entities are NEVER merged despite substring similarity."""
    df = pd.DataFrame({
        "organization": [
            "Novartis", "Novartis Ltd", "Novartis Store", "Novartis Corporation", "Novartis HQ",
            "Novartis", "Novartis",
        ]
    })

    props = get_proposed_standardizations(df)
    # None of the qualified entities ('Novartis Ltd', 'Novartis Store') should be proposed to merge into 'Novartis'
    merges_into_novartis = [
        p["original_value"] for p in props
        if p["normalized_value"] == "Novartis" and p["original_value"] != "Novartis"
    ]
    assert len(merges_into_novartis) == 0


def test_date_engine_preserves_timestamps():
    """Verify date normalization preserves hours, minutes, seconds and sub-seconds."""
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
    assert res_df["event_time"].iloc[1].hour == 14
    assert res_df["event_time"].iloc[1].minute == 45
    assert res_df["event_time"].iloc[1].second == 22


def test_unambiguous_dmy_column_resolution():
    """Verify that a column with unambiguous DMY proof safely resolves dayfirst."""
    df = pd.DataFrame({
        "date_col": [
            "25/01/2026",  # 25 > 12 -> DMY proof
            "31/05/2026",  # 31 > 12 -> DMY proof
            "14/08/2026",  # 14 > 12 -> DMY proof
            "02/03/2026",  # <= 12 -> resolved with column-wide proof
        ]
    })

    diag = analyze_date_column(df["date_col"])
    assert diag.inferred_convention == "DMY"
    assert diag.dayfirst is True
    assert diag.is_ambiguous is False
    assert diag.mixed_conventions_detected is False

    res_df, record = normalize_date_column(df, "date_col")
    assert record.status == "applied"
    # 02/03/2026 should be 2nd March 2026 (day=2, month=3)
    assert res_df["date_col"].iloc[3].day == 2
    assert res_df["date_col"].iloc[3].month == 3
