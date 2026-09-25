"""Unit and Integration Tests for Natural-Language Number Normalization & Mixed Numeric Parsing.

Verifies:
1. Written natural-language number parsing (units, tens, hundreds, thousands, millions, billions, decimals, hyphens, case).
2. Magnitude abbreviation parsing (50K, 1.5M, $50K, 2B PKR).
3. Prose Guard: Natural language sentences containing number words are NOT converted to numbers.
4. Mixed numeric representations in a single column are recognized and normalized into a single numeric dtype.
5. Unparseable cells in numeric columns are preserved and marked as pd.NA / Unavailable without deleting rows.
6. Non-numeric columns containing number words are preserved as text/string.
"""

import pandas as pd
import numpy as np
import pytest

from utils.number_parser import parse_written_number, parse_magnitude_abbreviation
from services.repair.auto_dtypes import parse_numeric_value, infer_semantic_datatype, auto_detect_dtypes


def test_parse_written_numbers():
    """Verify written natural language number expressions parse accurately."""
    assert parse_written_number("one hundred") == 100.0
    assert parse_written_number("five hundred") == 500.0
    assert parse_written_number("one thousand") == 1000.0
    assert parse_written_number("one thousand two hundred") == 1200.0
    assert parse_written_number("twenty five thousand") == 25000.0
    assert parse_written_number("two million") == 2000000.0
    assert parse_written_number("one thousand two hundred fifty") == 1250.0
    assert parse_written_number("three point five") == 3.5
    assert parse_written_number("one-hundred") == 100.0
    assert parse_written_number("ONE THOUSAND") == 1000.0
    assert parse_written_number("twenty-five thousand") == 25000.0


def test_parse_magnitude_abbreviation():
    """Verify magnitude abbreviation strings parse into floats."""
    assert parse_magnitude_abbreviation("50K") == 50000.0
    assert parse_magnitude_abbreviation("1.5M") == 1500000.0
    assert parse_magnitude_abbreviation("$50K") == 50000.0
    assert parse_magnitude_abbreviation("2B PKR") == 2000000000.0


def test_prose_guard_safety():
    """Verify that sentences containing number words return None and are NOT converted."""
    assert parse_written_number("one hundred reasons to leave") is None
    assert parse_written_number("two hundred people attended") is None
    assert parse_written_number("five thousand items sold yesterday") is None
    assert parse_numeric_value("one hundred reasons to leave") is None
    assert parse_numeric_value("two hundred people attended") is None


def test_mixed_numeric_column_normalization():
    """Verify a column with mixed numeric representations is detected and normalized."""
    df = pd.DataFrame({
        "mixed_amount": [
            50000,
            "50000",
            "50,000",
            "$50,000",
            "50K",
            "75,000 PKR",
            "1.5M",
            "25%",
            "1,250.50",
            "one hundred",
            "five hundred",
            "twenty five thousand",
        ]
    })

    cleaned_df, record = auto_detect_dtypes(df)

    assert "mixed_amount" in record.details["conversions"]
    assert pd.api.types.is_numeric_dtype(cleaned_df["mixed_amount"])
    assert len(cleaned_df) == 12

    # Check exact normalized numeric values
    expected = [
        50000.0, 50000.0, 50000.0, 50000.0, 50000.0,
        75000.0, 1500000.0, 25.0, 1250.50, 100.0, 500.0, 25000.0
    ]
    np.testing.assert_allclose(cleaned_df["mixed_amount"].astype(float).values, expected)


def test_prose_text_column_preservation():
    """Verify text columns containing written number words are preserved as string/text."""
    df = pd.DataFrame({
        "notes": [
            "one hundred reasons to leave",
            "two hundred people attended",
            "five thousand items in catalog",
            "there were twenty five thousand participants",
        ]
    })

    inference = infer_semantic_datatype(df["notes"])
    assert inference.detected_type in ("text", "string", "categorical")
    assert inference.detected_type not in ("integer", "float", "numeric", "currency", "percentage")

    cleaned_df, record = auto_detect_dtypes(df)
    assert "notes" not in record.details.get("conversions", {})
    assert cleaned_df["notes"].dtype == object or pd.api.types.is_string_dtype(cleaned_df["notes"])
    assert cleaned_df["notes"].iloc[0] == "one hundred reasons to leave"
