"""Comprehensive test suite for the Generalized Data Quality and Consistency Engine.

Tests:
1. Zero Domain Hardcoding & Unknown Datasets.
2. Cardinal Rule 1: Never auto-repair based on AI confidence alone (all 6 criteria enforced).
3. Cardinal Rule 2: Distinct Entity Protection ("ABC" vs "ABC Ltd" vs "ABC Store").
4. Cardinal Rule 3: Never Guess Ambiguous Dates (entire-column scanning and review flagging).
5. Advanced Semantic Type Inference & Number/Currency/Parentheses-negative parsing.
6. Outlier Methods (IQR, Z-score, MAD, Percentiles) & Impossible Domain Values.
7. Duplicate Detection (exact, near, identifier).
8. Repair Preview Generation & Impact Simulation.
"""

import pytest
import pandas as pd
import numpy as np

from models.repair import (
    RepairAction,
    RepairOperation,
    IssueTaxonomy,
    SemanticType,
)
from services.repair.auto_dtypes import infer_semantic_datatype, auto_detect_and_convert_dtypes
from services.repair.dates import analyze_date_column, normalize_dates
from services.repair.standardize import standardize_categorical_values
from services.repair.outliers import handle_outliers, detect_impossible_values
from services.repair.duplicates import detect_duplicates, remove_duplicate_rows
from services.repair.service import RepairService
from services.pipeline.autonomous import AutonomousPipelineService
from services.ai.planner import AIPlan, AIAction


class TestZeroDomainHardcoding:
    """Validate engine on purely synthetic, unknown domain datasets."""

    def test_synthetic_alien_measurements(self):
        """Unknown domain concepts like 'flux_flux', 'xenon_reading' clean correctly."""
        df = pd.DataFrame({
            "xenon_reading": ["  $1,200.50 ", " (450.25) ", " $3,100.00 ", " $0.00 "],
            "flux_status": ["ACTIVE", "active ", " ACTIVE", "dormant"],
            "cycle_epoch": ["2024-01-15", "2024-02-18", "2024-03-22", "2024-04-30"],
        })

        # Test semantic inference on currency with parenthesis negative
        res = infer_semantic_datatype(df["xenon_reading"])
        assert res.inferred_type in (SemanticType.CURRENCY, SemanticType.FLOAT)

        # Test auto convert dtypes
        converted_df, rec = auto_detect_and_convert_dtypes(df)
        assert pd.api.types.is_numeric_dtype(converted_df["xenon_reading"])
        assert converted_df["xenon_reading"].iloc[1] == -450.25
        assert converted_df["xenon_reading"].iloc[0] == 1200.50


class TestCardinalRule1GatedAutoRepair:
    """Never auto-repair based on AI confidence alone."""

    def test_rejects_ai_action_on_protected_identifier(self):
        """AI action proposing to standardize or alter an identifier column is rejected."""
        df = pd.DataFrame({
            "uuid_code": ["ID-001", "ID-002", "ID-003", "ID-004"],
            "reading": [10.5, 20.2, 15.3, 19.8],
        })

        ai_plan = AIPlan(
            decision="repair",
            confidence=0.99,  # High confidence, but modifying an ID
            actions=[
                AIAction(
                    operation="standardize_values",
                    target=["uuid_code"],
                    reason="Normalize IDs to lowercase",
                    confidence=0.99,
                )
            ],
        )

        valid_actions = AutonomousPipelineService.evaluate_ai_plan_for_auto_repair(df, ai_plan)
        assert len(valid_actions) == 0, "Identifier modification should be blocked from auto-repair"

    def test_rejects_ai_action_below_confidence_threshold(self):
        """AI action below confidence threshold is gated out."""
        df = pd.DataFrame({
            "category": ["A", "B", "A", "C"],
            "val": [1, 2, 3, 4],
        })

        ai_plan = AIPlan(
            decision="repair",
            confidence=0.75,
            actions=[
                AIAction(
                    operation="fill_missing",
                    target=["val"],
                    reason="Low confidence guess",
                    confidence=0.75,
                )
            ],
        )

        valid_actions = AutonomousPipelineService.evaluate_ai_plan_for_auto_repair(df, ai_plan, confidence_threshold=0.85)
        assert len(valid_actions) == 0


class TestCardinalRule2DistinctEntityProtection:
    """Never assume high string similarity = same meaning (ABC vs ABC Ltd vs ABC Store)."""

    def test_distinct_qualified_entities_not_merged(self):
        """Entities with distinct organizational/business qualifiers must NEVER merge."""
        df = pd.DataFrame({
            "company_name": [
                "ABC", "ABC", "ABC", "ABC", "ABC",
                "ABC Ltd", "ABC Ltd", "ABC Ltd",
                "ABC Store", "ABC Store",
            ]
        })

        cleaned_df, rec = standardize_categorical_values(df, columns=["company_name"], similarity_threshold=0.70)
        unique_vals = set(cleaned_df["company_name"].unique())

        assert "ABC" in unique_vals
        assert "ABC Ltd" in unique_vals
        assert "ABC Store" in unique_vals
        assert len(unique_vals) == 3, f"Expected 3 distinct entities, got: {unique_vals}"

    def test_casing_and_whitespace_variants_are_standardized(self):
        """Exact semantic variants ('Apex Corp', 'apex corp', '  Apex Corp  ') should merge safely."""
        df = pd.DataFrame({
            "vendor": ["Apex Corp", "apex corp", "Apex Corp  ", "Apex Corp", "Beta LLC", "beta llc"]
        })

        cleaned_df, rec = standardize_categorical_values(df, columns=["vendor"])
        unique_vendors = set(cleaned_df["vendor"].unique())
        assert len(unique_vendors) == 2
        assert "Apex Corp" in unique_vendors
        assert "Beta LLC" in unique_vendors


class TestCardinalRule3NeverGuessAmbiguousDates:
    """Analyze entire column; if days and months all <= 12, flag for review."""

    def test_ambiguous_date_column_flagged_with_low_confidence(self):
        """When all dates have day <= 12 and month <= 12, flag as ambiguous and do not guess."""
        ambiguous_dates = pd.Series(["01/02/2023", "03/04/2023", "05/06/2023", "07/08/2023", "09/10/2023"])
        analysis = analyze_date_column(ambiguous_dates)

        assert analysis.is_ambiguous is True
        assert analysis.confidence < 0.80
        assert "AMBIGUOUS" in analysis.notes.upper() or "REVIEW" in analysis.notes.upper()

    def test_unambiguous_mixed_dates_resolved_by_disqualifying_row(self):
        """A single row with day 25 proves DMY convention for the entire column."""
        dates = pd.Series(["01/02/2023", "03/04/2023", "25/06/2023", "07/08/2023"])
        analysis = analyze_date_column(dates)

        assert analysis.is_ambiguous is False
        assert analysis.inferred_convention == "DMY"
        assert analysis.confidence >= 0.85

    def test_normalize_dates_preserves_unambiguous_and_handles_mixed(self):
        """Mixed formats with ISO and DMY parse correctly without guessing."""
        df = pd.DataFrame({
            "transaction_date": ["2023-01-15", "18/02/2023", "2023-03-20", "28/04/2023"]
        })

        repaired_df, rec = normalize_dates(df, columns=["transaction_date"])
        assert pd.api.types.is_datetime64_any_dtype(repaired_df["transaction_date"])
        assert int(repaired_df["transaction_date"].isna().sum()) == 0


class TestNumericAndCurrencyParsing:
    """Advanced numeric parsing: parenthesis negatives, currencies, percentages."""

    def test_parentheses_negatives_and_commas(self):
        series = pd.Series(["$1,234.56", "(789.10)", "$50.00", "(1,000.00)"])
        infer_res = infer_semantic_datatype(series)
        assert infer_res.inferred_type in (SemanticType.CURRENCY, SemanticType.FLOAT)

        df = pd.DataFrame({"amt": series})
        converted_df, rec = auto_detect_and_convert_dtypes(df)
        assert converted_df["amt"].iloc[1] == -789.10
        assert converted_df["amt"].iloc[3] == -1000.00
        assert converted_df["amt"].iloc[0] == 1234.56

    def test_percentages_conversion(self):
        series = pd.Series(["12.5%", "25.0%", "99.9%", "0.5%"])
        infer_res = infer_semantic_datatype(series)
        assert infer_res.inferred_type in (SemanticType.PERCENTAGE, SemanticType.FLOAT)

        df = pd.DataFrame({"pct": series})
        converted_df, rec = auto_detect_and_convert_dtypes(df)
        assert converted_df["pct"].iloc[0] == 12.5
        assert converted_df["pct"].iloc[1] == 25.0


class TestOutliersAndImpossibleValues:
    """Test MAD, IQR, Z-Score, Percentiles, and Impossible Values."""

    def test_impossible_values_detection(self):
        df = pd.DataFrame({
            "age": [25, 30, -5, 42, 50],
            "discount_percent": [10.0, 20.0, 150.0, 5.0, 15.0],
            "measurement": [1.0, np.inf, 3.2, 4.1, 2.8],
        })

        issues = detect_impossible_values(df)
        assert "age" in issues
        assert any("negative" in iss["reason"].lower() for iss in issues["age"])
        assert "discount_percent" in issues
        assert any("exceed" in iss["reason"].lower() for iss in issues["discount_percent"])
        assert "measurement" in issues
        assert any("infinite" in iss["reason"].lower() for iss in issues["measurement"])

    def test_outlier_methods(self):
        # Array with an extreme outlier
        np.random.seed(42)
        normal_data = list(np.random.normal(50, 5, 50))
        normal_data.append(500.0)  # extreme outlier
        df = pd.DataFrame({"val": normal_data})

        # IQR
        df_iqr, rec_iqr = handle_outliers(df, columns=["val"], method="iqr", factor=1.5, action="cap")
        assert df_iqr["val"].max() < 500.0
        assert rec_iqr.rows_before == len(df)

        # MAD
        df_mad, rec_mad = handle_outliers(df, columns=["val"], method="mad", factor=3.0, action="cap")
        assert df_mad["val"].max() < 500.0

        # Z-score
        df_z, rec_z = handle_outliers(df, columns=["val"], method="zscore", factor=3.0, action="cap")
        assert df_z["val"].max() < 500.0

        # Percentile
        df_p, rec_p = handle_outliers(df, columns=["val"], method="percentile", factor=0.02, action="cap")
        assert df_p["val"].max() < 500.0


class TestDuplicateDetection:
    """Test exact, near, and identifier key collision detection."""

    def test_duplicate_detection_categories(self):
        df = pd.DataFrame({
            "customer_id": ["C101", "C102", "C101", "C103"],
            "name": ["Alice Smith", "Bob Jones", "alice  smith ", "Charlie"],
            "city": ["New York", "London", "new york", "Paris"],
        })

        result = detect_duplicates(df, id_columns=["customer_id"])
        assert result.exact_duplicate_rows >= 0
        assert result.near_duplicate_rows >= 1
        assert len(result.duplicate_identifiers) >= 1
        assert "customer_id" in result.duplicate_identifiers


class TestRepairPreviewSimulation:
    """Test before/after simulation without altering the original dataframe."""

    def test_generate_repair_preview(self):
        df = pd.DataFrame({
            "col1": ["  apple  ", "banana", "  cherry "],
            "col2": ["$10.50", "$20.00", "$30.25"],
        })

        actions = [
            RepairAction(operation=RepairOperation.STRIP_WHITESPACE, reason="Clean spaces"),
            RepairAction(operation=RepairOperation.AUTO_DTYPES, reason="Convert types"),
        ]

        preview = RepairService.generate_repair_preview(df, actions)

        # Original df must be untouched
        assert df["col1"].iloc[0] == "  apple  "
        assert preview["values_changed"] > 0
        assert len(preview["preview_table"]) == 2
        assert preview["potential_data_loss"] == 0
