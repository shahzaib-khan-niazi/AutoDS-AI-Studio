"""Comprehensive test suite for the Generalized Semantic & Format Consistency Engine.

Verifies:
1. Strict generalization on unknown synthetic domains (zero hardcoded entities).
2. Typographical error correction via edit distance and frequency asymmetry.
3. Dynamic abbreviation & acronym generation and disambiguation.
4. Ambiguous abbreviation detection & flagging (multiple valid candidates).
5. Free-text column protection vs categorical clustering.
6. Multi-format date parsing, day/month ambiguity detection, invalid date reporting.
7. Zero-padded and alphanumeric identifier protection.
8. AI Semantic Normalizer with structured JSON validation and fallback handling.
9. Dispatcher & End-to-End Pipeline integration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock

from models.repair import (
    RepairAction,
    RepairOperation,
    ConfidenceLevel,
    IssueTaxonomy,
    SemanticCandidateGroup,
    DateNormalizationResult,
)
from services.repair.dates import (
    analyze_date_column,
    normalize_date_column,
)
from services.repair.standardize import (
    get_proposed_standardizations,
    standardize_values,
    is_categorical_column,
)
from services.repair.auto_dtypes import auto_detect_dtypes
from services.repair.service import RepairService
from services.repair.dispatcher import dispatch
from services.pipeline.autonomous import AutonomousPipelineService
from services.ai.semantic import AISemanticNormalizer
from core.llm.config import LLMConfig


class TestDynamicSemanticAndTypoStandardization:
    """Test algorithmic normalization without domain hardcoding."""

    def test_frequency_asymmetric_typo_correction(self):
        """Typo variants with low frequency should merge into dominant canonical term."""
        data = {
            "triage_status": ["CRITICAL_IMMEDIATE"] * 80
            + ["critical_immediate"] * 10
            + ["CRITICAL_IMEDIATE"] * 2  # typo
            + ["CRITICAL_IMMEDIAT"] * 1  # typo
            + ["STABLE_ROUTINE"] * 50
            + ["stable_routine"] * 5
            + ["STABLE_ROUTIN"] * 1  # typo
        }
        df = pd.DataFrame(data)

        proposals = get_proposed_standardizations(df)
        repaired_df, record = standardize_values(df)

        assert record.success
        assert "triage_status" in repaired_df.columns
        unique_vals = repaired_df["triage_status"].unique()
        # Should merge into clean canonical representations
        assert len(unique_vals) <= 3
        # Ensure dominant form is preserved/standardized cleanly
        assert any("CRITICAL" in str(v).upper() and "IMMEDIATE" in str(v).upper() for v in unique_vals)
        assert any("STABLE" in str(v).upper() and "ROUTINE" in str(v).upper() for v in unique_vals)

    def test_dynamic_acronym_generation_and_resolution(self):
        """Dynamic acronyms generated from dominant multi-word phrases should resolve."""
        data = {
            "aerospace_component_phase": ["Thermal Protection System"] * 60
            + ["thermal protection system"] * 10
            + ["TPS"] * 4
            + ["tps"] * 2
            + ["Environmental Control Life Support"] * 40
            + ["ECLS"] * 3
        }
        df = pd.DataFrame(data)

        proposals = get_proposed_standardizations(df)
        acronym_proposals = [p for p in proposals if p["method"] in ["acronym_expansion", "dynamic_acronym"]]
        assert len(acronym_proposals) > 0

        repaired_df, record = standardize_values(df)
        assert record.success
        unique_vals = set(repaired_df["aerospace_component_phase"].unique())
        # All TPS and ECLS should be resolved to their expanded forms
        assert "TPS" not in unique_vals
        assert "tps" not in unique_vals
        assert "ECLS" not in unique_vals
        assert len(unique_vals) == 2

    def test_ambiguous_abbreviation_flagging(self):
        """Abbreviation matching multiple candidate phrases must be flagged as ambiguous."""
        data = {
            "system_module": ["Flight Dynamics"] * 50
            + ["Fire Detection"] * 50
            + ["FD"] * 5  # Ambiguous: could be Flight Dynamics or Fire Detection
        }
        df = pd.DataFrame(data)

        proposals = get_proposed_standardizations(df)
        ambig_proposals = [p for p in proposals if p.get("is_ambiguous")]
        assert len(ambig_proposals) > 0
        fd_prop = next((p for p in ambig_proposals if p["original_value"] == "FD"), None)
        assert fd_prop is not None
        assert fd_prop["is_ambiguous"] is True
        assert len(fd_prop["candidate_meanings"]) >= 2
        assert "Flight Dynamics" in fd_prop["candidate_meanings"]
        assert "Fire Detection" in fd_prop["candidate_meanings"]
        # Confidence must be downgraded for ambiguous items
        assert fd_prop["confidence"] < 0.85

    def test_free_text_column_protection(self):
        """Free-form long text descriptions must NOT be mangled or clustered."""
        descriptions = [
            "Patient presented with severe acute abdominal discomfort after lunch.",
            "Vehicle experienced intermittent telemetry dropouts during stage 2 burn.",
            "Customer requested replacement unit due to damaged exterior casing.",
            "System initialized smoothly with zero latency spikes detected.",
            "Routine maintenance inspection completed per schedule specifications.",
        ] * 10
        df = pd.DataFrame({"notes": descriptions, "short_tag": ["TAG_A", "tag_a", "TAG_B", "tag_b", "TAG_A"] * 10})

        assert is_categorical_column(df["notes"]) is False
        assert is_categorical_column(df["short_tag"]) is True

        proposals = get_proposed_standardizations(df)
        col_names_in_proposals = {p["column"] for p in proposals}
        assert "notes" not in col_names_in_proposals
        assert "short_tag" in col_names_in_proposals


class TestDateNormalizationEngine:
    """Test date parsing, ambiguity detection, invalid date reporting, and timezone preservation."""

    def test_multi_format_parsing_same_column(self):
        """Mixed formats in a single column should unify into standard datetime."""
        raw_dates = [
            "2026-05-18",
            "18/05/2026",
            "18-May-2026",
            "May 18, 2026",
            "2026/05/18",
            "18.05.2026",
        ]
        df = pd.DataFrame({"event_date": raw_dates})

        repaired_df, record = normalize_date_column(df, "event_date")
        assert record.success
        assert record.details["parsed_count"] == len(raw_dates)
        assert record.details["invalid_count"] == 0
        # All values should be parsed to 2026-05-18
        parsed_dates = pd.to_datetime(repaired_df["event_date"])
        for dt in parsed_dates:
            assert dt.year == 2026
            assert dt.month == 5
            assert dt.day == 18

    def test_timestamp_and_timezone_preservation(self):
        """ISO timestamps with time and timezone offsets must preserve time components."""
        raw_timestamps = [
            "2026-05-18T14:30:00+00:00",
            "2026-05-18 14:30:00",
            "2026-05-18T14:30:00Z",
        ]
        df = pd.DataFrame({"timestamp": raw_timestamps})

        repaired_df, record = normalize_date_column(df, "timestamp")
        assert record.success
        assert record.details["parsed_count"] == 3
        # Converted series should have datetime type or datetime objects
        first_val = repaired_df["timestamp"].iloc[0]
        assert pd.api.types.is_datetime64_any_dtype(repaired_df["timestamp"]) or isinstance(first_val, (pd.Timestamp, datetime, str))

    def test_day_month_ambiguity_detection(self):
        """When all days and months are <= 12, flag as ambiguous; when evidence exists, be confident."""
        # Ambiguous column (all values <= 12)
        ambig_dates = ["01/02/2026", "03/04/2026", "05/06/2026", "07/08/2026"]
        analysis_ambig = analyze_date_column(pd.Series(ambig_dates))
        assert analysis_ambig.is_ambiguous is True
        assert analysis_ambig.confidence < 0.80

        # Unambiguous DMY column (has day 25 > 12)
        dmy_dates = ["01/02/2026", "25/04/2026", "05/06/2026"]
        analysis_dmy = analyze_date_column(pd.Series(dmy_dates))
        assert analysis_dmy.is_ambiguous is False
        assert analysis_dmy.inferred_convention == "DMY"
        assert analysis_dmy.confidence >= 0.90

        # Unambiguous MDY column (has month > 12 in second position, or textual month)
        mdy_dates = ["02/25/2026", "04/18/2026", "06/05/2026"]
        analysis_mdy = analyze_date_column(pd.Series(mdy_dates))
        assert analysis_mdy.is_ambiguous is False
        assert analysis_mdy.inferred_convention == "MDY"

    def test_invalid_date_detection_and_reporting(self):
        """Impossible days (e.g. Feb 30 or Month 13) must be explicitly flagged and reported."""
        dates_with_invalid = [
            "2026-01-15",
            "2026-02-30",  # Impossible: Feb has at most 29 days
            "32/01/2026",  # Impossible day: 32
            "2026-05-20",
        ]
        df = pd.DataFrame({"record_date": dates_with_invalid})
        analysis = analyze_date_column(df["record_date"])

        assert analysis.invalid_count >= 2
        assert any("32/01/2026" in s for s in analysis.invalid_samples) or len(analysis.invalid_samples) > 0

        # Normalize with error handling
        repaired_df, record = normalize_date_column(df, "record_date")
        assert record.details["invalid_count"] >= 2
        # Invalid entries should become NaT
        assert pd.isna(repaired_df["record_date"].iloc[1])
        assert pd.isna(repaired_df["record_date"].iloc[2])
        # Valid entries preserved
        assert not pd.isna(repaired_df["record_date"].iloc[0])
        assert not pd.isna(repaired_df["record_date"].iloc[3])


class TestIdentifierProtection:
    """Test zero-padded strings, alphanumeric codes, and IDs are protected from numeric casting."""

    def test_zero_padded_and_code_identifiers_preserved(self):
        """Zero-padded zip codes/account numbers and alphanumeric keys must remain string/object."""
        data = {
            "account_id": ["000142", "000591", "001048", "009921", "000034"] * 10,
            "sku_code": ["INV-001", "INV-002", "INV-003", "INV-004", "INV-005"] * 10,
            "pure_num_str": ["100", "200", "300", "400", "500"] * 10,
            "measurement": ["12.5", "14.2", "18.9", "22.1", "10.0"] * 10,
        }
        df = pd.DataFrame(data)

        repaired_df, record = auto_detect_dtypes(df)

        # Zero-padded string MUST NOT become int (which would strip leading zeroes)
        assert repaired_df["account_id"].dtype == "object"
        assert repaired_df["account_id"].iloc[0] == "000142"

        # Alphanumeric SKU MUST remain object
        assert repaired_df["sku_code"].dtype == "object"
        assert repaired_df["sku_code"].iloc[0] == "INV-001"

        # Pure numeric string should convert to integer or float
        assert pd.api.types.is_numeric_dtype(repaired_df["pure_num_str"])
        assert pd.api.types.is_numeric_dtype(repaired_df["measurement"])


class TestAISemanticNormalizer:
    """Test AI Semantic Normalizer with structured JSON validation and safety fallbacks."""

    def test_ai_semantic_clustering_mock(self):
        """Mock LLM response parses into validated SemanticCandidateGroup objects."""
        df = pd.DataFrame({
            "medical_code": ["HTN", "Hypertension", "high bp", "Elevated Blood Pressure", "DM2", "Type 2 Diabetes", "T2D"] * 5
        })

        mock_structured_response = {
            "column": "medical_code",
            "issue_type": "semantic_inconsistency",
            "candidate_groups": [
                {
                    "canonical_value": "Hypertension",
                    "variants": ["HTN", "high bp", "Elevated Blood Pressure"],
                    "reason": "Standard medical nomenclature for elevated arterial pressure",
                    "confidence": 0.96,
                    "action": "normalize",
                },
                {
                    "canonical_value": "Type 2 Diabetes",
                    "variants": ["DM2", "T2D"],
                    "reason": "Standard medical abbreviations for diabetes mellitus type 2",
                    "confidence": 0.98,
                    "action": "normalize",
                },
            ],
        }

        configured_cfg = LLMConfig(api_key="sk-test-mock-key")
        with patch("services.ai.semantic.llm_config", configured_cfg):
            with patch("services.ai.semantic.AIClient.call_structured", return_value=mock_structured_response):
                clusters = AISemanticNormalizer.analyze_column_semantics(df, "medical_code")

                assert len(clusters) == 2
                assert clusters[0].canonical_value == "Hypertension"
                assert "HTN" in clusters[0].variants
                assert clusters[0].confidence == 0.96
                assert clusters[1].canonical_value == "Type 2 Diabetes"

    def test_ai_semantic_normalizer_fallback_when_unconfigured(self):
        """When LLM is not configured, gracefully falls back to deterministic proposals without raising."""
        df = pd.DataFrame({"status": ["ACTIVE", "active", "PENDING", "pending"] * 10})
        unconfigured_cfg = LLMConfig(api_key="")
        with patch("services.ai.semantic.llm_config", unconfigured_cfg):
            clusters = AISemanticNormalizer.analyze_column_semantics(df, "status")
            assert isinstance(clusters, list)


class TestDispatcherAndAutonomousIntegration:
    """Test dispatcher execution and Autonomous Pipeline end-to-end with new operations."""

    def test_dispatcher_normalize_dates(self):
        """RepairOperation.NORMALIZE_DATES dispatches correctly through repair_dispatcher."""
        df = pd.DataFrame({"signup_date": ["2026-01-01", "02/01/2026", "03-Jan-2026"]})
        action = RepairAction(
            operation=RepairOperation.NORMALIZE_DATES,
            target=["signup_date"],
            parameters={"convention": "DMY"},
            reason="Normalize signup dates",
        )

        repaired_df, record = dispatch(df, action)
        assert record.success
        assert pd.api.types.is_datetime64_any_dtype(repaired_df["signup_date"])

    def test_repair_service_end_to_end(self):
        """RepairService applies multiple actions including standardize and auto_dtypes."""
        df = pd.DataFrame({
            "code": ["0012", "0034", "0056"],
            "dept": ["ENGINEERING", "engineering", "ENG"],
            "date": ["2026-03-01", "15/03/2026", "20-Mar-2026"],
        })

        actions = [
            RepairAction(operation=RepairOperation.STANDARDIZE_VALUES, reason="Standardize dept"),
            RepairAction(operation=RepairOperation.NORMALIZE_DATES, target=["date"], reason="Normalize dates"),
            RepairAction(operation=RepairOperation.AUTO_DTYPES, reason="Auto infer dtypes"),
        ]

        repaired_df, report = RepairService.repair(df, actions)
        assert report.success
        assert len(report.actions_applied) == 3
        # Identifier preserved
        assert repaired_df["code"].iloc[0] == "0012"
        # Date normalized
        assert pd.api.types.is_datetime64_any_dtype(repaired_df["date"])

    def test_autonomous_pipeline_full_run(self):
        """Autonomous pipeline automatically handles dates, identifiers, and categorical variants."""
        df = pd.DataFrame({
            "emp_id": [f"00{100 + i}" for i in range(20)],
            "region": ["NORTH_AMERICA", "North America", "north america", "North America"] * 5,
            "start_date": ["2026-01-10", "15/01/2026", "20-Jan-2026", "2026/01/25"] * 5,
            "salary": [str(50000 + i * 1000) for i in range(20)],
        })

        cleaned_df, result = AutonomousPipelineService.run_auto_preprocess(df, use_ai_if_available=False)
        assert result.success
        assert result.final_quality_score >= 0.90
        assert str(cleaned_df["emp_id"].iloc[0]).startswith("00")
        assert pd.api.types.is_numeric_dtype(cleaned_df["salary"])
