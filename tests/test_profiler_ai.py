"""Unit tests for AIProfilerInterpreter (services/profiler/ai.py).

Verifies structured LLM prompt construction, JSON interpretation parsing,
evidence verification, and fallback behavior. Uses mocks (no real API calls).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.llm.config import LLMConfig
from core.schemas.dataset_profile import DatasetProfile, ColumnProfile, DatasetQualityIssue
from services.profiler.ai import AIProfilerInterpreter
from services.profiler.service import DatasetProfiler


@pytest.fixture
def sample_profile() -> DatasetProfile:
    """Fixture providing a sample DatasetProfile."""
    return DatasetProfile(
        filename="test_data.csv",
        row_count=100,
        column_count=2,
        duplicate_row_count=5,
        duplicate_row_percentage=5.0,
        memory_usage_mb=0.2,
        quality_score=85.0,
        quality_status="Good",
        columns=[
            ColumnProfile(
                name="age",
                dtype="Integer",
                pandas_dtype="int64",
                non_null_count=90,
                missing_count=10,
                missing_percentage=10.0,
                unique_count=45,
                unique_percentage=45.0,
                is_numeric=True,
                min_val=18,
                max_val=75,
                mean_val=42.5,
            ),
            ColumnProfile(
                name="city",
                dtype="String/Text",
                pandas_dtype="object",
                non_null_count=100,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=4,
                unique_percentage=4.0,
                is_categorical=True,
                top_values={"NY": 50, "LA": 30, "SF": 20},
            ),
        ],
        detected_issues=[
            DatasetQualityIssue(
                issue_type="missing_values",
                column="age",
                severity="medium",
                description="Column 'age' has 10 missing values (10.0%).",
                metric_value="10.0%",
            )
        ],
    )


class TestAIProfilerInterpreter:
    """Unit tests for AIProfilerInterpreter class."""

    @patch("services.profiler.ai.LLMService")
    @patch("services.profiler.ai.llm_config")
    def test_successful_ai_analysis(
        self, mock_config: MagicMock, mock_llm_cls: MagicMock, sample_profile: DatasetProfile
    ) -> None:
        """Interpreter should call LLMService and parse structured response."""
        mock_config.is_configured = True

        mock_svc = MagicMock()
        mock_llm_cls.return_value = mock_svc

        fake_llm_json = json.dumps({
            "dataset_summary": "Customer demographic dataset containing age and city features.",
            "likely_domain": "Customer Analytics",
            "important_columns": ["age", "city"],
            "quality_assessment": "Generally high quality with minor missing demographic values.",
            "detected_issues": [
                {
                    "issue_type": "missing_values",
                    "column": "age",
                    "severity": "medium",
                    "explanation": "Age has 10 missing entries.",
                    "evidence": "Column 'age' has 10 missing values (10.0%).",
                    "recommendation": "Impute missing ages using median.",
                }
            ],
            "priority_issues": ["Impute missing values in age column"],
            "recommended_next_steps": ["Apply median imputation to age"],
            "warnings": [],
        })

        mock_result = MagicMock()
        mock_result.content = fake_llm_json
        mock_svc.generate_structured.return_value = mock_result

        analysis = AIProfilerInterpreter.analyze(sample_profile)

        assert analysis.likely_domain == "Customer Analytics"
        assert len(analysis.important_columns) == 2
        assert len(analysis.detected_issues) == 1
        assert analysis.detected_issues[0].column == "age"
        assert analysis.confidence_score >= 0.80
        assert analysis.confidence_level == "High"

    @patch("services.profiler.ai.llm_config")
    def test_unconfigured_api_key_returns_fallback(
        self, mock_config: MagicMock, sample_profile: DatasetProfile
    ) -> None:
        """When OPENROUTER_API_KEY is not set, return clean deterministic fallback analysis."""
        mock_config.is_configured = False

        analysis = AIProfilerInterpreter.analyze(sample_profile)

        assert analysis.likely_domain == "General Structured Data"
        assert len(analysis.warnings) == 1
        assert "not configured" in analysis.warnings[0]
        assert len(analysis.detected_issues) == 1

    def test_compact_profile_format(self, sample_profile: DatasetProfile) -> None:
        """Compact profile dictionary should be clean and token-efficient."""
        compact = AIProfilerInterpreter._prepare_compact_profile(sample_profile)

        assert compact["filename"] == "test_data.csv"
        assert compact["row_count"] == 100
        assert len(compact["columns"]) == 2
        assert compact["columns"][0]["name"] == "age"
        assert compact["columns"][0]["missing_pct"] == "10.0%"
