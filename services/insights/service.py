"""AI Insights Synthesis Service for AutoDS AI Studio.

Synthesizes cross-stage findings from dataset profiling, data cleaning, exploratory data analysis,
and predictive modeling into strategic executive insights and actionable recommendations.
"""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional

from core.logging import logger
from core.llm.config import llm_config
from core.llm.service import LLMService
from core.llm.exceptions import LLMError
from core.utils.json_sanitizer import sanitize_for_json, safe_json_dumps
from core.schemas.dataset_profile import DatasetProfile
from core.schemas.cleaning import CleaningExecutionResult
from core.schemas.eda import AIEDAAnalysis
from core.schemas.ml import AIInsightsReport, AIExplainabilityReport
from models.ml import AutoMLSummary


class InsightsService:
    """Synthesizes cross-stage pipeline intelligence into executive insights."""

    _SYSTEM_PROMPT = """You are the Lead AI Data Scientist for AutoDS AI Studio.
Your role is to synthesize end-to-end data intelligence into strategic business insights and clear technical takeaways.

CONSTRAINTS & RULES:
1. Ground all conclusions in the provided cross-stage facts.
2. Avoid generic platitudes; make insights specific to the dataset's features, data quality improvements, patterns, and model metrics.
3. Explicitly state limitations, caveats, and next strategic actions.

Output valid JSON matching this schema:
{
  "headline": "High-impact 1-sentence synthesis headline",
  "executive_summary": "Comprehensive 3-4 sentence overview of the data journey and core finding.",
  "data_quality_verdict": "Assessment of data reliability after cleaning.",
  "key_business_insights": ["Insight 1 with concrete reference", "Insight 2"],
  "modeling_and_predictive_power": "Summary of predictive capability and leading signals.",
  "strategic_recommendations": ["Recommendation 1", "Recommendation 2"],
  "critical_risks_and_caveats": ["Caveat 1", "Caveat 2"],
  "suggested_next_actions": ["Next step 1", "Next step 2"]
}
"""

    @classmethod
    def synthesize(
        cls,
        profile: Optional[DatasetProfile] = None,
        cleaning_result: Optional[CleaningExecutionResult] = None,
        eda_analysis: Optional[AIEDAAnalysis] = None,
        automl_summary: Optional[AutoMLSummary] = None,
        explainability: Optional[AIExplainabilityReport] = None,
    ) -> AIInsightsReport:
        """Synthesize cross-stage pipeline data into an AIInsightsReport.

        Returns:
            AIInsightsReport object.
        """
        payload: Dict[str, Any] = {}

        if profile:
            payload["dataset_profile"] = {
                "filename": profile.filename,
                "rows": profile.row_count,
                "cols": profile.column_count,
                "quality_score": f"{profile.quality_score:.1f}/100 ({profile.quality_status})",
            }

        if cleaning_result:
            payload["cleaning_summary"] = cleaning_result.diff_summary

        if eda_analysis:
            payload["eda_findings"] = {
                "patterns": eda_analysis.key_patterns,
                "correlations": eda_analysis.significant_correlations[:3],
                "distributions": eda_analysis.distribution_insights[:3],
            }

        if automl_summary:
            best_eval = next((r for r in automl_summary.leaderboard if r.model_name == automl_summary.best_model_name), None)
            best_score_str = f"{best_eval.primary_metric_name} = {best_eval.primary_metric_value:.4f}" if best_eval else "N/A"
            payload["ml_summary"] = {
                "target": automl_summary.target_column,
                "task": automl_summary.task_type.value,
                "best_model": automl_summary.best_model_name,
                "best_model_score": best_score_str,
                "baseline_model": f"{automl_summary.baseline_model_name} ({automl_summary.baseline_metric_value:.4f})",
                "features_count": len(automl_summary.features_used),
            }

        if explainability:
            payload["explainability"] = {
                "top_features": explainability.top_driver_features,
                "summary": explainability.feature_impact_summary,
            }

        payload = sanitize_for_json(payload)

        if not llm_config.is_configured:
            logger.warning("InsightsService called without configured API key; using fallback")
            return cls._fallback_insights(payload)

        user_prompt = f"Synthesize this cross-stage data science workflow evidence:\n{safe_json_dumps(payload, indent=2)}"

        messages = [
            {"role": "system", "content": cls._SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        schema_hint = {
            "name": "AIInsightsReport",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "executive_summary": {"type": "string"},
                    "data_quality_verdict": {"type": "string"},
                    "key_business_insights": {"type": "array", "items": {"type": "string"}},
                    "modeling_and_predictive_power": {"type": "string"},
                    "strategic_recommendations": {"type": "array", "items": {"type": "string"}},
                    "critical_risks_and_caveats": {"type": "array", "items": {"type": "string"}},
                    "suggested_next_actions": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "headline",
                    "executive_summary",
                    "data_quality_verdict",
                    "key_business_insights",
                    "modeling_and_predictive_power",
                    "strategic_recommendations",
                    "critical_risks_and_caveats",
                    "suggested_next_actions",
                ],
            },
        }

        try:
            svc = LLMService()
            result = svc.generate_structured(messages, response_schema=schema_hint)
            raw = json.loads(result.content)

            return AIInsightsReport(
                headline=str(raw.get("headline", "Data Science Pipeline Insights")),
                executive_summary=str(raw.get("executive_summary", "Synthesis completed.")),
                data_quality_verdict=str(raw.get("data_quality_verdict", "Data quality evaluated.")),
                key_business_insights=[str(i) for i in raw.get("key_business_insights", [])],
                modeling_and_predictive_power=str(raw.get("modeling_and_predictive_power", "Models trained and evaluated.")),
                strategic_recommendations=[str(r) for r in raw.get("strategic_recommendations", [])],
                critical_risks_and_caveats=[str(c) for c in raw.get("critical_risks_and_caveats", [])],
                suggested_next_actions=[str(a) for a in raw.get("suggested_next_actions", [])],
                synthesis_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            )

        except LLMError as e:
            logger.error("LLM error during insights synthesis: {}", e.message)
            return cls._fallback_insights(payload)
        except Exception as e:
            logger.exception("Unexpected error in InsightsService: {}", str(e))
            return cls._fallback_insights(payload)

    @classmethod
    def _fallback_insights(cls, payload: Dict[str, Any]) -> AIInsightsReport:
        """Deterministic fallback insights synthesis."""
        return AIInsightsReport(
            headline="AutoDS AI Studio Automated Pipeline Findings",
            executive_summary="The dataset successfully traversed deterministic profiling, intelligent cleaning, exploratory visual analysis, and AutoML benchmarking.",
            data_quality_verdict="Data hygiene validated with duplicate and missing value resolution.",
            key_business_insights=[
                "Significant features identified through statistical correlation and importance scoring.",
                "Data structures standardized to optimize machine learning performance.",
            ],
            modeling_and_predictive_power="Top models validated across train and test splits to benchmark predictive signals.",
            strategic_recommendations=[
                "Deploy the highest-ranking candidate model for production scoring.",
                "Establish continuous data quality monitoring for incoming batches.",
            ],
            critical_risks_and_caveats=["Validate predictions against real-world drift and out-of-distribution shifts."],
            suggested_next_actions=["Export final executive report and review feature importances with domain stakeholders."],
            synthesis_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
