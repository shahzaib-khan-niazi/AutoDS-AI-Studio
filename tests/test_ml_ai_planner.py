"""Unit tests for AIMLPlanner, Explainability, and Insights."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from core.schemas.ml import AIMLPlan, AIExplainabilityReport, AIInsightsReport
from models.ml import AutoMLSummary, ModelEvaluationResult, MLTaskType, FeatureImportance
from services.ml.ai_planner import AIMLPlanner
from services.explainability.service import ExplainabilityService
from services.insights.service import InsightsService
from services.reports.service import ReportGeneratorService


def test_ai_ml_planner_fallback() -> None:
    """AIMLPlanner creates clean fallback strategy when API key is missing."""
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0],
    })

    with patch("services.ml.ai_planner.llm_config") as mock_cfg:
        mock_cfg.is_configured = False
        plan = AIMLPlanner.plan(df, user_target="target")

        assert isinstance(plan, AIMLPlan)
        assert plan.target_column == "target"
        assert len(plan.recommended_features) == 2
        assert len(plan.candidate_models) > 0


def test_explainability_service_fallback() -> None:
    """ExplainabilityService generates structured explanation."""
    summary = AutoMLSummary(
        target_column="price",
        task_type=MLTaskType.REGRESSION,
        rows_trained=100,
        features_used=["rooms", "area"],
        leaderboard=[
            ModelEvaluationResult(
                model_name="Random Forest Regressor",
                task_type=MLTaskType.REGRESSION,
                metrics={"R²": 0.85},
                train_score=0.92,
                test_score=0.85,
                primary_metric_name="R²",
                primary_metric_value=0.85,
                fit_time_seconds=0.1,
            )
        ],
        best_model_name="Random Forest Regressor",
        feature_importances=[
            FeatureImportance(feature="area", importance=0.65),
            FeatureImportance(feature="rooms", importance=0.35),
        ],
    )

    with patch("services.explainability.service.llm_config") as mock_cfg:
        mock_cfg.is_configured = False
        exp = ExplainabilityService.explain(summary)

        assert isinstance(exp, AIExplainabilityReport)
        assert exp.best_model_name == "Random Forest Regressor"
        assert "area" in exp.top_driver_features


def test_insights_and_report_generator() -> None:
    """Insights and Report Generator compile valid artifacts."""
    with patch("services.insights.service.llm_config") as mock_cfg:
        mock_cfg.is_configured = False
        insights = InsightsService.synthesize()

        assert isinstance(insights, AIInsightsReport)
        assert len(insights.headline) > 0

        report_md = ReportGeneratorService.generate_markdown_report(
            dataset_name="sales.csv",
            insights=insights,
        )
        assert "# 📊 AutoDS AI Studio — Executive Data Science Report" in report_md
        assert "sales.csv" in report_md
