"""End-to-end Integration Audit Tests for the ML Workflow in AutoDS AI Studio.

Verifies complete integration from target selection through fitted pipeline,
prediction, explainability, cross-stage AI insights, executive report generation,
error boundaries, and session-state dataset invalidation.
"""

import pytest
import pandas as pd
import numpy as np

from models.ml import MLTaskType, AutoMLSummary
from services.ml.planner import MLPlanner
from services.ml.trainer import AutoMLTrainer
from services.explainability.service import ExplainabilityService
from services.insights.service import InsightsService
from services.reports.service import ReportGeneratorService
from core.state import set_dataset, initialize_state


def test_end_to_end_ml_integration_flow():
    """Verify complete production ML flow:
    Dataset -> Target Selection -> Preprocessing -> Fitting -> Best Pipeline
    -> Prediction -> Feature Importance -> Explainability -> Insights -> Final Report.
    """
    np.random.seed(42)
    n = 120

    ids = [f"ID_{i:04d}" for i in range(n)]
    ages = np.random.randint(18, 70, size=n).astype(float)
    dates = pd.date_range("2021-01-01", periods=n, freq="D").astype(str)
    plans = np.random.choice(["Basic", "Pro", "Enterprise"], size=n)
    spends = np.random.uniform(20.0, 500.0, size=n)
    # Target correlated with age and spend
    y = ((ages > 40) | (spends > 250.0)).astype(int)

    df = pd.DataFrame({
        "user_id": ids,
        "age": ages,
        "join_date": dates,
        "plan": plans,
        "monthly_spend": spends,
        "churn": y,
    })

    # 1. Target selection and readiness check
    target_col = "churn"
    target_quality = MLPlanner.check_target_quality(df, target_col)
    assert target_quality["is_valid"] is True

    # 2. Feature analysis
    f_analysis = MLPlanner.analyze_features(df, target_col)
    assert "user_id" in f_analysis["dropped_features"]
    assert "user_id" not in f_analysis["numeric_cols"]
    assert "join_date" in f_analysis["datetime_cols"]
    assert "churn" not in f_analysis["numeric_cols"]

    # 3. AutoML Training
    summary = AutoMLTrainer.train(df, target_column=target_col, task_type=MLTaskType.BINARY_CLASSIFICATION)

    assert summary.rows_trained + summary.rows_tested == n
    assert summary.best_model_name != "None"
    assert summary.best_pipeline is not None

    # 4. Fitted Pipeline Prediction Verification
    sample_test_X = df.drop(columns=[target_col, "user_id"]).head(10)
    preds = summary.best_pipeline.predict(sample_test_X)
    assert len(preds) == 10

    # 5. Feature importances mapped back from fitted best pipeline
    assert len(summary.feature_importances) > 0
    top_feature_names = [fi.feature for fi in summary.feature_importances]
    assert any("age" in f or "spend" in f or "join_date" in f for f in top_feature_names)

    # 6. Explainability Service
    exp_report = ExplainabilityService.explain(summary)
    assert exp_report.best_model_name == summary.best_model_name
    assert len(exp_report.top_driver_features) > 0

    # 7. AI Insights Service (compact structured input, zero raw data sent)
    insights_report = InsightsService.synthesize(automl_summary=summary, explainability=exp_report)
    assert insights_report.headline is not None
    assert len(insights_report.strategic_recommendations) > 0

    # 8. Final Report compilation
    markdown_report = ReportGeneratorService.generate_markdown_report(
        dataset_name="test_integration.csv",
        automl_summary=summary,
        explainability=exp_report,
        insights=insights_report,
    )
    assert "# 📊 AutoDS AI Studio — Executive Data Science Report" in markdown_report
    assert summary.best_model_name in markdown_report


def test_new_dataset_upload_resets_ml_state(monkeypatch):
    """Verify uploading a new dataset invalidates stale trained models, predictions,
    explainability results, AI insights, and reports.
    """
    dummy_state = {}
    monkeypatch.setattr("streamlit.session_state", dummy_state)

    initialize_state()

    # Populate state with dummy trained model results
    dummy_state["automl_summary"] = "STALE_AUTOML_SUMMARY"
    dummy_state["explainability_results"] = "STALE_EXPLAINABILITY"
    dummy_state["ai_insights"] = "STALE_INSIGHTS"
    dummy_state["final_report"] = "STALE_REPORT"
    dummy_state["ml_plan"] = "STALE_ML_PLAN"

    new_df = pd.DataFrame({"col_1": [1, 2, 3], "col_2": [10, 20, 30]})
    set_dataset(new_df, "new_dataset.csv", is_original=True)

    # Verify all ML state keys are cleared
    assert dummy_state["automl_summary"] is None
    assert dummy_state["explainability_results"] is None
    assert dummy_state["ai_insights"] is None
    assert dummy_state["final_report"] is None
    assert dummy_state["ml_plan"] is None
    assert dummy_state["dataset_name"] == "new_dataset.csv"


def test_error_handling_boundaries():
    """Verify ML pipeline gracefully handles error edge cases without unhandled crashes."""
    df_valid = pd.DataFrame({"x": np.random.randn(20), "y": np.random.randint(0, 2, size=20)})

    # 1. Target column non-existent
    summary_missing = AutoMLTrainer.train(df_valid, target_column="non_existent_col")
    assert summary_missing.rows_trained == 0
    assert len(summary_missing.warnings) > 0

    # 2. Target constant / zero variance
    df_const_target = pd.DataFrame({"x": np.random.randn(20), "y_const": [1] * 20})
    summary_const = AutoMLTrainer.train(df_const_target, target_column="y_const")
    assert summary_const.rows_trained == 0
    assert len(summary_const.warnings) > 0

    # 3. Insufficient non-null target rows (< 10)
    df_small_target = pd.DataFrame({"x": range(20), "y_sparse": [1, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]})
    summary_small = AutoMLTrainer.train(df_small_target, target_column="y_sparse")
    assert summary_small.rows_trained == 0
    assert len(summary_small.warnings) > 0

    # 4. Dataset with only identifier feature
    uuids = [f"UUID_{i}" for i in range(20)]
    df_only_id = pd.DataFrame({"customer_uuid": uuids, "target_y": np.random.randint(0, 2, size=20)})
    summary_id_only = AutoMLTrainer.train(df_only_id, target_column="target_y")
    assert summary_id_only.rows_trained == 0
    assert len(summary_id_only.warnings) > 0
    assert "No usable predictive features" in summary_id_only.warnings[0]
