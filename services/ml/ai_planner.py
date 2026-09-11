"""AI Machine Learning Planner for AutoDS AI Studio.

Evaluates dataset characteristics and candidate targets, recommends problem formulation (Classification vs Regression),
feature selection, candidate models, and primary evaluation metrics using LLMService.
"""

import json
from typing import Any, Dict, List, Optional
import pandas as pd

from core.logging import logger
from core.llm.config import llm_config
from core.llm.service import LLMService
from core.llm.exceptions import LLMError
from core.utils.json_sanitizer import sanitize_for_json, safe_json_dumps
from core.schemas.dataset_profile import DatasetProfile
from core.schemas.ml import AIMLPlan
from services.ml.planner import MLPlanner
from models.ml import MLTaskType


class AIMLPlanner:
    """Generates structured ML training strategies and recommendations."""

    _SYSTEM_PROMPT = """You are the AI Machine Learning Planner for AutoDS AI Studio.
Your role is to analyze a dataset summary and recommend the most effective machine learning formulation.

CONSTRAINTS & RULES:
1. ONLY recommend target columns and features that exist in the profile.
2. Formulate task as either: "binary_classification", "multiclass_classification", "regression", or "clustering_or_unsupervised".
3. Exclude identifier columns, constant columns, and any columns likely to cause data leakage.
4. Recommend Scikit-Learn models (e.g., RandomForest, GradientBoosting, LogisticRegression, Ridge).
5. Recommend appropriate evaluation metrics (Accuracy, F1, Precision, Recall for classification; R2, RMSE, MAE for regression).

Output valid JSON matching this schema:
{
  "recommended_task": "binary_classification",
  "target_column": "Churn",
  "target_reasoning": "Binary outcome representing customer churn event.",
  "recommended_features": ["Age", "MonthlyCharges", "Tenure"],
  "excluded_features": ["CustomerID"],
  "exclusion_reasons": {"CustomerID": "Unique identifier with no predictive value"},
  "candidate_models": ["Random Forest Classifier", "Gradient Boosting Classifier", "Logistic Regression"],
  "primary_evaluation_metric": "F1 Score",
  "metric_rationale": "Balances precision and recall in imbalanced class distributions.",
  "potential_data_leakage_risks": ["Check that features are measured strictly before the outcome"]
}
"""

    @classmethod
    def plan(
        cls, df: pd.DataFrame, profile: Optional[DatasetProfile] = None, user_target: Optional[str] = None
    ) -> AIMLPlan:
        """Generate structured ML plan for dataset.

        Args:
            df: DataFrame to model.
            profile: Optional precomputed DatasetProfile.
            user_target: Optional user-specified target column.

        Returns:
            AIMLPlan object.
        """
        # Determine candidate target and auto-detected task
        target_col = user_target or cls._detect_candidate_target(df, profile)
        detected_task = MLPlanner.detect_task_type(df, target_col) if target_col else MLTaskType.BINARY_CLASSIFICATION

        summary_dict = sanitize_for_json(cls._prepare_ml_summary(df, profile, target_col, detected_task))

        if not llm_config.is_configured:
            logger.warning("AIMLPlanner called without configured API key; using deterministic fallback")
            return cls._fallback_plan(df, target_col, detected_task)

        user_prompt = f"Develop a machine learning strategy for this dataset:\n{safe_json_dumps(summary_dict, indent=2)}"

        messages = [
            {"role": "system", "content": cls._SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        schema_hint = {
            "name": "AIMLPlan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "recommended_task": {
                        "type": "string",
                        "enum": ["binary_classification", "multiclass_classification", "regression", "clustering_or_unsupervised"],
                    },
                    "target_column": {"type": ["string", "null"]},
                    "target_reasoning": {"type": "string"},
                    "recommended_features": {"type": "array", "items": {"type": "string"}},
                    "excluded_features": {"type": "array", "items": {"type": "string"}},
                    "exclusion_reasons": {"type": "object"},
                    "candidate_models": {"type": "array", "items": {"type": "string"}},
                    "primary_evaluation_metric": {"type": "string"},
                    "metric_rationale": {"type": "string"},
                    "potential_data_leakage_risks": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "recommended_task",
                    "target_reasoning",
                    "recommended_features",
                    "excluded_features",
                    "exclusion_reasons",
                    "candidate_models",
                    "primary_evaluation_metric",
                    "metric_rationale",
                    "potential_data_leakage_risks",
                ],
            },
        }

        try:
            svc = LLMService()
            result = svc.generate_structured(messages, response_schema=schema_hint)
            raw = json.loads(result.content)

            task_str = raw.get("recommended_task", "binary_classification")
            target_str = raw.get("target_column") or target_col

            return AIMLPlan(
                recommended_task=task_str,
                target_column=target_str,
                target_reasoning=raw.get("target_reasoning", "Selected predictive target."),
                recommended_features=[str(f) for f in raw.get("recommended_features", []) if f in df.columns and f != target_str],
                excluded_features=[str(e) for e in raw.get("excluded_features", [])],
                exclusion_reasons={str(k): str(v) for k, v in raw.get("exclusion_reasons", {}).items()},
                candidate_models=[str(m) for m in raw.get("candidate_models", [])],
                primary_evaluation_metric=raw.get("primary_evaluation_metric", "Accuracy"),
                metric_rationale=raw.get("metric_rationale", "Standard performance metric."),
                potential_data_leakage_risks=[str(r) for r in raw.get("potential_data_leakage_risks", [])],
                confidence_score=0.90,
            )

        except LLMError as e:
            logger.error("LLM error during ML planning: {}", e.message)
            return cls._fallback_plan(df, target_col, detected_task)
        except Exception as e:
            logger.exception("Unexpected error in AIMLPlanner: {}", str(e))
            return cls._fallback_plan(df, target_col, detected_task)

    @classmethod
    def _detect_candidate_target(cls, df: pd.DataFrame, profile: Optional[DatasetProfile] = None) -> Optional[str]:
        """Heuristic for selecting most likely target column."""
        # Check column names with target-like tokens
        target_tokens = ["target", "label", "churn", "price", "sales", "salary", "outcome", "status", "class", "default"]
        for col in df.columns:
            if any(tok in col.lower() for tok in target_tokens):
                return col

        # Fallback to last column
        return df.columns[-1] if len(df.columns) > 0 else None

    @classmethod
    def _prepare_ml_summary(
        cls, df: pd.DataFrame, profile: Optional[DatasetProfile], target_col: Optional[str], task_type: MLTaskType
    ) -> Dict[str, Any]:
        """Build compact summary for ML strategy generation."""
        cols_info = []
        for c in df.columns:
            cols_info.append({
                "name": c,
                "dtype": str(df[c].dtype),
                "unique_count": df[c].nunique(),
                "missing_pct": f"{(df[c].isna().sum() / max(len(df), 1)) * 100:.1f}%",
            })

        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "target_candidate": target_col,
            "detected_task_type": task_type.value,
            "columns": cols_info,
            "possible_id_columns": profile.possible_id_columns if profile else [],
            "constant_columns": profile.constant_columns if profile else [],
        }

    @classmethod
    def _fallback_plan(
        cls, df: pd.DataFrame, target_col: Optional[str], task_type: MLTaskType
    ) -> AIMLPlan:
        """Deterministic fallback ML plan when AI is unconfigured or fails."""
        is_reg = task_type == MLTaskType.REGRESSION
        task_str = "regression" if is_reg else "binary_classification"
        metric_str = "R² Score" if is_reg else "Accuracy"

        features = [c for c in df.columns if c != target_col and df[c].nunique() > 1]
        excluded = [c for c in df.columns if c == target_col or df[c].nunique() <= 1]

        candidates = (
            ["Random Forest Regressor", "Gradient Boosting Regressor", "Ridge Regression"]
            if is_reg
            else ["Random Forest Classifier", "Gradient Boosting Classifier", "Logistic Regression"]
        )

        return AIMLPlan(
            recommended_task=task_str,
            target_column=target_col,
            target_reasoning=f"Auto-detected target '{target_col}' ({task_type.value}).",
            recommended_features=features,
            excluded_features=excluded,
            exclusion_reasons={c: "Target or zero-variance column" for c in excluded},
            candidate_models=candidates,
            primary_evaluation_metric=metric_str,
            metric_rationale="Standard primary evaluation metric for task.",
            potential_data_leakage_risks=["Ensure features are available at inference time."],
            confidence_score=0.80,
        )
