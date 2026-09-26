"""Machine Learning models and schemas for AutoML on clean datasets."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class MLTaskType(str, Enum):
    """Machine Learning task classification."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class FeatureImportance(BaseModel):
    """Feature importance item."""

    feature: str
    importance: float


class ModelEvaluationResult(BaseModel):
    """Evaluation metrics for a trained model."""

    model_name: str
    task_type: MLTaskType
    metrics: dict[str, float] = Field(default_factory=dict)
    train_score: float = 0.0
    test_score: float = 0.0
    cv_score_mean: float = 0.0
    cv_score_std: float = 0.0
    primary_metric_name: str = "Score"
    primary_metric_value: float = 0.0
    fit_time_seconds: float = 0.0
    is_baseline: bool = False


class AutoMLSummary(BaseModel):
    """Summary of AutoML model training and leaderboard."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_column: str
    task_type: MLTaskType
    rows_trained: int = 0
    rows_tested: int = 0
    features_used: list[str] = Field(default_factory=list)
    dropped_features: dict[str, str] = Field(default_factory=dict)
    leaderboard: list[ModelEvaluationResult] = Field(default_factory=list)
    best_model_name: str = ""
    baseline_model_name: str = ""
    baseline_metric_value: float = 0.0
    best_pipeline: Optional[Any] = Field(default=None, exclude=True)
    feature_importances: list[FeatureImportance] = Field(default_factory=list)
    ml_readiness_report: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

