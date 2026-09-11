"""Schemas for AI ML Planning, Explainability, and Synthesis Insights."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class AIMLPlan(BaseModel):
    """Structured AI Machine Learning Plan."""

    recommended_task: Literal["binary_classification", "multiclass_classification", "regression", "clustering_or_unsupervised"]
    target_column: Optional[str] = None
    target_reasoning: str
    recommended_features: List[str] = Field(default_factory=list)
    excluded_features: List[str] = Field(default_factory=list)
    exclusion_reasons: Dict[str, str] = Field(default_factory=dict)
    candidate_models: List[str] = Field(default_factory=list)
    primary_evaluation_metric: str
    metric_rationale: str
    potential_data_leakage_risks: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.85, ge=0.0, le=1.0)


class AIExplainabilityReport(BaseModel):
    """Structured AI explanation of trained model feature importances."""

    best_model_name: str
    primary_metric_name: str
    primary_metric_score: float
    top_driver_features: List[str] = Field(default_factory=list)
    feature_impact_summary: str
    key_findings: List[str] = Field(default_factory=list)
    potential_biases_or_risks: List[str] = Field(default_factory=list)
    practical_takeaways: List[str] = Field(default_factory=list)


class AIInsightsReport(BaseModel):
    """Cross-stage executive insights synthesized across profile, cleaning, EDA, and ML."""

    headline: str
    executive_summary: str
    data_quality_verdict: str
    key_business_insights: List[str] = Field(default_factory=list)
    modeling_and_predictive_power: str
    strategic_recommendations: List[str] = Field(default_factory=list)
    critical_risks_and_caveats: List[str] = Field(default_factory=list)
    suggested_next_actions: List[str] = Field(default_factory=list)
    synthesis_timestamp: str = ""
