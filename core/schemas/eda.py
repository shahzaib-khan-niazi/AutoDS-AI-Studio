"""Schemas for EDA metrics and AI EDA interpretation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CorrelationItem(BaseModel):
    """Correlation metric between two columns."""

    feature_a: str
    feature_b: str
    correlation: float
    strength: str  # "Strong Positive", "Moderate Positive", "Strong Negative", etc.


class AIEDAAnalysis(BaseModel):
    """Structured AI interpretation of Exploratory Data Analysis."""

    overview: str
    key_patterns: List[str] = Field(default_factory=list)
    significant_correlations: List[str] = Field(default_factory=list)
    distribution_insights: List[str] = Field(default_factory=list)
    potential_interactions: List[str] = Field(default_factory=list)
    anomalous_observations: List[str] = Field(default_factory=list)
    recommendations_for_modeling: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.85, ge=0.0, le=1.0)
