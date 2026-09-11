"""Structure detection models."""

from enum import Enum
from pydantic import BaseModel, Field


class StructureType(str, Enum):
    """Types of dataset structures that can be detected."""

    NORMAL_TABLE = "NORMAL_TABLE"
    WIDE_TABLE = "WIDE_TABLE"
    LONG_TABLE = "LONG_TABLE"
    PIVOT_TABLE = "PIVOT_TABLE"
    CROSSTAB = "CROSSTAB"
    MULTI_HEADER = "MULTI_HEADER"
    TIME_SERIES = "TIME_SERIES"
    SURVEY = "SURVEY"
    TRANSACTIONAL = "TRANSACTIONAL"
    UNKNOWN = "UNKNOWN"


class RuleEvidence(BaseModel):
    """Evidence produced by a single detection rule."""

    rule_name: str
    structure_type: StructureType
    score: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    reason: str = ""


class StructureResult(BaseModel):
    """Final structure detection result."""

    structure: StructureType = StructureType.UNKNOWN
    confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    all_scores: list[RuleEvidence] = Field(default_factory=list)
    recommended_action: str = ""
    warnings: list[str] = Field(default_factory=list)
