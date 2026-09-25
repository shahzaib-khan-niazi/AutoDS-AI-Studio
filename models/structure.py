"""Structure detection models."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class StructureType(str, Enum):
    """Types of dataset structures that can be detected."""

    NORMAL_TABLE = "NORMAL_TABLE"
    WIDE_TABLE = "WIDE_TABLE"
    LONG_TABLE = "LONG_TABLE"
    PIVOT_TABLE = "PIVOT_TABLE"
    CROSSTAB = "CROSSTAB"
    MULTI_HEADER = "MULTI_HEADER"
    REPEATED_CATEGORY_BLOCKS = "REPEATED_CATEGORY_BLOCKS"
    SPACER_ARTIFACTS = "SPACER_ARTIFACTS"
    SUBTOTAL_COLUMNS_ROWS = "SUBTOTAL_COLUMNS_ROWS"
    METADATA_BANNERS = "METADATA_BANNERS"
    TIME_SERIES = "TIME_SERIES"
    SURVEY = "SURVEY"
    TRANSACTIONAL = "TRANSACTIONAL"
    EMBEDDED_STRUCTURED_RECORDS = "EMBEDDED_STRUCTURED_RECORDS"
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


class StructuralAudit(BaseModel):
    """Line-level and cell-level audit of a structural transformation."""

    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    source_columns: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(default_factory=list)
    source_cells_considered: int = 0
    source_cells_retained: int = 0
    source_cells_discarded: int = 0
    discarded_cell_reasons: dict[str, int] = Field(default_factory=dict)
    confidence_score: float = 1.0
    ambiguity_flags: list[str] = Field(default_factory=list)
    validation_result: dict[str, Any] = Field(default_factory=dict)


class StructuralPlan(BaseModel):
    """Validated structural repair plan."""

    detected_issue: str
    evidence: list[str] = Field(default_factory=list)
    proposed_transformation: str
    source_rows: list[int] = Field(default_factory=list)
    source_columns: list[str] = Field(default_factory=list)
    target_schema: dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0
    ambiguity_flags: list[str] = Field(default_factory=list)
    requires_human_approval: bool = True
    before_after_sample: dict[str, Any] = Field(default_factory=dict)
    audit: Optional[StructuralAudit] = None

