"""Schemas for AI Cleaning Planner, Validator, and Executor."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class CleaningAction(BaseModel):
    """A discrete, validated data cleaning operation."""

    operation: Literal[
        "fill_missing_numeric",
        "fill_missing_categorical",
        "drop_missing_rows",
        "drop_missing_column",
        "drop_duplicates",
        "convert_dtype",
        "normalize_categories",
        "remove_constant_column",
        "cap_outliers",
        "strip_whitespace",
        "explode_multi_value_cells",
        "remove_repeated_headers",
        "remove_metadata_rows",
        "unpivot",
        "flatten_headers",
    ]
    column: Optional[str] = None
    method: Optional[str] = None  # e.g., "median", "mean", "mode", "constant", "iqr", "zscore"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reason: str
    evidence: str
    risk: Literal["low", "medium", "high"] = "low"
    priority: Literal["critical", "high", "medium", "low"] = "medium"


class CleaningPlan(BaseModel):
    """Structured AI Cleaning Plan derived from a DatasetProfile."""

    summary: str
    actions: List[CleaningAction] = Field(default_factory=list)
    priority_order: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.85, ge=0.0, le=1.0)
    confidence_level: str = "High"


class CleaningValidationResult(BaseModel):
    """Result of validating an AI cleaning plan against available columns and constraints."""

    is_valid: bool
    approved_actions: List[CleaningAction] = Field(default_factory=list)
    rejected_actions: List[Dict[str, Any]] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)


class CleaningExecutionResult(BaseModel):
    """Quantitative summary of changes made by the deterministic cleaning executor."""

    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    missing_before: int
    missing_after: int
    duplicates_before: int
    duplicates_after: int
    operations_applied: List[str] = Field(default_factory=list)
    changed_columns: List[str] = Field(default_factory=list)
    diff_summary: str = ""
