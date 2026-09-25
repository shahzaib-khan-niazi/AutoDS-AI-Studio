"""Repair operation models and issue taxonomy."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class RepairOperation(str, Enum):
    """Supported repair operations."""

    REMOVE_DUPLICATE_ROWS = "remove_duplicate_rows"
    RENAME_COLUMNS = "rename_columns"
    DROP_EMPTY_ROWS = "drop_empty_rows"
    DROP_EMPTY_COLUMNS = "drop_empty_columns"
    CONVERT_TYPES = "convert_types"
    FILL_MISSING = "fill_missing"
    DROP_HIGH_MISSING_COLUMNS = "drop_high_missing_columns"
    HANDLE_OUTLIERS = "handle_outliers"
    UNPIVOT = "unpivot"
    FLATTEN_HEADERS = "flatten_headers"
    REMOVE_TITLE_ROWS = "remove_title_rows"
    STRIP_WHITESPACE = "strip_whitespace"
    STANDARDIZE_VALUES = "standardize_values"
    AUTO_DTYPES = "auto_dtypes"
    NORMALIZE_DATES = "normalize_dates"
    EXPLODE_MULTI_VALUE_CELLS = "explode_multi_value_cells"
    REMOVE_REPEATED_HEADERS = "remove_repeated_headers"
    REMOVE_METADATA_ROWS = "remove_metadata_rows"
    REMOVE_SPACER_ROWS_COLS = "remove_spacer_rows_cols"
    UNPIVOT_HORIZONTAL_CATEGORY_BLOCKS = "unpivot_horizontal_category_blocks"
    REMOVE_SUBTOTAL_ELEMENTS = "remove_subtotal_elements"
    AUTO_RECONSTRUCT_STRUCTURE = "auto_reconstruct_structure"
    RECONSTRUCT_EMBEDDED_RECORDS = "reconstruct_embedded_records"
    REMOVE_ROWS = "remove_rows"
    REMOVE_COLUMNS = "remove_columns"
    REMOVE_ROWS_AND_COLUMNS = "remove_rows_and_columns"
    REPLACE_VALUES = "replace_values"


class IssueTaxonomy(str, Enum):
    """Unified issue taxonomy across the data consistency engine."""

    STRUCTURAL = "STRUCTURAL"
    FORMAT = "FORMAT"
    SEMANTIC = "SEMANTIC"
    TYPOGRAPHICAL = "TYPOGRAPHICAL"
    DATATYPE = "DATATYPE"
    MISSING = "MISSING"
    DUPLICATE = "DUPLICATE"
    OUTLIER = "OUTLIER"
    WHITESPACE = "WHITESPACE"
    DATE = "DATE"
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    IDENTIFIER = "IDENTIFIER"
    CURRENCY = "CURRENCY"
    PERCENTAGE = "PERCENTAGE"
    IMPOSSIBLE_VALUE = "IMPOSSIBLE_VALUE"


class SemanticType(str, Enum):
    """Inferred semantic datatype classification."""

    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    STRING = "string"
    IDENTIFIER = "identifier"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    TEXT = "text"


class ConfidenceLevel(str, Enum):
    """Confidence classification for proposed repairs."""

    SAFE = "SAFE"      # >= 0.95: safe for automatic execution (when deterministic evidence exists)
    HIGH = "HIGH"      # 0.85 - 0.94: high confidence, recommendation
    REVIEW = "REVIEW"  # 0.70 - 0.84: user review recommended
    LOW = "LOW"        # < 0.70: do not apply automatically


class TypeInferenceResult(BaseModel):
    """Result of confidence-based semantic datatype inference for a column."""

    column: str
    detected_type: str
    confidence: float = 1.0
    evidence: list[str] = Field(default_factory=list)
    original_dtype: str = "object"
    sample_values: list[Any] = Field(default_factory=list)
    suspicious_values: list[Any] = Field(default_factory=list)
    format_patterns: list[str] = Field(default_factory=list)
    is_ambiguous: bool = False
    suggested_action: str = "convert"

    @property
    def inferred_type(self) -> str:
        return self.detected_type


class DuplicateDetectionResult(BaseModel):
    """Comprehensive duplicate row, near-duplicate, and identifier collision result."""

    exact_duplicates_count: int = 0
    near_duplicates_count: int = 0
    identifier_duplicates: dict[str, int] = Field(default_factory=dict)
    affected_rows: list[int] = Field(default_factory=list)
    recommended_action: str = "review"
    details: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def exact_duplicate_rows(self) -> int:
        return self.exact_duplicates_count

    @property
    def near_duplicate_rows(self) -> int:
        return self.near_duplicates_count

    @property
    def duplicate_identifiers(self) -> dict[str, int]:
        return self.identifier_duplicates


class AuditLogEntry(BaseModel):
    """Detailed audit record representing a single change to the dataset."""

    column: Optional[str] = None
    issue_type: str = "GENERAL"
    original_value: Optional[Any] = None
    new_value: Optional[Any] = None
    method: str = "deterministic"  # "deterministic" | "ai" | "user"
    confidence: float = 1.0
    reason: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    status: str = "applied"  # "applied" | "rejected" | "flagged"
    risk_level: str = "safe"  # "safe" | "medium" | "high"


class SemanticCandidateGroup(BaseModel):
    """A group of equivalent value representations identified in a categorical column."""

    column: str
    canonical_value: str
    variants: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float = 1.0
    issue_type: IssueTaxonomy = IssueTaxonomy.SEMANTIC
    method: str = "semantic_clustering"
    is_ambiguous: bool = False
    possible_meanings: list[str] = Field(default_factory=list)
    affected_rows: int = 0


class DateNormalizationResult(BaseModel):
    """Result summary of general date parsing and ambiguity detection for a column."""

    column: str
    detected_semantic_type: str = "datetime"
    inferred_convention: str = "ISO"  # 'DMY', 'MDY', 'YMD', 'ISO', 'MIXED', 'AMBIGUOUS', 'UNKNOWN'
    dayfirst: bool = False
    is_ambiguous: bool = False
    has_time: bool = False
    has_timezone: bool = False
    total_non_null: int = 0
    parsed_count: int = 0
    unambiguous_count: int = 0
    ambiguous_count: int = 0
    invalid_count: int = 0
    mixed_conventions_detected: bool = False
    detected_formats: list[str] = Field(default_factory=list)
    invalid_samples: list[str] = Field(default_factory=list)
    row_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    parse_ratio: float = 0.0
    confidence: float = 1.0

    @property
    def notes(self) -> str:
        if self.mixed_conventions_detected:
            return "MIXED DATE CONVENTIONS DETECTED — REVIEW REQUIRED"
        if self.is_ambiguous:
            return "AMBIGUOUS DATE — REVIEW REQUIRED"
        return f"Convention: {self.inferred_convention}"




class RepairAction(BaseModel):
    """A single repair action to be executed."""

    operation: RepairOperation
    target: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    confidence: float = 1.0
    safe: bool = True
    issue_type: Optional[IssueTaxonomy] = None


class ValidationResult(BaseModel):
    """Result of post-repair validation."""

    valid: bool = True
    confidence: float = 1.0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RepairRecord(BaseModel):
    """Record of a single repair operation for history tracking."""

    operation: str
    timestamp: datetime = Field(default_factory=datetime.now)
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    success: bool = True
    warnings: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    column: Optional[str] = None
    issue_type: Optional[str] = None
    original_value: Optional[Any] = None
    new_value: Optional[Any] = None
    method: str = "deterministic"
    confidence: float = 1.0
    reason: str = ""
    status: str = "applied"
    risk_level: str = "safe"


class RepairResult(BaseModel):
    """Complete result of a repair operation."""

    success: bool = True
    actions_applied: list[str] = Field(default_factory=list)
    records: list[RepairRecord] = Field(default_factory=list)
    validation: ValidationResult = Field(default_factory=ValidationResult)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
