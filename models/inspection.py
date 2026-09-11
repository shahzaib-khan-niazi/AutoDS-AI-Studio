"""Inspection result models."""

from pydantic import BaseModel, Field


class ColumnDetail(BaseModel):
    """Detailed information about a single column."""

    name: str
    dtype: str
    non_null_count: int = 0
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    unique_percentage: float = 0.0
    is_potential_id: bool = False
    is_constant: bool = False
    is_mixed_type: bool = False
    sample_values: list[str] = Field(default_factory=list)


class NumericStats(BaseModel):
    """Statistics for a numeric column."""

    column: str
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    skewness: float = 0.0


class CategoricalStats(BaseModel):
    """Statistics for a categorical column."""

    column: str
    count: int = 0
    unique: int = 0
    top: str = ""
    top_frequency: int = 0


class DatetimeStats(BaseModel):
    """Statistics for a datetime column."""

    column: str
    count: int = 0
    min_date: str = ""
    max_date: str = ""


class QualityReport(BaseModel):
    """Data quality assessment."""

    total_cells: int = 0
    missing_cells: int = 0
    missing_percentage: float = 0.0
    duplicate_rows: int = 0
    empty_rows: int = 0
    empty_columns: list[str] = Field(default_factory=list)
    constant_columns: list[str] = Field(default_factory=list)
    mixed_type_columns: list[str] = Field(default_factory=list)
    high_cardinality_columns: list[str] = Field(default_factory=list)
    possible_id_columns: list[str] = Field(default_factory=list)
    suspicious_column_names: list[str] = Field(default_factory=list)
    quality_score: float = 1.0
    quality_notes: list[str] = Field(default_factory=list)


class InspectionResult(BaseModel):
    """Complete inspection result combining all analysis."""

    row_count: int = 0
    column_count: int = 0
    column_details: list[ColumnDetail] = Field(default_factory=list)
    numeric_stats: list[NumericStats] = Field(default_factory=list)
    categorical_stats: list[CategoricalStats] = Field(default_factory=list)
    datetime_stats: list[DatetimeStats] = Field(default_factory=list)
    quality: QualityReport = Field(default_factory=QualityReport)
