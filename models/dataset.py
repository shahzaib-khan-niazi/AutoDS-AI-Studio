"""Dataset metadata models."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """Metadata about an uploaded dataset."""

    filename: str
    rows: int = 0
    columns: int = 0
    memory_usage_bytes: int = 0
    column_names: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    merged_cells: list[str] = Field(default_factory=list)
    sheet_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def memory_usage_mb(self) -> float:
        """Memory usage in megabytes."""
        return self.memory_usage_bytes / (1024 * 1024)

