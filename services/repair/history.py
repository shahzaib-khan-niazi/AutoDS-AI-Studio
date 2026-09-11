"""Repair history tracking.

Maintains a chronological log of all repair operations applied to the dataset.
"""

from models.repair import RepairRecord
from core.logging import logger


class RepairHistory:
    """Tracks repair operations in order."""

    def __init__(self) -> None:
        self._records: list[RepairRecord] = []

    def add(self, record: RepairRecord) -> None:
        """Add a repair record to history.

        Args:
            record: The RepairRecord to add.
        """
        self._records.append(record)
        logger.debug(
            "History: {} ({} → {} rows)",
            record.operation,
            record.rows_before,
            record.rows_after,
        )

    @property
    def records(self) -> list[RepairRecord]:
        """Get all repair records."""
        return list(self._records)

    @property
    def count(self) -> int:
        """Number of repairs in history."""
        return len(self._records)

    def summary(self) -> list[dict[str, object]]:
        """Get a summary of all repairs for display.

        Returns:
            List of dicts with operation, rows_before, rows_after, success.
        """
        result: list[dict[str, object]] = []
        for record in self._records:
            result.append({
                "operation": record.operation,
                "timestamp": str(record.timestamp),
                "rows_before": record.rows_before,
                "rows_after": record.rows_after,
                "columns_before": record.columns_before,
                "columns_after": record.columns_after,
                "success": record.success,
            })
        return result
