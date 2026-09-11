"""Inspector service — main interface for dataset inspection.

Combines statistics and quality assessment into a single InspectionResult.
"""

import pandas as pd

from core.logging import logger
from core.exceptions import AutoDSException
from models.inspection import InspectionResult, ColumnDetail
from services.inspector.statistics import (
    compute_numeric_stats,
    compute_categorical_stats,
    compute_datetime_stats,
)
from services.inspector.quality import assess_quality


class InspectorService:
    """Service for inspecting datasets."""

    @classmethod
    def inspect(cls, df: pd.DataFrame) -> InspectionResult:
        """Perform a full inspection of a DataFrame.

        Args:
            df: The DataFrame to inspect.

        Returns:
            InspectionResult with statistics and quality assessment.
        """
        logger.info("Starting inspection ({} rows × {} cols)", len(df), len(df.columns))

        # Column details
        column_details = cls._build_column_details(df)

        # Statistics by dtype category
        numeric_stats = compute_numeric_stats(df)
        categorical_stats = compute_categorical_stats(df)
        datetime_stats = compute_datetime_stats(df)

        # Quality assessment
        quality = assess_quality(df)

        result = InspectionResult(
            row_count=len(df),
            column_count=len(df.columns),
            column_details=column_details,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            datetime_stats=datetime_stats,
            quality=quality,
        )

        logger.info(
            "Inspection complete — quality score: {:.0%}", quality.quality_score
        )

        return result

    @staticmethod
    def _build_column_details(df: pd.DataFrame) -> list[ColumnDetail]:
        """Build detailed information for each column.

        Args:
            df: Source DataFrame.

        Returns:
            List of ColumnDetail, one per column.
        """
        details: list[ColumnDetail] = []
        total_rows = len(df)

        for col in df.columns:
            col_str = str(col)
            series = df[col]
            non_null = int(series.notna().sum())
            null_count = int(series.isna().sum())
            null_pct = null_count / total_rows if total_rows > 0 else 0.0
            unique = int(series.nunique())
            unique_pct = unique / total_rows if total_rows > 0 else 0.0

            # Check if potential ID column
            is_id = unique == total_rows and total_rows > 1

            # Check if constant
            is_const = unique <= 1 and non_null > 0

            # Check for mixed types in object columns
            is_mixed = False
            if series.dtype == object:
                non_null_vals = series.dropna()
                if len(non_null_vals) > 0:
                    types_found = set()
                    for val in non_null_vals.head(100):
                        types_found.add(type(val).__name__)
                    is_mixed = len(types_found) > 1

            # Sample values (up to 5 unique non-null)
            sample_vals: list[str] = []
            try:
                non_null_vals = series.dropna().unique()
                for v in non_null_vals[:5]:
                    sample_vals.append(str(v))
            except Exception:
                pass

            details.append(ColumnDetail(
                name=col_str,
                dtype=str(series.dtype),
                non_null_count=non_null,
                null_count=null_count,
                null_percentage=null_pct,
                unique_count=unique,
                unique_percentage=unique_pct,
                is_potential_id=is_id,
                is_constant=is_const,
                is_mixed_type=is_mixed,
                sample_values=sample_vals,
            ))

        return details
