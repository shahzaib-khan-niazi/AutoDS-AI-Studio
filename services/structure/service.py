"""Structure service — public API for structure detection."""

import pandas as pd

from core.logging import logger
from core.exceptions import StructureDetectionError
from models.structure import StructureResult
from services.structure.detector import StructureDetector


class StructureService:
    """Service for detecting dataset structure."""

    @classmethod
    def detect(cls, df: pd.DataFrame) -> StructureResult:
        """Detect the structure type of a DataFrame.

        Args:
            df: Source DataFrame.

        Returns:
            StructureResult.

        Raises:
            StructureDetectionError: If detection fails.
        """
        try:
            return StructureDetector.detect(df)
        except Exception as e:
            logger.exception("Structure detection failed: {}", str(e))
            raise StructureDetectionError(
                message="Failed to detect dataset structure.",
                details=str(e),
            )
