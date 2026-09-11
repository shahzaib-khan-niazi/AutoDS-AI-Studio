"""Application constants for AutoDS AI Studio.

Fixed values used throughout the application.
"""

from typing import Final

# Supported file extensions for upload
SUPPORTED_EXTENSIONS: Final[list[str]] = [".csv", ".xlsx", ".xls", ".parquet"]

# Preview limits
MAX_PREVIEW_ROWS: Final[int] = 500
DEFAULT_PREVIEW_ROWS: Final[int] = 50

# Quality thresholds
HIGH_MISSING_THRESHOLD: Final[float] = 0.5
HIGH_CARDINALITY_THRESHOLD: Final[float] = 0.95
CONSTANT_COLUMN_THRESHOLD: Final[int] = 1

# Structure detection
MIN_WIDE_COLUMNS: Final[int] = 8
MIN_CONFIDENCE_THRESHOLD: Final[float] = 0.3

# Repair safety
MAX_ROW_LOSS_PERCENTAGE: Final[float] = 0.5
MAX_COLUMN_LOSS_PERCENTAGE: Final[float] = 0.5

# Application info
APP_NAME: Final[str] = "AutoDS AI Studio"
APP_VERSION: Final[str] = "1.0.0"
APP_ICON: Final[str] = "🤖"
