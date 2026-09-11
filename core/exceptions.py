"""Custom exceptions for AutoDS AI Studio.

All application-specific exceptions inherit from AutoDSException.
Use these instead of generic exceptions for clearer error handling.
"""


class AutoDSException(Exception):
    """Base exception for all AutoDS AI Studio errors."""

    def __init__(self, message: str = "", details: str = "") -> None:
        self.message = message
        self.details = details
        super().__init__(self.message)


class DatasetLoadError(AutoDSException):
    """Raised when a dataset cannot be loaded from file."""
    pass


class DatasetValidationError(AutoDSException):
    """Raised when a loaded dataset fails validation checks."""
    pass


class StructureDetectionError(AutoDSException):
    """Raised when structure detection encounters an error."""
    pass


class RepairError(AutoDSException):
    """Raised when a repair operation fails."""
    pass


class RepairValidationError(AutoDSException):
    """Raised when post-repair validation fails."""
    pass


class AIServiceError(AutoDSException):
    """Raised when the AI service encounters an error."""
    pass


class AIValidationError(AutoDSException):
    """Raised when AI output fails schema or safety validation."""
    pass
