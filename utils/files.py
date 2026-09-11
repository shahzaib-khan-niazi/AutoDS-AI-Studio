"""File utility functions."""

import os
from datetime import datetime
from core.constants import SUPPORTED_EXTENSIONS


def get_file_extension(filename: str) -> str:
    """Get the lowercase file extension.

    Args:
        filename: Name of the file.

    Returns:
        Extension including the dot (e.g., '.csv').
    """
    _, ext = os.path.splitext(filename)
    return ext.lower()


def is_supported_extension(filename: str) -> bool:
    """Check if a file has a supported extension.

    Args:
        filename: Name of the file.

    Returns:
        True if the extension is in SUPPORTED_EXTENSIONS.
    """
    return get_file_extension(filename) in SUPPORTED_EXTENSIONS


def generate_timestamped_filename(original_name: str) -> str:
    """Generate a timestamped filename for archival.

    Example: 'dataset.csv' -> '20260825_143000_dataset.csv'

    Args:
        original_name: Original filename.

    Returns:
        Timestamped filename string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{original_name}"


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)
