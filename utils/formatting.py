"""Formatting utility functions for display."""


def format_bytes(num_bytes: int) -> str:
    """Format bytes into a human-readable string.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Formatted string like '1.5 MB'.
    """
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_number(value: int) -> str:
    """Format a number with comma separators.

    Args:
        value: Integer value.

    Returns:
        Formatted string like '1,000,000'.
    """
    return f"{value:,}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage string.

    Args:
        value: Float value (0.0 to 1.0).
        decimals: Number of decimal places.

    Returns:
        Formatted string like '95.5%'.
    """
    return f"{value * 100:.{decimals}f}%"


def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate a string with ellipsis if too long.

    Args:
        text: Input string.
        max_length: Maximum length before truncation.

    Returns:
        Truncated string with '...' appended if needed.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
