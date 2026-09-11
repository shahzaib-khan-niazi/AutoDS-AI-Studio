"""Logging configuration for AutoDS AI Studio.

Provides a single configured logger instance for the entire application.
Import `logger` from this module — do not create loggers elsewhere.
"""

import sys
from loguru import logger

# Remove default handler
logger.remove()

# Console handler — human-readable, INFO level
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# File handler — detailed, DEBUG level
logger.add(
    "data/reports/autods.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
)
