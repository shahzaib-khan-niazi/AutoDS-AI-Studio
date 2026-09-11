"""Centralized JSON Sanitization Utility for AutoDS AI Studio.

Ensures that any dataset profile, statistics, metadata, or complex nested Python object
is converted into a 100% JSON-serializable structure before transmission to LLMs,
OpenRouter, or standard json.dumps().

Recursively handles:
- datetime.datetime, datetime.date, datetime.time -> ISO-8601 strings
- pandas.Timestamp -> ISO-8601 string
- pandas.Timedelta -> ISO-8601 string / formatted duration
- pandas.NA, pandas.NaT -> None (JSON null)
- numpy datetime64, timedelta64 -> ISO-8601 string / duration string
- numpy numeric scalars (int64, int32, float64, float32, bool_, etc.) -> native int, float, bool
- non-finite floats (NaN, Inf, -Inf, np.nan, np.inf) -> None (JSON null)
- numpy.ndarray, pandas.Series, pandas.Index -> native Python lists
- pandas.DataFrame -> list of dict records
- Pydantic models, dataclasses, Enums -> serializable primitives
- dict, list, tuple, set -> recursively sanitized JSON structures
"""

import dataclasses
import datetime
from enum import Enum
import math
from typing import Any, Optional, Set
import numpy as np
import pandas as pd

from core.logging import logger


def sanitize_for_json(obj: Any, max_depth: int = 50, _seen_ids: Optional[Set[int]] = None) -> Any:
    """Recursively convert any Python/Pandas/NumPy structure into a JSON-safe object.

    Args:
        obj: The object to sanitize.
        max_depth: Maximum recursion depth to prevent infinite recursion on circular refs.
        _seen_ids: Internal set tracking visited object IDs to guard against circular references.

    Returns:
        A completely JSON-safe representation composed solely of standard Python primitives
        (dict, list, str, int, float, bool, None).
    """
    if max_depth <= 0:
        return str(obj)

    if _seen_ids is None:
        _seen_ids = set()

    # Fast path for None and primitive booleans (Note: bool is a subclass of int in Python)
    if obj is None:
        return None

    if isinstance(obj, bool):
        return obj

    # Pandas NA / NaT singleton handling
    if obj is pd.NA or obj is pd.NaT:
        return None

    # Handle datetime types -> ISO-8601 string
    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return obj.isoformat()

    if isinstance(obj, pd.Timedelta):
        if pd.isna(obj):
            return None
        return str(obj)

    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()

    if isinstance(obj, np.datetime64):
        if pd.isna(obj):
            return None
        try:
            return pd.Timestamp(obj).isoformat()
        except Exception:
            return str(obj)

    if isinstance(obj, np.timedelta64):
        if pd.isna(obj):
            return None
        try:
            return str(pd.Timedelta(obj))
        except Exception:
            return str(obj)

    # Handle NumPy booleans
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    # Handle NumPy integers and Python integers
    if isinstance(obj, (np.integer, int)):
        return int(obj)

    # Handle floats (Python float and NumPy floats)
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)

    # Handle Strings
    if isinstance(obj, str):
        return obj

    # Handle Bytes
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")

    # Handle Enums
    if isinstance(obj, Enum):
        return obj.value

    # Guard against circular references for container types
    obj_id = id(obj)
    if obj_id in _seen_ids:
        return f"<CircularRef: {type(obj).__name__}>"

    # Handle Pydantic BaseModel (v1 and v2 support)
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        _seen_ids.add(obj_id)
        try:
            return sanitize_for_json(obj.model_dump(), max_depth=max_depth - 1, _seen_ids=_seen_ids)
        finally:
            _seen_ids.discard(obj_id)
    elif hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        _seen_ids.add(obj_id)
        try:
            return sanitize_for_json(obj.dict(), max_depth=max_depth - 1, _seen_ids=_seen_ids)
        finally:
            _seen_ids.discard(obj_id)

    # Handle Dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        _seen_ids.add(obj_id)
        try:
            return sanitize_for_json(dataclasses.asdict(obj), max_depth=max_depth - 1, _seen_ids=_seen_ids)
        finally:
            _seen_ids.discard(obj_id)

    # Handle Pandas DataFrame -> list of dict records
    if isinstance(obj, pd.DataFrame):
        _seen_ids.add(obj_id)
        try:
            return [
                {
                    str(k): sanitize_for_json(v, max_depth=max_depth - 1, _seen_ids=_seen_ids)
                    for k, v in row.items()
                }
                for row in obj.to_dict(orient="records")
            ]
        finally:
            _seen_ids.discard(obj_id)

    # Handle Pandas Series / Index
    if isinstance(obj, (pd.Series, pd.Index)):
        _seen_ids.add(obj_id)
        try:
            return [
                sanitize_for_json(item, max_depth=max_depth - 1, _seen_ids=_seen_ids)
                for item in obj.tolist()
            ]
        finally:
            _seen_ids.discard(obj_id)

    # Handle NumPy ndarray
    if isinstance(obj, np.ndarray):
        _seen_ids.add(obj_id)
        try:
            return [
                sanitize_for_json(item, max_depth=max_depth - 1, _seen_ids=_seen_ids)
                for item in obj.tolist()
            ]
        finally:
            _seen_ids.discard(obj_id)

    # Handle Dictionaries / Mapping types
    if isinstance(obj, dict):
        _seen_ids.add(obj_id)
        try:
            return {
                str(k): sanitize_for_json(v, max_depth=max_depth - 1, _seen_ids=_seen_ids)
                for k, v in obj.items()
            }
        finally:
            _seen_ids.discard(obj_id)

    # Handle Iterables (list, tuple, set, frozenset)
    if isinstance(obj, (list, tuple, set, frozenset)):
        _seen_ids.add(obj_id)
        try:
            return [
                sanitize_for_json(item, max_depth=max_depth - 1, _seen_ids=_seen_ids)
                for item in obj
            ]
        finally:
            _seen_ids.discard(obj_id)

    # Fallback for arbitrary objects
    if hasattr(obj, "__dict__"):
        _seen_ids.add(obj_id)
        try:
            return {
                str(k): sanitize_for_json(v, max_depth=max_depth - 1, _seen_ids=_seen_ids)
                for k, v in vars(obj).items()
                if not str(k).startswith("_")
            }
        finally:
            _seen_ids.discard(obj_id)

    return str(obj)


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """Sanitize any object and serialize it safely to a JSON string without error.

    Args:
        obj: The object to serialize.
        **kwargs: Additional keyword arguments passed directly to json.dumps.

    Returns:
        Valid JSON formatted string.
    """
    import json

    sanitized = sanitize_for_json(obj)
    try:
        return json.dumps(sanitized, **kwargs)
    except Exception as exc:
        logger.error("safe_json_dumps fallback triggered: {}", str(exc))
        # Log problematic type if still somehow failing
        return json.dumps(str(sanitized), **kwargs)
