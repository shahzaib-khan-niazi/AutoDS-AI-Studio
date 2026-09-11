"""Tests for core.utils.json_sanitizer (JSON Sanitization Layer)."""

import datetime
from decimal import Decimal
import json
import math
import numpy as np
import pandas as pd
import pytest

from core.utils.json_sanitizer import sanitize_for_json, safe_json_dumps
from services.ai.planner import AIPlanner
from models.inspection import InspectionResult, QualityReport
from models.structure import StructureResult, StructureType


class TestJSONSanitizerPrimitives:
    """Test individual data types and their sanitized representations."""

    def test_datetime_objects(self):
        dt = datetime.datetime(2026, 7, 31, 15, 30, 0)
        sanitized = sanitize_for_json(dt)
        assert sanitized == "2026-07-31T15:30:00"
        assert json.dumps(sanitized) == '"2026-07-31T15:30:00"'

    def test_date_objects(self):
        d = datetime.date(2026, 7, 31)
        sanitized = sanitize_for_json(d)
        assert sanitized == "2026-07-31"
        assert json.dumps(sanitized) == '"2026-07-31"'

    def test_time_objects(self):
        t = datetime.time(15, 30, 45)
        sanitized = sanitize_for_json(t)
        assert sanitized == "15:30:45"
        assert json.dumps(sanitized) == '"15:30:45"'

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2026-07-31 15:30:00")
        sanitized = sanitize_for_json(ts)
        assert sanitized == "2026-07-31T15:30:00"
        assert json.dumps(sanitized) == '"2026-07-31T15:30:00"'

    def test_pandas_nat_and_na(self):
        assert sanitize_for_json(pd.NaT) is None
        assert sanitize_for_json(pd.NA) is None
        assert json.dumps(sanitize_for_json({"nat": pd.NaT, "na": pd.NA})) == '{"nat": null, "na": null}'

    def test_pandas_timedelta(self):
        td = pd.Timedelta(days=5, hours=3, minutes=12)
        sanitized = sanitize_for_json(td)
        assert isinstance(sanitized, str)
        assert "5 days" in sanitized or "P" in sanitized
        assert json.dumps(sanitized)

    def test_numpy_integers(self):
        i64 = np.int64(42)
        i32 = np.int32(100)
        assert sanitize_for_json(i64) == 42
        assert isinstance(sanitize_for_json(i64), int)
        assert sanitize_for_json(i32) == 100
        assert isinstance(sanitize_for_json(i32), int)
        assert json.dumps(sanitize_for_json({"i64": i64, "i32": i32})) == '{"i64": 42, "i32": 100}'

    def test_numpy_floats_and_non_finites(self):
        f64 = np.float64(3.14159)
        assert sanitize_for_json(f64) == pytest.approx(3.14159)
        assert isinstance(sanitize_for_json(f64), float)

        # Non-finites become None (JSON null)
        assert sanitize_for_json(np.nan) is None
        assert sanitize_for_json(float("nan")) is None
        assert sanitize_for_json(np.inf) is None
        assert sanitize_for_json(float("inf")) is None
        assert sanitize_for_json(-np.inf) is None
        assert sanitize_for_json(float("-inf")) is None

        payload = {
            "valid_float": np.float32(2.5),
            "nan_val": np.nan,
            "inf_val": np.inf,
            "neg_inf_val": -np.inf,
        }
        dumped = json.dumps(sanitize_for_json(payload))
        assert json.loads(dumped) == {
            "valid_float": 2.5,
            "nan_val": None,
            "inf_val": None,
            "neg_inf_val": None,
        }

    def test_numpy_booleans(self):
        b_true = np.bool_(True)
        b_false = np.bool_(False)
        assert sanitize_for_json(b_true) is True
        assert sanitize_for_json(b_false) is False
        assert isinstance(sanitize_for_json(b_true), bool)
        assert json.dumps(sanitize_for_json([b_true, b_false])) == "[true, false]"

    def test_numpy_ndarray(self):
        arr = np.array([1, 2, 3, np.nan, 5])
        sanitized = sanitize_for_json(arr)
        assert sanitized == [1.0, 2.0, 3.0, None, 5.0]
        assert json.dumps(sanitized) == "[1.0, 2.0, 3.0, null, 5.0]"

    def test_numpy_datetime64(self):
        dt64 = np.datetime64("2026-07-31T15:30:00")
        sanitized = sanitize_for_json(dt64)
        assert "2026-07-31" in sanitized
        assert json.dumps(sanitized)


class TestJSONSanitizerNestedStructures:
    """Test complex nested dictionaries, lists, mixed types, and circular references."""

    def test_nested_dictionaries(self):
        nested = {
            "level1": {
                "level2": {
                    "timestamp": pd.Timestamp("2026-01-01 12:00:00"),
                    "count": np.int64(99),
                    "missing": np.nan,
                    "active": np.bool_(True),
                }
            }
        }
        sanitized = sanitize_for_json(nested)
        assert json.dumps(sanitized)
        parsed = json.loads(json.dumps(sanitized))
        assert parsed["level1"]["level2"]["timestamp"] == "2026-01-01T12:00:00"
        assert parsed["level1"]["level2"]["count"] == 99
        assert parsed["level1"]["level2"]["missing"] is None
        assert parsed["level1"]["level2"]["active"] is True

    def test_mixed_list_and_tuples(self):
        mixed = [
            datetime.datetime(2026, 5, 20),
            (np.int32(1), np.float64(2.5), np.nan),
            {"set_data": {pd.Timestamp("2026-01-01"), "text", np.int64(7)}},
        ]
        sanitized = sanitize_for_json(mixed)
        dumped = json.dumps(sanitized)
        assert dumped
        parsed = json.loads(dumped)
        assert isinstance(parsed, list)
        assert parsed[0] == "2026-05-20T00:00:00"

    def test_circular_reference_guard(self):
        d1 = {"name": "node1"}
        d2 = {"name": "node2", "parent": d1}
        d1["child"] = d2

        # Must not throw RecursionError
        sanitized = sanitize_for_json(d1)
        dumped = json.dumps(sanitized)
        assert "node1" in dumped
        assert "<CircularRef" in dumped

    def test_safe_json_dumps_helper(self):
        payload = {
            "date": pd.Timestamp("2026-07-31"),
            "amount": np.float64(100.5),
            "flag": np.bool_(True),
        }
        dumped_str = safe_json_dumps(payload, indent=2)
        assert isinstance(dumped_str, str)
        parsed = json.loads(dumped_str)
        assert parsed["date"] == "2026-07-31T00:00:00"
        assert parsed["amount"] == 100.5
        assert parsed["flag"] is True


class TestDatasetProfileSanitization:
    """Test end-to-end dataset profile generation and AI Planner safety."""

    def test_profile_with_datetime_column_does_not_fail_json_dumps(self):
        df = pd.DataFrame({
            "order_id": [1, 2, 3],
            "order_date": pd.to_datetime(["2026-01-01", "2026-02-15", "2026-03-30"]),
            "amount": [10.5, 20.0, np.nan],
            "is_active": [True, False, True],
        })

        # Generate profile via AIPlanner
        profile = AIPlanner.build_profile(df)

        # Verify json.dumps succeeds without exception
        profile_json = json.dumps(profile, indent=2)
        assert profile_json
        parsed = json.loads(profile_json)

        assert parsed["row_count"] == 3
        assert parsed["column_count"] == 4
        assert "column_profiles" in parsed

        # Check that datetime sample values are strings
        date_col = next((c for c in parsed["column_profiles"] if c["column"] == "order_date"), None)
        assert date_col is not None
        for sample in date_col["sample_values"]:
            assert isinstance(sample, str)

    def test_dataframe_remains_unmutated(self):
        original_dates = pd.to_datetime(["2026-01-01", "2026-02-15", "2026-03-30"])
        df = pd.DataFrame({
            "order_id": [1, 2, 3],
            "order_date": original_dates.copy(),
            "amount": [10.5, 20.0, 30.0],
        })

        original_dtype = df["order_date"].dtype

        # Profile the dataset
        profile = AIPlanner.build_profile(df)
        assert json.dumps(profile)

        # Verify original dataframe datatypes and values are preserved
        assert df["order_date"].dtype == original_dtype
        assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
        assert df["order_date"].iloc[0] == pd.Timestamp("2026-01-01")
