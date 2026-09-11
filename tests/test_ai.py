"""Tests for AI planner, validator, and schema conversion."""

import numpy as np
import pandas as pd
import pytest

from models.ai import AIPlan, AIAction
from models.repair import RepairOperation
from services.ai.planner import AIPlanner
from services.ai.validator import AIPlanValidator


def test_ai_build_profile() -> None:
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "score": [95, 80, 75],
    })

    profile = AIPlanner.build_profile(df)
    assert profile["row_count"] == 3
    assert profile["column_count"] == 2
    assert "name" in profile["columns"]
    assert "score" in profile["columns"]
    assert len(profile["sample_rows"]) == 3


def test_ai_plan_validator_success() -> None:
    raw_output = {
        "decision": "WIDE_TABLE",
        "confidence": 0.92,
        "reasoning_summary": "The dataset contains categories as columns.",
        "actions": [
            {
                "operation": "unpivot",
                "target": ["col1", "col2"],
                "parameters": {"var_name": "Month", "value_name": "Sales"},
                "reason": "Convert horizontal categories to rows",
            }
        ],
        "warnings": ["Ensure ID column is preserved"],
    }

    available_cols = ["id", "col1", "col2"]
    plan = AIPlanValidator.validate_plan(raw_output, available_cols)

    assert plan.decision == "WIDE_TABLE"
    assert plan.confidence == 0.92
    assert len(plan.actions) == 1
    assert plan.actions[0].operation == "unpivot"
    assert plan.actions[0].target == ["col1", "col2"]


def test_ai_plan_validator_rejects_unallowed_operation() -> None:
    raw_output = {
        "decision": "MALICIOUS_PLAN",
        "confidence": 0.99,
        "reasoning_summary": "Attempting arbitrary code execution",
        "actions": [
            {
                "operation": "eval_arbitrary_python_code",
                "target": ["col1"],
                "parameters": {"code": "import os; os.system('rm -rf')"},
                "reason": "Unsafe",
            },
            {
                "operation": "remove_duplicate_rows",
                "target": [],
                "parameters": {},
                "reason": "Safe op",
            }
        ],
    }

    plan = AIPlanValidator.validate_plan(raw_output, ["col1"])
    # The unallowed operation must be rejected
    assert len(plan.actions) == 1
    assert plan.actions[0].operation == "remove_duplicate_rows"
    assert any("Rejected unknown operation" in w for w in plan.warnings)


def test_convert_ai_actions_to_repair_actions() -> None:
    plan = AIPlan(
        decision="CLEANING_PLAN",
        confidence=0.9,
        reasoning_summary="Clean duplicates and empty columns",
        actions=[
            AIAction(
                operation="remove_duplicate_rows",
                reason="Deduplicate",
            ),
            AIAction(
                operation="drop_empty_columns",
                target=["empty_col"],
                reason="Drop nulls",
            ),
        ],
    )

    repair_actions = AIPlanner.convert_ai_actions_to_repair_actions(plan)
    assert len(repair_actions) == 2
    assert repair_actions[0].operation == RepairOperation.REMOVE_DUPLICATE_ROWS
    assert repair_actions[1].operation == RepairOperation.DROP_EMPTY_COLUMNS


def test_ai_analyze_structure_with_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({
        "order_id": [101, 102, 103],
        "created_at": pd.to_datetime(["2026-07-01 10:00:00", "2026-07-02 11:30:00", "2026-07-03 14:15:00"]),
        "revenue": [150.75, 300.0, np.nan],
    })

    mock_response = {
        "decision": "NORMAL_TABLE",
        "confidence": 0.95,
        "reasoning_summary": "Standard relational table with datetime timestamp.",
        "actions": [],
        "warnings": [],
    }

    from services.ai.client import AIClient
    monkeypatch.setattr(AIClient, "call_structured", lambda prompt, **kwargs: mock_response)

    # Must complete without raising TypeError: Object of type datetime is not JSON serializable
    plan = AIPlanner.analyze_structure(df)
    assert plan.decision == "NORMAL_TABLE"
    assert plan.confidence == 0.95


def test_ai_generate_repair_plan_with_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({
        "order_id": [101, 102, 103],
        "created_at": pd.to_datetime(["2026-07-01 10:00:00", "2026-07-02 11:30:00", "2026-07-03 14:15:00"]),
        "revenue": [150.75, 300.0, np.nan],
    })

    mock_response = {
        "decision": "REPAIR_PLAN",
        "confidence": 0.90,
        "reasoning_summary": "Repair plan with date column.",
        "actions": [
            {
                "operation": "fill_missing",
                "target": ["revenue"],
                "parameters": {"method": "median"},
                "reason": "Fill revenue missing",
            }
        ],
        "warnings": [],
    }

    from services.ai.client import AIClient
    monkeypatch.setattr(AIClient, "call_structured", lambda prompt, **kwargs: mock_response)

    # Must complete without raising TypeError
    plan = AIPlanner.generate_repair_plan(df)
    assert plan.decision == "REPAIR_PLAN"
    assert len(plan.actions) == 1
