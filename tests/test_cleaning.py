"""Unit tests for AI Cleaning Planner, Validator, and Executor."""

import json
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from core.schemas.cleaning import CleaningAction, CleaningPlan
from core.schemas.dataset_profile import DatasetProfile, ColumnProfile
from services.cleaning.planner import AICleaningPlanner
from services.cleaning.validator import CleaningPlanValidator
from services.cleaning.executor import CleaningExecutor


@pytest.fixture
def sample_profile() -> DatasetProfile:
    return DatasetProfile(
        filename="test.csv",
        row_count=100,
        column_count=3,
        duplicate_row_count=10,
        duplicate_row_percentage=10.0,
        memory_usage_mb=0.1,
        quality_score=70.0,
        columns=[
            ColumnProfile(
                name="age",
                dtype="Integer",
                pandas_dtype="int64",
                non_null_count=80,
                missing_count=20,
                missing_percentage=20.0,
                unique_count=40,
                unique_percentage=40.0,
                is_numeric=True,
            ),
            ColumnProfile(
                name="city",
                dtype="String/Text",
                pandas_dtype="object",
                non_null_count=90,
                missing_count=10,
                missing_percentage=10.0,
                unique_count=5,
                unique_percentage=5.0,
                is_categorical=True,
            ),
            ColumnProfile(
                name="const_col",
                dtype="String/Text",
                pandas_dtype="object",
                non_null_count=100,
                missing_count=0,
                missing_percentage=0.0,
                unique_count=1,
                unique_percentage=1.0,
                is_constant=True,
            ),
        ],
        constant_columns=["const_col"],
    )


class TestCleaningPlannerAndValidator:
    """Tests for AICleaningPlanner, CleaningPlanValidator, and CleaningExecutor."""

    def test_fallback_cleaning_plan(self, sample_profile: DatasetProfile) -> None:
        """When LLM is unconfigured, deterministic fallback cleaning plan is generated."""
        with patch("services.cleaning.planner.llm_config") as mock_cfg:
            mock_cfg.is_configured = False
            plan = AICleaningPlanner.plan(sample_profile)

            assert isinstance(plan, CleaningPlan)
            assert len(plan.actions) >= 3
            assert any(a.operation == "drop_duplicates" for a in plan.actions)
            assert any(a.operation == "remove_constant_column" for a in plan.actions)
            assert any(a.operation == "fill_missing_numeric" for a in plan.actions)

    def test_cleaning_plan_validator_approved_and_rejected(self) -> None:
        """Validator should approve valid actions and reject actions on nonexistent columns."""
        df = pd.DataFrame({
            "age": [25, np.nan, 30],
            "city": ["NY", "LA", None],
        })

        valid_action = CleaningAction(
            operation="fill_missing_numeric",
            column="age",
            method="median",
            reason="Fill age",
            evidence="20% missing",
        )
        invalid_col_action = CleaningAction(
            operation="fill_missing_numeric",
            column="nonexistent_col",
            method="median",
            reason="Fill missing",
            evidence="Evidence",
        )
        invalid_op_action = CleaningAction(
            operation="fill_missing_numeric",
            column="age",
            method="unsupported_method",
            reason="Fill missing",
            evidence="Evidence",
        )

        plan = CleaningPlan(
            summary="Test plan",
            actions=[valid_action, invalid_col_action, invalid_op_action],
            priority_order=[],
            warnings=[],
        )

        val_res = CleaningPlanValidator.validate_plan(plan, df)
        assert val_res.is_valid is False
        assert len(val_res.approved_actions) == 1
        assert len(val_res.rejected_actions) == 2

    def test_cleaning_executor_execution(self) -> None:
        """Executor should apply approved actions and report before/after diffs."""
        df = pd.DataFrame({
            "id": [1, 1, 2, 3],
            "val": [10.0, 10.0, np.nan, 30.0],
            "text": [" a ", " a ", " b ", " c "],
            "const": [1, 1, 1, 1],
        })

        actions = [
            CleaningAction(operation="drop_duplicates", reason="Remove dups", evidence="Evidence"),
            CleaningAction(operation="fill_missing_numeric", column="val", method="median", reason="Fill", evidence="Evidence"),
            CleaningAction(operation="strip_whitespace", column="text", reason="Strip", evidence="Evidence"),
            CleaningAction(operation="remove_constant_column", column="const", reason="Drop const", evidence="Evidence"),
        ]

        cleaned_df, result = CleaningExecutor.execute_plan(df, actions)

        assert len(cleaned_df) == 3  # 1 duplicate dropped
        assert cleaned_df["val"].isna().sum() == 0  # Missing imputed
        assert cleaned_df["text"].iloc[0] == "a"  # Stripped
        assert "const" not in cleaned_df.columns  # Constant dropped
        assert result.rows_before == 4
        assert result.rows_after == 3
        assert result.cols_before == 4
        assert result.cols_after == 3
