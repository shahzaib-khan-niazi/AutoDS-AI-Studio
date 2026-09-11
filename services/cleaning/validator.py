"""Cleaning Plan Validator for AutoDS AI Studio.

Ensures that AI-generated cleaning actions reference existing columns, utilize supported operations,
and carry required parameters before any execution is permitted.
"""

from typing import Any, Dict, List, Set, Tuple
import pandas as pd

from core.logging import logger
from core.schemas.cleaning import CleaningAction, CleaningPlan, CleaningValidationResult


class CleaningPlanValidator:
    """Validates CleaningPlan actions against actual DataFrame columns and safety constraints."""

    SUPPORTED_OPERATIONS: Set[str] = {
        "fill_missing_numeric",
        "fill_missing_categorical",
        "drop_missing_rows",
        "drop_missing_column",
        "drop_duplicates",
        "convert_dtype",
        "normalize_categories",
        "remove_constant_column",
        "cap_outliers",
        "strip_whitespace",
    }

    ALLOWED_NUMERIC_METHODS: Set[str] = {"median", "mean", "zero", "constant"}
    ALLOWED_CATEGORICAL_METHODS: Set[str] = {"mode", "missing_token", "constant"}
    ALLOWED_OUTLIER_METHODS: Set[str] = {"iqr", "zscore"}

    @classmethod
    def validate_plan(
        cls, plan: CleaningPlan, df: pd.DataFrame
    ) -> CleaningValidationResult:
        """Validate an AI cleaning plan against the current dataset.

        Args:
            plan: The CleaningPlan to inspect.
            df: Current DataFrame.

        Returns:
            CleaningValidationResult containing approved and rejected actions with reasons.
        """
        available_cols = set(df.columns)
        approved_actions: List[CleaningAction] = []
        rejected_actions: List[Dict[str, Any]] = []
        validation_errors: List[str] = []

        for idx, action in enumerate(plan.actions):
            is_action_valid, error_msg = cls._validate_single_action(action, available_cols)
            if is_action_valid:
                approved_actions.append(action)
            else:
                rejected_actions.append({
                    "action_index": idx,
                    "operation": action.operation,
                    "column": action.column,
                    "rejection_reason": error_msg,
                })
                validation_errors.append(f"Action #{idx+1} ({action.operation}): {error_msg}")

        overall_valid = len(approved_actions) > 0 and len(rejected_actions) == 0

        logger.info(
            "Cleaning plan validation: {} approved, {} rejected",
            len(approved_actions),
            len(rejected_actions),
        )

        return CleaningValidationResult(
            is_valid=overall_valid,
            approved_actions=approved_actions,
            rejected_actions=rejected_actions,
            validation_errors=validation_errors,
        )

    @classmethod
    def _validate_single_action(
        cls, action: CleaningAction, available_cols: Set[str]
    ) -> Tuple[bool, str]:
        """Validate a single CleaningAction."""
        if action.operation not in cls.SUPPORTED_OPERATIONS:
            return False, f"Unsupported operation '{action.operation}'"

        # Global dataset actions (don't require a specific column)
        if action.operation == "drop_duplicates":
            return True, ""

        # Column-targeted actions must specify an existing column
        if not action.column:
            return False, f"Operation '{action.operation}' requires a target column name."

        if action.column not in available_cols:
            return False, f"Column '{action.column}' does not exist in dataset."

        # Specific operation method checks
        if action.operation == "fill_missing_numeric":
            method = (action.method or "median").lower()
            if method not in cls.ALLOWED_NUMERIC_METHODS:
                return False, f"Invalid numeric imputation method '{method}'. Allowed: {cls.ALLOWED_NUMERIC_METHODS}"

        if action.operation == "fill_missing_categorical":
            method = (action.method or "mode").lower()
            if method not in cls.ALLOWED_CATEGORICAL_METHODS:
                return False, f"Invalid categorical imputation method '{method}'. Allowed: {cls.ALLOWED_CATEGORICAL_METHODS}"

        if action.operation == "cap_outliers":
            method = (action.method or "iqr").lower()
            if method not in cls.ALLOWED_OUTLIER_METHODS:
                return False, f"Invalid outlier capping method '{method}'. Allowed: {cls.ALLOWED_OUTLIER_METHODS}"

        return True, ""
