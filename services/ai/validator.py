"""AI Plan Validator.

Ensures that AI outputs adhere strictly to expected schemas, allowed operations,
and valid target columns before any Python execution is allowed.
"""

from typing import Any
from core.exceptions import AIValidationError
from core.logging import logger
from models.ai import AIPlan, AIAction
from models.repair import RepairOperation


# Set of allowed operation string values
ALLOWED_OPERATIONS = {op.value for op in RepairOperation}


class AIPlanValidator:
    """Validates structured plans returned by AI."""

    @classmethod
    def validate_plan(
        cls,
        raw_output: dict[str, Any],
        available_columns: list[str],
    ) -> AIPlan:
        """Validate and construct an AIPlan from raw AI output.

        Args:
            raw_output: Parsed JSON dict from LLM.
            available_columns: List of valid column names in the current dataset.

        Returns:
            Validated AIPlan instance.

        Raises:
            AIValidationError: If the output fails validation.
        """
        if not isinstance(raw_output, dict):
            raise AIValidationError(
                message="AI output is not a JSON object.",
                details=f"Received: {type(raw_output).__name__}",
            )

        decision = str(raw_output.get("decision", "No decision provided"))
        raw_confidence = raw_output.get("confidence", 0.5)

        try:
            confidence = float(raw_confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        reasoning_summary = str(raw_output.get("reasoning_summary", ""))
        raw_actions = raw_output.get("actions", [])
        raw_warnings = raw_output.get("warnings", [])

        warnings: list[str] = [str(w) for w in raw_warnings if isinstance(w, str)]
        valid_actions: list[AIAction] = []

        if isinstance(raw_actions, list):
            for i, raw_act in enumerate(raw_actions):
                if not isinstance(raw_act, dict):
                    warnings.append(f"Skipped non-object action at index {i}")
                    continue

                operation = str(raw_act.get("operation", "")).strip().lower()
                if operation not in ALLOWED_OPERATIONS:
                    warnings.append(f"Rejected unknown operation: '{operation}'")
                    logger.warning("AI recommended unallowed operation: {}", operation)
                    continue

                target = raw_act.get("target", [])
                if isinstance(target, str):
                    target = [target]
                elif not isinstance(target, list):
                    target = []

                # Validate target columns
                valid_target = [str(col) for col in target if str(col) in available_columns]
                invalid_cols = [str(col) for col in target if str(col) not in available_columns]
                if invalid_cols:
                    warnings.append(
                        f"Ignored non-existent target column(s) in action '{operation}': {', '.join(invalid_cols)}"
                    )

                parameters = raw_act.get("parameters", {})
                if not isinstance(parameters, dict):
                    parameters = {}

                reason = str(raw_act.get("reason", ""))

                valid_actions.append(
                    AIAction(
                        operation=operation,
                        target=valid_target,
                        parameters=parameters,
                        reason=reason,
                    )
                )

        plan = AIPlan(
            decision=decision,
            confidence=confidence,
            reasoning_summary=reasoning_summary,
            actions=valid_actions,
            warnings=warnings,
        )

        logger.info(
            "AI Plan validated successfully: {} actions, confidence {:.0%}",
            len(plan.actions),
            plan.confidence,
        )

        return plan
