"""AI Planner for AutoDS AI Studio.

Constructs compact dataset profiles, prompts LLM for reasoning/planning,
validates the response, and converts AI actions to executable RepairActions.
"""

import json
from typing import Any, Optional
import pandas as pd

from core.logging import logger
from core.exceptions import AIServiceError, AIValidationError
from core.utils.json_sanitizer import sanitize_for_json, safe_json_dumps
from models.ai import AIPlan, AIAction
from models.repair import RepairAction, RepairOperation
from models.inspection import InspectionResult
from models.structure import StructureResult
from services.ai.client import AIClient
from services.ai.prompts import STRUCTURE_REASONING_PROMPT, REPAIR_PLANNING_PROMPT
from services.ai.validator import AIPlanValidator


class AIPlanner:
    """Orchestrates AI planning and reasoning for datasets."""

    @classmethod
    def build_profile(
        cls,
        df: pd.DataFrame,
        inspection: Optional[InspectionResult] = None,
        structure: Optional[StructureResult] = None,
    ) -> dict[str, Any]:
        """Build a compact, privacy-conscious dataset profile.

        NEVER includes the entire dataset. Only schema, sample rows, and summary stats.

        Args:
            df: Source DataFrame.
            inspection: Optional pre-computed inspection report.
            structure: Optional pre-computed structure report.

        Returns:
            Dictionary profile suitable for JSON prompt injection.
        """
        # Column types
        dtypes_map = {str(c): str(df[c].dtype) for c in df.columns}

        # Null counts
        null_counts = {str(c): int(df[c].isna().sum()) for c in df.columns}

        # Sample rows (first 3 rows converted safely to strings)
        sample_rows: list[dict[str, Any]] = []
        for _, row in df.head(3).iterrows():
            row_dict = {}
            for col in df.columns:
                val = row[col]
                row_dict[str(col)] = str(val) if pd.notna(val) else None
            sample_rows.append(row_dict)

        profile: dict[str, Any] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": [str(c) for c in df.columns],
            "dtypes": dtypes_map,
            "null_counts": null_counts,
            "duplicate_rows": int(df.duplicated().sum()),
            "sample_rows": sample_rows,
        }

        if inspection is not None:
            profile["quality_score"] = round(inspection.quality.quality_score, 2)
            profile["empty_rows"] = inspection.quality.empty_rows
            profile["empty_columns"] = inspection.quality.empty_columns
            profile["constant_columns"] = inspection.quality.constant_columns
            profile["mixed_type_columns"] = inspection.quality.mixed_type_columns

        if structure is not None:
            profile["detected_structure"] = structure.structure.value
            profile["structure_confidence"] = round(structure.confidence, 2)
            profile["structure_evidence"] = structure.evidence

        # Enrich with detailed column profiles from DatasetProfiler
        try:
            from services.profiler.service import DatasetProfiler
            det_profile = DatasetProfiler.profile(df)
            profile["column_profiles"] = [
                {
                    "column": col.name,
                    "pandas_dtype": col.pandas_dtype,
                    "inferred_semantic_type": col.inferred_semantic_type,
                    "null_count": col.missing_count,
                    "null_percentage": col.missing_percentage,
                    "unique_count": col.unique_count,
                    "sample_values": col.sample_values[:3] if col.sample_values else [],
                    "suspicious_values": col.suspicious_values[:3] if col.suspicious_values else [],
                    "format_patterns": col.format_patterns,
                    "possible_datatype": col.possible_datatype,
                    "datatype_confidence": col.datatype_confidence,
                    "datatype_evidence": col.datatype_evidence,
                }
                for col in det_profile.columns
            ]
        except Exception as e:
            logger.debug("AIPlanner column profile enrichment skipped: {}", str(e))

        return sanitize_for_json(profile)

    @classmethod
    def analyze_structure(
        cls,
        df: pd.DataFrame,
        inspection: Optional[InspectionResult] = None,
    ) -> AIPlan:
        """Use AI to analyze ambiguous structure.

        Args:
            df: Source DataFrame.
            inspection: Optional inspection result.

        Returns:
            Validated AIPlan.
        """
        profile = cls.build_profile(df, inspection=inspection)
        profile_json = safe_json_dumps(profile, indent=2)

        prompt = STRUCTURE_REASONING_PROMPT.format(profile_json=profile_json)

        logger.info("Requesting AI structure reasoning...")
        raw_output = AIClient.call_structured(prompt)

        available_cols = [str(c) for c in df.columns]
        return AIPlanValidator.validate_plan(raw_output, available_cols)

    @classmethod
    def generate_repair_plan(
        cls,
        df: pd.DataFrame,
        inspection: Optional[InspectionResult] = None,
        structure: Optional[StructureResult] = None,
    ) -> AIPlan:
        """Use AI to generate an intelligent, prioritized repair plan.

        Args:
            df: Source DataFrame.
            inspection: Optional inspection result.
            structure: Optional structure result.

        Returns:
            Validated AIPlan.
        """
        profile = cls.build_profile(df, inspection=inspection, structure=structure)
        profile_json = safe_json_dumps(profile, indent=2)

        prompt = REPAIR_PLANNING_PROMPT.format(profile_json=profile_json)

        logger.info("Requesting AI repair plan...")
        raw_output = AIClient.call_structured(prompt)

        available_cols = [str(c) for c in df.columns]
        return AIPlanValidator.validate_plan(raw_output, available_cols)

    @classmethod
    def convert_ai_actions_to_repair_actions(
        cls, plan: AIPlan
    ) -> list[RepairAction]:
        """Convert validated AI actions into executable RepairAction objects.

        Args:
            plan: Validated AIPlan.

        Returns:
            List of RepairAction ready for RepairService.repair().
        """
        repair_actions: list[RepairAction] = []

        for ai_act in plan.actions:
            try:
                op_enum = RepairOperation(ai_act.operation)
                repair_actions.append(
                    RepairAction(
                        operation=op_enum,
                        target=ai_act.target,
                        parameters=ai_act.parameters,
                        reason=ai_act.reason,
                        confidence=plan.confidence,
                        safe=True,
                    )
                )
            except ValueError:
                logger.warning("Skipped unmapped operation: {}", ai_act.operation)

        return repair_actions
