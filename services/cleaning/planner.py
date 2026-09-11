"""AI Cleaning Planner for AutoDS AI Studio.

Generates structured, safe data cleaning plans from a DatasetProfile using LLMService.
No raw DataFrame data is transmitted; only compact structural metadata and quality facts.
"""

import json
from typing import Any, Dict, List, Optional
import pandas as pd

from core.logging import logger
from core.llm.config import llm_config
from core.llm.service import LLMService
from core.llm.exceptions import LLMError
from core.utils.json_sanitizer import sanitize_for_json, safe_json_dumps
from core.schemas.dataset_profile import DatasetProfile
from core.schemas.cleaning import CleaningAction, CleaningPlan


class AICleaningPlanner:
    """Generates structured cleaning plans using the LLM orchestration layer."""

    _SYSTEM_PROMPT = """You are the AI Data Cleaning Planner for AutoDS AI Studio.
Your role is to analyze a deterministic DatasetProfile JSON and produce an intelligent, safe, prioritized cleaning plan.

CONSTRAINTS & RULES:
1. ONLY propose operations from this allowed list:
   - "fill_missing_numeric" (methods: "median", "mean", "zero", "constant")
   - "fill_missing_categorical" (methods: "mode", "missing_token", "constant")
   - "drop_missing_rows" (for rows with critical missing keys)
   - "drop_missing_column" (only when missingness > 60%)
   - "drop_duplicates" (removes exact duplicate rows)
   - "convert_dtype" (e.g. target_dtype: "datetime", "numeric", "category")
   - "normalize_categories" (strips whitespace, standardizes case)
   - "remove_constant_column" (columns with only 1 unique value)
   - "cap_outliers" (methods: "iqr", "zscore")
   - "strip_whitespace" (string/text cleanup)
2. Every action MUST include concrete evidence from the profile.
3. Mark risk appropriately: "low", "medium", or "high" (destructive drops must be "high").
4. Priority must be: "critical", "high", "medium", or "low".

Output valid JSON matching this schema:
{
  "summary": "High level summary of cleaning strategy",
  "actions": [
    {
      "operation": "fill_missing_numeric",
      "column": "Age",
      "method": "median",
      "parameters": {},
      "reason": "Impute missing ages to preserve rows without skewing distribution.",
      "evidence": "Age column has 143 missing values (14.3%).",
      "risk": "low",
      "priority": "high"
    }
  ],
  "priority_order": ["Action 1", "Action 2"],
  "warnings": ["Any warnings about data loss or irreversible changes"]
}
"""

    @classmethod
    def plan(cls, profile: DatasetProfile) -> CleaningPlan:
        """Generate structured cleaning plan from a validated DatasetProfile.

        Args:
            profile: Deterministic DatasetProfile.

        Returns:
            Structured CleaningPlan.
        """
        if not llm_config.is_configured:
            logger.warning("AICleaningPlanner called without configured API key; using deterministic fallback")
            return cls._fallback_plan(profile, reason="OPENROUTER_API_KEY not set")

        compact_profile = sanitize_for_json(cls._prepare_compact_profile(profile))
        user_prompt = f"Create a structured cleaning plan for this dataset profile:\n{safe_json_dumps(compact_profile, indent=2)}"

        messages = [
            {"role": "system", "content": cls._SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        schema_hint = {
            "name": "CleaningPlan",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "enum": [
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
                                    ],
                                },
                                "column": {"type": ["string", "null"]},
                                "method": {"type": ["string", "null"]},
                                "parameters": {"type": "object"},
                                "reason": {"type": "string"},
                                "evidence": {"type": "string"},
                                "risk": {"type": "string", "enum": ["low", "medium", "high"]},
                                "priority": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                            },
                            "required": ["operation", "reason", "evidence", "risk", "priority"],
                        },
                    },
                    "priority_order": {"type": "array", "items": {"type": "string"}},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "actions", "priority_order", "warnings"],
            },
        }

        try:
            svc = LLMService()
            result = svc.generate_structured(messages, response_schema=schema_hint)
            raw = json.loads(result.content)

            actions: List[CleaningAction] = []
            for item in raw.get("actions", []):
                actions.append(
                    CleaningAction(
                        operation=item.get("operation"),
                        column=item.get("column"),
                        method=item.get("method"),
                        parameters=item.get("parameters", {}),
                        reason=item.get("reason", "Standard data hygiene."),
                        evidence=item.get("evidence", "Identified in dataset profile."),
                        risk=item.get("risk", "low").lower(),
                        priority=item.get("priority", "medium").lower(),
                    )
                )

            return CleaningPlan(
                summary=raw.get("summary", "Cleaning plan generated."),
                actions=actions,
                priority_order=[str(p) for p in raw.get("priority_order", [])],
                warnings=[str(w) for w in raw.get("warnings", [])],
                confidence_score=0.90 if actions else 0.75,
                confidence_level="High" if actions else "Medium",
            )

        except LLMError as e:
            logger.error("LLM error during cleaning planning: {}", e.message)
            return cls._fallback_plan(profile, reason=f"AI service error: {e.message}")
        except Exception as e:
            logger.exception("Unexpected error in AICleaningPlanner: {}", str(e))
            return cls._fallback_plan(profile, reason=f"Unexpected error: {str(e)}")

    @classmethod
    def _prepare_compact_profile(cls, profile: DatasetProfile) -> Dict[str, Any]:
        """Convert DatasetProfile into compact JSON representation for the LLM."""
        return {
            "filename": profile.filename,
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "duplicate_rows": f"{profile.duplicate_row_count} ({profile.duplicate_row_percentage:.1f}%)",
            "quality_score": f"{profile.quality_score:.1f} / 100",
            "missing_columns": [
                {
                    "name": col.name,
                    "dtype": col.dtype,
                    "missing_count": col.missing_count,
                    "missing_percentage": f"{col.missing_percentage:.1f}%",
                    "is_numeric": col.is_numeric,
                }
                for col in profile.columns
                if col.missing_count > 0
            ],
            "constant_columns": profile.constant_columns,
            "high_cardinality_columns": profile.high_cardinality_columns,
            "possible_id_columns": profile.possible_id_columns,
        }

    @classmethod
    def _fallback_plan(cls, profile: DatasetProfile, reason: str) -> CleaningPlan:
        """Generate a deterministic fallback cleaning plan when AI is unconfigured or unavailable."""
        actions: List[CleaningAction] = []

        # 1. Duplicates
        if profile.duplicate_row_count > 0:
            actions.append(
                CleaningAction(
                    operation="drop_duplicates",
                    reason="Eliminate redundant observations to prevent training bias.",
                    evidence=f"Dataset contains {profile.duplicate_row_count} duplicate rows ({profile.duplicate_row_percentage:.1f}%).",
                    risk="low",
                    priority="high",
                )
            )

        # 2. Constant columns
        for col_name in profile.constant_columns:
            actions.append(
                CleaningAction(
                    operation="remove_constant_column",
                    column=col_name,
                    reason="Constant columns carry zero variance/information.",
                    evidence=f"Column '{col_name}' has <= 1 unique value.",
                    risk="low",
                    priority="medium",
                )
            )

        # 3. Missing values
        for col in profile.columns:
            if col.missing_count > 0:
                if col.missing_percentage > 60.0:
                    actions.append(
                        CleaningAction(
                            operation="drop_missing_column",
                            column=col.name,
                            reason="Column has severe missingness (>60%).",
                            evidence=f"Column '{col.name}' is {col.missing_percentage:.1f}% missing.",
                            risk="high",
                            priority="medium",
                        )
                    )
                elif col.is_numeric:
                    actions.append(
                        CleaningAction(
                            operation="fill_missing_numeric",
                            column=col.name,
                            method="median",
                            reason="Impute numeric missing values using median to resist outliers.",
                            evidence=f"Column '{col.name}' has {col.missing_count} missing values ({col.missing_percentage:.1f}%).",
                            risk="low",
                            priority="high",
                        )
                    )
                else:
                    actions.append(
                        CleaningAction(
                            operation="fill_missing_categorical",
                            column=col.name,
                            method="mode",
                            reason="Impute categorical missing values using mode/most frequent category.",
                            evidence=f"Column '{col.name}' has {col.missing_count} missing values ({col.missing_percentage:.1f}%).",
                            risk="low",
                            priority="medium",
                        )
                    )

        return CleaningPlan(
            summary=f"Rule-based deterministic cleaning plan ({len(actions)} actions).",
            actions=actions,
            priority_order=[f"{a.operation} on {a.column or 'dataset'}" for a in actions],
            warnings=[f"Deterministic fallback used ({reason})"],
            confidence_score=0.80,
            confidence_level="Medium",
        )
