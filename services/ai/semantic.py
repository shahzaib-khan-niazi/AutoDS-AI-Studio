"""AI Semantic Categorical Normalizer module.

Provides structured AI reasoning to resolve ambiguous semantic categories, acronyms,
and candidate equivalence groups in categorical columns.

Strict architecture:
DATASET -> PROFILE -> DETECT -> CANDIDATES -> AI ANALYSIS -> VALIDATOR -> CONFIDENCE -> USER APPROVAL -> PYTHON EXECUTION
"""

import json
from typing import Any, Optional
import pandas as pd

from core.logging import logger
from core.llm.config import llm_config
from core.utils.json_sanitizer import sanitize_for_json, safe_json_dumps
from models.repair import IssueTaxonomy, SemanticCandidateGroup
from services.ai.client import AIClient
from services.repair.standardize import clean_unicode_and_whitespace, get_proposed_standardizations, is_categorical_column


SEMANTIC_NORMALIZATION_PROMPT = """You are an expert Data Science and Semantic Entity Normalization Specialist.
Analyze the following categorical column values and proposed candidate equivalence groups.
Determine whether the candidate values truly represent the SAME underlying semantic concept or if they are genuinely DIFFERENT entities that must remain separate.

Column: "{column_name}"
Column Data Type: "{dtype}"
Total Rows: {total_rows}
Unique Value Frequencies:
{value_frequencies_json}

Potential Equivalence Candidates to Evaluate:
{candidates_json}

RULES:
1. ONLY group values together if they clearly represent the EXACT same underlying entity or concept (e.g. abbreviations of the full name, spelling variants, formatting differences).
2. NEVER group values if they represent different entities (e.g. "Apple" vs "Apple Store", or "Standard" vs "Standard Plus").
3. For each group, choose the most descriptive, complete, and standard canonical representation.
4. Provide a clear reason and calibrated confidence score between 0.0 and 1.0.
5. Return ONLY a valid JSON object matching this schema:
{{
    "column": "{column_name}",
    "issue_type": "semantic_inconsistency",
    "candidate_groups": [
        {{
            "canonical_value": "<standard chosen representation>",
            "variants": ["<variant_1>", "<variant_2>"],
            "reason": "<clear explanation why these are equivalent>",
            "confidence": <float between 0.0 and 1.0>,
            "action": "normalize"
        }}
    ]
}}
"""


class AISemanticNormalizer:
    """Service to evaluate and validate semantic equivalence in categorical columns via AI."""

    @classmethod
    def analyze_column_semantics(
        cls, df: pd.DataFrame, column: str
    ) -> list[SemanticCandidateGroup]:
        """Analyze a categorical column with AI reasoning to identify equivalent value groups.

        Args:
            df: Source DataFrame.
            column: Categorical column to inspect.

        Returns:
            List of validated SemanticCandidateGroup objects.
        """
        if column not in df.columns or not is_categorical_column(df[column]):
            return []

        series = df[column].dropna()
        if len(series) == 0:
            return []

        val_counts = series.astype(str).value_counts().to_dict()
        if len(val_counts) <= 1:
            return []

        # Get initial algorithmic candidate proposals
        proposals = get_proposed_standardizations(df, columns=[column])
        candidate_summary = [
            {
                "original_value": p["original_value"],
                "normalized_value": p["normalized_value"],
                "method": p["method"],
                "reason": p["reason"],
                "confidence": p["confidence"],
            }
            for p in proposals
        ]

        if not llm_config.is_configured:
            # Fallback to algorithmic candidate groups if AI is not configured
            return cls._convert_proposals_to_groups(column, proposals, val_counts)

        try:
            logger.info("Requesting AI semantic reasoning for column '{}'", column)
            prompt = SEMANTIC_NORMALIZATION_PROMPT.format(
                column_name=column,
                dtype=str(series.dtype),
                total_rows=len(series),
                value_frequencies_json=safe_json_dumps(sanitize_for_json(val_counts), indent=2),
                candidates_json=safe_json_dumps(sanitize_for_json(candidate_summary), indent=2),
            )

            raw_output = AIClient.call_structured(prompt)
            validated_groups = cls._validate_ai_groups(df, column, raw_output, val_counts)
            if validated_groups:
                return validated_groups
        except Exception as e:
            logger.warning("AI semantic analysis failed ({}), falling back to algorithmic: {}", type(e).__name__, str(e))

        return cls._convert_proposals_to_groups(column, proposals, val_counts)

    @classmethod
    def _validate_ai_groups(
        cls,
        df: pd.DataFrame,
        column: str,
        raw_output: dict[str, Any],
        val_counts: dict[str, int],
    ) -> list[SemanticCandidateGroup]:
        """Strict Python validation of AI-proposed semantic candidate groups."""
        validated: list[SemanticCandidateGroup] = []
        raw_groups = raw_output.get("candidate_groups", [])

        if not isinstance(raw_groups, list):
            return []

        unique_vals_in_col = set(val_counts.keys())

        for grp in raw_groups:
            if not isinstance(grp, dict):
                continue

            canonical = str(grp.get("canonical_value", "")).strip()
            variants = [str(v).strip() for v in grp.get("variants", []) if str(v).strip()]
            reason = str(grp.get("reason", "AI-identified semantic equivalence"))
            
            try:
                conf = float(grp.get("confidence", 0.90))
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                conf = 0.85

            if not canonical or not variants:
                continue

            # Ensure all variants actually exist in the column
            valid_variants = [v for v in variants if v in unique_vals_in_col and v != canonical]
            if not valid_variants:
                continue

            total_affected = sum(val_counts.get(v, 0) for v in valid_variants)

            validated.append(
                SemanticCandidateGroup(
                    column=column,
                    canonical_value=canonical,
                    variants=valid_variants,
                    reason=reason,
                    confidence=conf,
                    issue_type=IssueTaxonomy.SEMANTIC,
                    method="ai_semantic_reasoning",
                    affected_rows=total_affected,
                )
            )

        return validated

    @classmethod
    def _convert_proposals_to_groups(
        cls, column: str, proposals: list[dict[str, Any]], val_counts: dict[str, int]
    ) -> list[SemanticCandidateGroup]:
        """Convert deterministic proposals into SemanticCandidateGroup representations."""
        # Group by canonical value
        groups_by_canonical: dict[str, dict[str, Any]] = {}

        for p in proposals:
            canon = p["normalized_value"]
            orig = p["original_value"]
            if canon is None:
                continue

            if canon not in groups_by_canonical:
                groups_by_canonical[canon] = {
                    "variants": [],
                    "reason": p["reason"],
                    "confidence": p["confidence"],
                    "method": p["method"],
                    "issue_type": p.get("issue_type", IssueTaxonomy.SEMANTIC.value),
                    "is_ambiguous": p.get("is_ambiguous", False),
                    "possible_meanings": p.get("possible_meanings", []),
                }

            groups_by_canonical[canon]["variants"].append(orig)
            groups_by_canonical[canon]["confidence"] = min(groups_by_canonical[canon]["confidence"], p["confidence"])

        result: list[SemanticCandidateGroup] = []
        for canon, data in groups_by_canonical.items():
            affected = sum(val_counts.get(v, 0) for v in data["variants"])
            
            try:
                tax_enum = IssueTaxonomy(data["issue_type"])
            except ValueError:
                tax_enum = IssueTaxonomy.SEMANTIC

            result.append(
                SemanticCandidateGroup(
                    column=column,
                    canonical_value=canon,
                    variants=data["variants"],
                    reason=data["reason"],
                    confidence=data["confidence"],
                    issue_type=tax_enum,
                    method=data["method"],
                    is_ambiguous=data.get("is_ambiguous", False),
                    possible_meanings=data.get("possible_meanings", []),
                    affected_rows=affected,
                )
            )

        return result
