"""AI Column Name Suggester module.

Provides AI-driven suggestions for clearer, standard column names based on column profiles
(datatype, sample values, null counts, distinct counts).
Follows the architecture:
DATA PROFILE -> AI SUGGESTION -> VALIDATION -> USER APPROVAL -> PYTHON EXECUTION -> VERIFICATION
"""

import json
import re
from typing import Any, Optional
import pandas as pd

from core.logging import logger
from core.llm.config import llm_config
from core.utils.json_sanitizer import sanitize_for_json, safe_json_dumps
from services.ai.client import AIClient
from services.repair.columns import standardize_name


COLUMN_SUGGESTION_PROMPT = """You are an expert Data Science and Data Engineering Assistant.
Analyze the following compact column profile information from a tabular dataset.
For each column, propose a clearer, more standard, professional column name (preferably in snake_case), along with an explanation of your reasoning and a confidence score between 0.0 and 1.0.

Columns Profile:
{columns_profile_json}

RULES:
1. ONLY suggest a rename if the current name is cryptic, abbreviated, uninformative, poorly formatted, or non-standard.
2. If the current column name is already clear, descriptive, and well-formatted, set `suggested_name` to the current name, confidence to 1.0, and reason to "Name is already clear and descriptive."
3. `suggested_name` must be a clean snake_case identifier (e.g., "customer_name", "order_id", "total_sales").
4. Never suggest special symbols, spaces, or invalid column characters.
5. Return ONLY a valid JSON object matching this schema:
{{
    "suggestions": [
        {{
            "current_name": "<exact current column name>",
            "suggested_name": "<clean snake_case proposed name>",
            "reason": "<clear 1-sentence reason based on data/sample values>",
            "confidence": <float between 0.0 and 1.0>
        }}
    ]
}}
"""


class AIColumnSuggester:
    """Service to generate and validate AI column name suggestions."""

    @classmethod
    def build_column_profile(
        cls, df: pd.DataFrame, columns: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
        """Build compact, privacy-safe column summaries for AI analysis."""
        target_cols: list[str] = list(columns) if columns is not None else [f"{c}" for c in df.columns]
        profiles = []

        for col in target_cols:
            if col not in df.columns:
                continue
            series = df[col]
            non_null = series.dropna()

            # Up to 4 unique sample values
            samples: list[str] = []
            for val in non_null.unique()[:4]:
                s_val = str(val)
                if len(s_val) > 40:
                    s_val = s_val[:37] + "..."
                samples.append(s_val)

            info: dict[str, Any] = {
                "column_name": col,
                "dtype": str(series.dtype),
                "null_count": series.isna().sum(),
                "unique_count": series.nunique(),
                "sample_values": samples,
            }

            if pd.api.types.is_numeric_dtype(series) and len(non_null) > 0:
                try:
                    info["min"] = float(non_null.min())
                    info["max"] = float(non_null.max())
                except Exception:
                    pass

            profiles.append(info)

        return profiles

    @classmethod
    def suggest_column_names(
        cls, df: pd.DataFrame, columns: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
        """Propose better column names via OpenRouter AI, with heuristic fallback.

        Args:
            df: Source DataFrame.
            columns: Optional subset of columns to evaluate.

        Returns:
            List of suggestion dicts:
            [
                {
                    "current_name": "...",
                    "suggested_name": "...",
                    "reason": "...",
                    "confidence": 0.95,
                    "changed": True/False
                }
            ]
        """
        profiles = cls.build_column_profile(df, columns)
        target_cols = [p["column_name"] for p in profiles]

        if llm_config.is_configured:
            try:
                logger.info("Requesting AI column name suggestions for {} columns", len(profiles))
                prompt = COLUMN_SUGGESTION_PROMPT.format(
                    columns_profile_json=safe_json_dumps(sanitize_for_json(profiles), indent=2)
                )
                raw = AIClient.call_structured(prompt)
                suggestions_list = raw.get("suggestions", [])
                validated = cls._validate_suggestions(df, suggestions_list, target_cols)
                if validated:
                    return validated
            except Exception as e:
                logger.warning("AI column suggestion failed ({}), falling back to heuristic: {}", type(e).__name__, str(e))

        # Heuristic fallback if AI not configured or failed
        return cls._heuristic_suggestions(df, profiles)

    @classmethod
    def _validate_suggestions(
        cls,
        df: pd.DataFrame,
        suggestions_raw: list[dict[str, Any]],
        expected_cols: list[str],
    ) -> list[dict[str, Any]]:
        """Validate AI suggested names for safety and consistency."""
        validated: list[dict[str, Any]] = []
        raw_map = {s.get("current_name"): s for s in suggestions_raw if isinstance(s, dict)}

        seen_targets: set[str] = set()

        for col in expected_cols:
            sug = raw_map.get(col)
            if not sug:
                continue

            raw_suggested = str(sug.get("suggested_name", "")).strip()
            # Clean to snake_case identifier
            clean_sug = standardize_name(raw_suggested, ["snake_case"])
            if not clean_sug:
                clean_sug = standardize_name(col, ["snake_case"])

            # Disambiguate if target is already used
            final_sug = clean_sug
            count = 2
            while final_sug in seen_targets or (final_sug != col and final_sug in df.columns):
                final_sug = f"{clean_sug}_{count}"
                count += 1

            seen_targets.add(final_sug)

            reason = str(sug.get("reason", "AI-recommended standard naming"))
            try:
                conf = float(sug.get("confidence", 0.9))
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                conf = 0.9

            validated.append({
                "current_name": col,
                "suggested_name": final_sug,
                "reason": reason,
                "confidence": conf,
                "changed": col != final_sug,
            })

        return validated

    @classmethod
    def _heuristic_suggestions(
        cls, df: pd.DataFrame, profiles: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Heuristic rule-based column name suggestions for offline/fallback mode."""
        abbrev_map = {
            "cust": "customer",
            "nm": "name",
            "amt": "amount",
            "qty": "quantity",
            "pct": "percentage",
            "val": "value",
            "num": "number",
            "cd": "code",
            "desc": "description",
            "cat": "category",
            "addr": "address",
            "tel": "phone",
            "dob": "date_of_birth",
        }

        results: list[dict[str, Any]] = []
        seen_targets: set[str] = set()

        for prof in profiles:
            col = prof["column_name"]
            curr = f"{col}"
            # 1. Expand common abbreviations token by token
            tokens = re.split(r"([_\s\-\/\\]+)", curr)
            expanded_tokens = []
            abbrev_expanded = False
            for tok in tokens:
                lower_tok = tok.lower()
                if lower_tok in abbrev_map:
                    expanded_tokens.append(abbrev_map[lower_tok])
                    abbrev_expanded = True
                else:
                    expanded_tokens.append(tok)

            expanded = "".join(expanded_tokens)

            # 2. Standardize to snake_case
            clean_name = standardize_name(expanded, ["snake_case"])

            # 3. Disambiguate if collision
            final_name = clean_name
            count = 2
            while final_name in seen_targets or (final_name != curr and final_name in df.columns):
                final_name = f"{clean_name}_{count}"
                count += 1

            seen_targets.add(final_name)

            if abbrev_expanded:
                reason = "Expanded common abbreviations and converted to snake_case."
                conf = 0.92
            elif curr != final_name:
                reason = "Standardized casing, removed special symbols, and cleaned whitespace."
                conf = 0.95
            else:
                reason = "Column name is already clear and standardized."
                conf = 1.0

            results.append({
                "current_name": curr,
                "suggested_name": final_name,
                "reason": reason,
                "confidence": conf,
                "changed": curr != final_name,
            })

        return results
