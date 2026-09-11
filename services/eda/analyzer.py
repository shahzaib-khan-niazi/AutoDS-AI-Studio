"""AI EDA Analyst for AutoDS AI Studio.

Calculates deterministic statistical metrics and patterns across numeric and categorical features,
and prompts LLMService to produce structured, evidence-backed exploratory data interpretations.
"""

import json
from typing import Any, Dict, List, Optional, cast
import numpy as np
import pandas as pd

from core.logging import logger
from core.llm.config import llm_config
from core.llm.service import LLMService
from core.llm.exceptions import LLMError
from core.utils.json_sanitizer import sanitize_for_json, safe_json_dumps
from core.schemas.eda import AIEDAAnalysis, CorrelationItem
from services.eda.service import EDAService
from utils.dataframe import safe_numeric_columns, safe_categorical_columns


class AIEDAAnalyst:
    """Computes statistical relationships and prompts AI for semantic exploratory insights."""

    _SYSTEM_PROMPT = """You are the AI EDA (Exploratory Data Analysis) Analyst for AutoDS AI Studio.
Your role is to interpret deterministic statistical metrics, correlation structures, and feature distributions.

CONSTRAINTS & RULES:
1. ONLY reference features, correlations, and numbers provided in the statistical summary.
2. Do NOT invent correlations, p-values, or distributions.
3. If no significant correlations exist, state so explicitly.
4. Highlight potential feature interactions and risks for predictive modeling.

Output valid JSON matching this schema:
{
  "overview": "2-3 sentence overview of dataset structure and distributions.",
  "key_patterns": ["Key pattern 1", "Key pattern 2"],
  "significant_correlations": ["Correlations explanation with evidence"],
  "distribution_insights": ["Skewness or distribution observations"],
  "potential_interactions": ["Interaction 1 between ColA and ColB"],
  "anomalous_observations": ["Outlier or distribution anomalies"],
  "recommendations_for_modeling": ["Feature recommendation 1", "Feature recommendation 2"]
}
"""

    @classmethod
    def analyze(cls, df: pd.DataFrame) -> AIEDAAnalysis:
        """Perform statistical EDA and generate structured AI interpretation.

        Args:
            df: DataFrame to analyze.

        Returns:
            AIEDAAnalysis object.
        """
        # Deterministic statistics calculation
        summary_dict, corr_items = cls._compute_eda_summary(df)
        summary_dict = sanitize_for_json(summary_dict)

        if not llm_config.is_configured:
            logger.warning("AIEDAAnalyst called without configured API key; using deterministic fallback")
            return cls._fallback_analysis(summary_dict, corr_items)

        user_prompt = f"Analyze these statistical EDA facts:\n{safe_json_dumps(summary_dict, indent=2)}"

        messages = [
            {"role": "system", "content": cls._SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        schema_hint = {
            "name": "AIEDAAnalysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "overview": {"type": "string"},
                    "key_patterns": {"type": "array", "items": {"type": "string"}},
                    "significant_correlations": {"type": "array", "items": {"type": "string"}},
                    "distribution_insights": {"type": "array", "items": {"type": "string"}},
                    "potential_interactions": {"type": "array", "items": {"type": "string"}},
                    "anomalous_observations": {"type": "array", "items": {"type": "string"}},
                    "recommendations_for_modeling": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "overview",
                    "key_patterns",
                    "significant_correlations",
                    "distribution_insights",
                    "potential_interactions",
                    "anomalous_observations",
                    "recommendations_for_modeling",
                ],
            },
        }

        try:
            svc = LLMService()
            result = svc.generate_structured(messages, response_schema=schema_hint)
            raw = json.loads(result.content)

            return AIEDAAnalysis(
                overview=raw.get("overview", "EDA completed."),
                key_patterns=[str(p) for p in raw.get("key_patterns", [])],
                significant_correlations=[str(c) for c in raw.get("significant_correlations", [])],
                distribution_insights=[str(d) for d in raw.get("distribution_insights", [])],
                potential_interactions=[str(i) for i in raw.get("potential_interactions", [])],
                anomalous_observations=[str(a) for a in raw.get("anomalous_observations", [])],
                recommendations_for_modeling=[str(r) for r in raw.get("recommendations_for_modeling", [])],
                confidence_score=0.90,
            )

        except LLMError as e:
            logger.error("LLM error during EDA analysis: {}", e.message)
            return cls._fallback_analysis(summary_dict, corr_items)
        except Exception as e:
            logger.exception("Unexpected error in AIEDAAnalyst: {}", str(e))
            return cls._fallback_analysis(summary_dict, corr_items)

    @classmethod
    def _compute_eda_summary(
        cls, df: pd.DataFrame
    ) -> tuple[Dict[str, Any], List[CorrelationItem]]:
        """Compute compact deterministic summary for EDA interpretation."""
        num_cols = safe_numeric_columns(df)
        cat_cols = safe_categorical_columns(df)

        numeric_stats = {}
        for col in num_cols[:15]:
            s = df[col].dropna()
            if len(s) > 0:
                numeric_stats[col] = {
                    "mean": round(s.mean(), 2),
                    "median": round(s.median(), 2),
                    "std": round(s.std(), 2) if len(s) > 1 else 0.0,
                    "min": round(s.min(), 2),
                    "max": round(s.max(), 2),
                }

        # Correlation analysis
        corr_items: List[CorrelationItem] = []
        if len(num_cols) >= 2:
            corr_matrix = EDAService.get_correlation_matrix(df)
            if corr_matrix is not None:
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        c1 = corr_matrix.columns[i]
                        c2 = corr_matrix.columns[j]
                        val = float(cast(Any, corr_matrix.iloc[i, j]))
                        if abs(val) >= 0.35:
                            strength = (
                                "Strong Positive" if val >= 0.7 else
                                ("Moderate Positive" if val >= 0.35 else
                                ("Strong Negative" if val <= -0.7 else "Moderate Negative"))
                            )
                            corr_items.append(
                                CorrelationItem(
                                    feature_a=c1,
                                    feature_b=c2,
                                    correlation=round(val, 2),
                                    strength=strength,
                                )
                            )

        # Categorical top distribution
        cat_summary = {}
        for col in cat_cols[:10]:
            vc = df[col].value_counts(dropna=True).head(3)
            cat_summary[col] = {
                "unique_count": df[col].nunique(),
                "top_categories": {str(k): v for k, v in vc.items()},
            }

        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns_count": len(num_cols),
            "categorical_columns_count": len(cat_cols),
            "numeric_features": numeric_stats,
            "correlations": [
                f"{c.feature_a} and {c.feature_b}: r = {c.correlation:.2f} ({c.strength})"
                for c in corr_items[:10]
            ],
            "categorical_features": cat_summary,
        }

        return summary, corr_items

    @classmethod
    def _fallback_analysis(
        cls, summary: Dict[str, Any], corr_items: List[CorrelationItem]
    ) -> AIEDAAnalysis:
        """Deterministic fallback analysis when AI is unavailable."""
        patterns = [
            f"Dataset has {summary.get('total_rows', 0):,} rows across {summary.get('numeric_columns_count', 0)} numeric and {summary.get('categorical_columns_count', 0)} categorical features."
        ]

        corr_lines = [
            f"Correlation between '{c.feature_a}' and '{c.feature_b}': r={c.correlation:.2f} ({c.strength})"
            for c in corr_items[:5]
        ] or ["No strong pairwise linear correlations (|r| >= 0.35) detected."]

        dist_lines = []
        for feat, st in list(summary.get("numeric_features", {}).items())[:4]:
            dist_lines.append(f"'{feat}': Mean={st.get('mean')}, Median={st.get('median')}, Range=[{st.get('min')}, {st.get('max')}].")

        return AIEDAAnalysis(
            overview=f"Exploratory analysis on {summary.get('total_rows', 0)} records and {summary.get('total_columns', 0)} attributes.",
            key_patterns=patterns,
            significant_correlations=corr_lines,
            distribution_insights=dist_lines or ["Normal numeric distributions."],
            potential_interactions=["Investigate interaction terms between correlated features."],
            anomalous_observations=["Evaluate extreme bounds in numeric distributions during modeling."],
            recommendations_for_modeling=["Scale numeric features", "Encode categorical variables"],
            confidence_score=0.75,
        )
