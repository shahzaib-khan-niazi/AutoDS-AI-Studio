"""AI Dataset Profiler Interpreter for AutoDS AI Studio.

Uses LLMService to generate structured AI dataset understanding and analysis.
Receives a validated DatasetProfile. Python calculates final confidence score.
"""

import json
import re
from typing import Any, Dict, List, Optional

from core.logging import logger
from core.llm.config import llm_config
from core.llm.service import LLMService
from core.llm.exceptions import LLMError
from core.utils.json_sanitizer import sanitize_for_json, safe_json_dumps
from core.schemas.dataset_profile import (
    DatasetProfile,
    DatasetAIAnalysis,
    DetectedIssue,
)


class AIProfilerInterpreter:
    """Interprets a validated DatasetProfile using LLMService."""

    _SYSTEM_PROMPT = """You are the AI Dataset Understanding component of AutoDS AI Studio.
Your role is to analyze the provided deterministic DatasetProfile JSON and produce a structured, high-insight dataset interpretation.

RULES:
1. Do NOT invent statistics, counts, or metrics that are not in the profile.
2. Every detected issue MUST include concrete evidence from the profile in the `evidence` field (e.g., "Column 'Age' has 143 missing values (14.3%).").
3. Do NOT attempt to execute data operations or code.
4. If information for a field is unavailable or unclear, state: "Not enough evidence."
5. Severity must strictly be one of: "low", "medium", "high", "critical".

Output valid JSON matching this schema:
{
  "dataset_summary": "Concise 2-3 sentence overview of what the dataset contains.",
  "likely_domain": "Domain category (e.g., E-commerce, Healthcare, Finance, HR, Real Estate).",
  "important_columns": ["col1", "col2"],
  "quality_assessment": "Summary of overall dataset health and main quality risks.",
  "detected_issues": [
    {
      "issue_type": "missing_values",
      "column": "Age",
      "severity": "high",
      "explanation": "High missing rate in key demographic feature.",
      "evidence": "Age has 143 missing values (14.3%).",
      "recommendation": "Impute using median or evaluate drop if not critical."
    }
  ],
  "priority_issues": ["Top 1-3 issues to address first"],
  "recommended_next_steps": ["Step 1", "Step 2"],
  "warnings": ["Warning 1"]
}
"""

    @classmethod
    def analyze(cls, profile: DatasetProfile) -> DatasetAIAnalysis:
        """Generate structured AI analysis from a validated DatasetProfile.

        Args:
            profile: Validated DatasetProfile object.

        Returns:
            DatasetAIAnalysis object.
        """
        if not llm_config.is_configured:
            logger.warning("AIProfilerInterpreter called without OPENROUTER_API_KEY")
            return cls._fallback_analysis(
                profile,
                reason="AI service is not configured. Please add OPENROUTER_API_KEY to your .env file.",
            )

        # Build compact profile representation for the LLM
        compact_profile = sanitize_for_json(cls._prepare_compact_profile(profile))
        user_prompt = f"Analyze this DatasetProfile JSON:\n{safe_json_dumps(compact_profile, indent=2)}"

        messages = [
            {"role": "system", "content": cls._SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        schema_hint = {
            "name": "DatasetAIAnalysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "dataset_summary": {"type": "string"},
                    "likely_domain": {"type": "string"},
                    "important_columns": {"type": "array", "items": {"type": "string"}},
                    "quality_assessment": {"type": "string"},
                    "detected_issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "issue_type": {"type": "string"},
                                "column": {"type": ["string", "null"]},
                                "severity": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high", "critical"],
                                },
                                "explanation": {"type": "string"},
                                "evidence": {"type": "string"},
                                "recommendation": {"type": "string"},
                            },
                            "required": [
                                "issue_type",
                                "severity",
                                "explanation",
                                "evidence",
                                "recommendation",
                            ],
                        },
                    },
                    "priority_issues": {"type": "array", "items": {"type": "string"}},
                    "recommended_next_steps": {"type": "array", "items": {"type": "string"}},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "dataset_summary",
                    "likely_domain",
                    "important_columns",
                    "quality_assessment",
                    "detected_issues",
                    "priority_issues",
                    "recommended_next_steps",
                    "warnings",
                ],
            },
        }

        try:
            svc = LLMService()
            result = svc.generate_structured(messages, response_schema=schema_hint)

            raw_dict = json.loads(result.content)

            # Parse issues
            issues_list: List[DetectedIssue] = []
            for item in raw_dict.get("detected_issues", []):
                sev = item.get("severity", "medium").lower()
                if sev not in ("low", "medium", "high", "critical"):
                    sev = "medium"
                issues_list.append(
                    DetectedIssue(
                        issue_type=str(item.get("issue_type", "general")),
                        column=item.get("column"),
                        severity=sev,
                        explanation=str(item.get("explanation", "")),
                        evidence=str(item.get("evidence", "Evidence from profile.")),
                        recommendation=str(item.get("recommendation", "")),
                    )
                )

            # Python calculates deterministic confidence score based on evidence strength
            confidence_score, confidence_level = cls._calculate_confidence(profile, issues_list)

            analysis = DatasetAIAnalysis(
                dataset_summary=str(raw_dict.get("dataset_summary", "Dataset profiled successfully.")),
                likely_domain=str(raw_dict.get("likely_domain", "General Data")),
                important_columns=[str(c) for c in raw_dict.get("important_columns", [])],
                quality_assessment=str(raw_dict.get("quality_assessment", "Quality evaluated.")),
                detected_issues=issues_list,
                priority_issues=[str(p) for p in raw_dict.get("priority_issues", [])],
                recommended_next_steps=[str(s) for s in raw_dict.get("recommended_next_steps", [])],
                warnings=[str(w) for w in raw_dict.get("warnings", [])],
                confidence_score=confidence_score,
                confidence_level=confidence_level,
            )

            logger.info(
                "AI Dataset Analysis completed | Domain='{}' | Issues={} | Confidence={:.2f} ({})",
                analysis.likely_domain,
                len(analysis.detected_issues),
                analysis.confidence_score,
                analysis.confidence_level,
            )

            return analysis

        except LLMError as e:
            logger.error("LLM failure during dataset analysis: {}", e.message)
            return cls._fallback_analysis(profile, reason=f"AI Service error: {e.message}")
        except Exception as e:
            logger.exception("Unexpected error during AI dataset analysis: {}", str(e))
            return cls._fallback_analysis(profile, reason=f"Unexpected error: {str(e)}")

    @classmethod
    def _prepare_compact_profile(cls, profile: DatasetProfile) -> Dict[str, Any]:
        """Convert DatasetProfile into a compact dict optimized for LLM token limits."""
        cols_summary = []
        for col in profile.columns:
            cd: Dict[str, Any] = {
                "name": col.name,
                "dtype": col.dtype,
                "missing_pct": f"{col.missing_percentage:.1f}%",
                "unique_count": col.unique_count,
            }
            if col.is_numeric:
                if col.min_val is not None and col.max_val is not None:
                    cd["range"] = [col.min_val, col.max_val]
                if col.mean_val is not None:
                    cd["mean"] = col.mean_val
            if col.is_categorical and col.top_values:
                cd["top_values"] = col.top_values
            if col.is_datetime:
                if col.min_date and col.max_date:
                    cd["date_range"] = [col.min_date, col.max_date]
            if col.is_possible_id:
                cd["tag"] = "possible_id"
            if col.is_constant:
                cd["tag"] = "constant"
            if col.is_high_cardinality:
                cd["tag"] = "high_cardinality"
            cols_summary.append(cd)

        return {
            "filename": profile.filename,
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "duplicate_rows": f"{profile.duplicate_row_count} ({profile.duplicate_row_percentage:.1f}%)",
            "quality_score": f"{profile.quality_score:.1f} / 100 ({profile.quality_status})",
            "columns": cols_summary,
            "deterministic_issues": [
                {
                    "issue_type": issue.issue_type,
                    "column": issue.column,
                    "severity": issue.severity,
                    "description": issue.description,
                }
                for issue in profile.detected_issues
            ],
        }

    @classmethod
    def _calculate_confidence(
        cls, profile: DatasetProfile, issues: List[DetectedIssue]
    ) -> tuple[float, str]:
        """Calculates a deterministic confidence score (0.0 to 1.0) based on evidence verification."""
        if not issues:
            return 0.90, "High"

        valid_evidence_count = 0
        column_names = {c.name.lower() for c in profile.columns}

        for issue in issues:
            ev = issue.evidence.lower()
            # Evidence must contain numbers, %, count, or a known column name
            has_numbers = bool(re.search(r"\d", ev))
            has_col_ref = any(c_name in ev for c_name in column_names) if column_names else False
            if has_numbers or has_col_ref or "%" in ev:
                valid_evidence_count += 1

        ratio = valid_evidence_count / len(issues)
        score = round(0.50 + (ratio * 0.45), 2)  # Base 0.50 + up to 0.45

        if score >= 0.80:
            level = "High"
        elif score >= 0.60:
            level = "Medium"
        else:
            level = "Low"

        return score, level

    @classmethod
    def _fallback_analysis(
        cls, profile: DatasetProfile, reason: str
    ) -> DatasetAIAnalysis:
        """Fallback analysis generated deterministically when AI is unavailable."""
        issues: List[DetectedIssue] = []
        for det in profile.detected_issues:
            issues.append(
                DetectedIssue(
                    issue_type=det.issue_type,
                    column=det.column,
                    severity=det.severity,
                    explanation=det.description,
                    evidence=det.metric_value or det.description,
                    recommendation="Review and resolve using Dataset Repair tools.",
                )
            )

        return DatasetAIAnalysis(
            dataset_summary=f"Dataset containing {profile.row_count} rows and {profile.column_count} columns.",
            likely_domain="General Structured Data",
            important_columns=[c.name for c in profile.columns[:5]],
            quality_assessment=f"Quality Score is {profile.quality_score:.1f}/100 ({profile.quality_status}).",
            detected_issues=issues,
            priority_issues=[det.description for det in profile.detected_issues[:3]],
            recommended_next_steps=["Review column quality metrics", "Proceed to Dataset Repair"],
            warnings=[f"Fallback analysis used: {reason}"],
            confidence_score=0.70,
            confidence_level="Medium",
        )
