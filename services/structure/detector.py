"""Structure detector — evaluates all rules and picks the strongest.

Combines rule scores with a minimum confidence threshold.
Falls back to UNKNOWN if no rule scores above the threshold.
"""

import pandas as pd

from core.logging import logger
from core.constants import MIN_CONFIDENCE_THRESHOLD
from models.structure import StructureType, StructureResult, RuleEvidence
from services.structure.rules import evaluate_all_rules


# Recommended actions for each structure type
_RECOMMENDED_ACTIONS: dict[StructureType, str] = {
    StructureType.NORMAL_TABLE: "No structural changes needed",
    StructureType.WIDE_TABLE: "Consider unpivoting (melt) to convert to long format",
    StructureType.LONG_TABLE: "No structural changes needed (already tidy)",
    StructureType.PIVOT_TABLE: "Consider unpivoting to normalize the data",
    StructureType.CROSSTAB: "Consider unpivoting to normalize the contingency table",
    StructureType.MULTI_HEADER: "Consider flattening headers or removing metadata rows",
    StructureType.TIME_SERIES: "Ensure datetime column is properly parsed and set as index if needed",
    StructureType.SURVEY: "No structural changes needed; consider encoding responses",
    StructureType.TRANSACTIONAL: "No structural changes needed; verify datetime parsing",
    StructureType.UNKNOWN: "Manual inspection recommended",
}


class StructureDetector:
    """Detects the structural type of a DataFrame using rule-based evaluation."""

    @classmethod
    def detect(cls, df: pd.DataFrame) -> StructureResult:
        """Detect the structure type of a DataFrame.

        Evaluates all rules, picks the highest-scoring one above the
        minimum confidence threshold. Falls back to UNKNOWN.

        Args:
            df: Source DataFrame.

        Returns:
            StructureResult with type, confidence, evidence, and recommendation.
        """
        logger.info("Running structure detection ({} rows × {} cols)", len(df), len(df.columns))

        # Evaluate all rules
        all_evidence = evaluate_all_rules(df)

        if not all_evidence:
            return cls._unknown_result(all_evidence)

        # Sort by score descending
        all_evidence.sort(key=lambda r: r.score, reverse=True)

        best = all_evidence[0]

        # Check minimum threshold
        if best.score < MIN_CONFIDENCE_THRESHOLD:
            logger.info("No structure scored above threshold ({:.0%})", MIN_CONFIDENCE_THRESHOLD)
            return cls._unknown_result(all_evidence)

        # Build warnings
        warnings: list[str] = []

        # Warn if second-best is close
        if len(all_evidence) >= 2:
            second = all_evidence[1]
            if second.score > 0 and (best.score - second.score) < 0.1:
                warnings.append(
                    f"Close alternative: {second.structure_type.value} "
                    f"(score: {second.score:.0%} vs {best.score:.0%})"
                )

        recommended = _RECOMMENDED_ACTIONS.get(
            best.structure_type,
            "Manual inspection recommended",
        )

        result = StructureResult(
            structure=best.structure_type,
            confidence=best.score,
            reasons=[best.reason] if best.reason else [],
            evidence=best.evidence,
            all_scores=all_evidence,
            recommended_action=recommended,
            warnings=warnings,
        )

        logger.info(
            "Structure detected: {} (confidence: {:.0%})",
            result.structure.value,
            result.confidence,
        )

        return result

    @staticmethod
    def _unknown_result(all_scores: list[RuleEvidence]) -> StructureResult:
        """Create an UNKNOWN structure result."""
        return StructureResult(
            structure=StructureType.UNKNOWN,
            confidence=0.0,
            reasons=["No structure type scored above the confidence threshold"],
            evidence=[],
            all_scores=all_scores,
            recommended_action=_RECOMMENDED_ACTIONS[StructureType.UNKNOWN],
            warnings=["Manual inspection recommended"],
        )
