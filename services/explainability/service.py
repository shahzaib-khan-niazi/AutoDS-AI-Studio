"""AI Explainability Service for AutoDS AI Studio.

Translates Scikit-Learn tree feature importances and evaluation metrics into clear,
evidence-based natural language explanations and practical business takeaways.
"""

import json
from typing import Any, Dict, List, Optional

from core.logging import logger
from core.llm.config import llm_config
from core.llm.service import LLMService
from core.llm.exceptions import LLMError
from core.utils.json_sanitizer import sanitize_for_json, safe_json_dumps
from core.schemas.ml import AIExplainabilityReport
from models.ml import AutoMLSummary


class ExplainabilityService:
    """Interprets model evaluation results and feature importance."""

    _SYSTEM_PROMPT = """You are the AI Model Explainability Specialist for AutoDS AI Studio.
Your role is to explain trained machine learning models based strictly on computed evaluation metrics and feature importances.

CONSTRAINTS & RULES:
1. ONLY reference features and metric scores provided in the model summary.
2. Clearly explain which features drive the model's predictions the most.
3. Highlight potential biases, risks, or real-world limitations.
4. Do NOT fabricate feature impact or performance metrics.

Output valid JSON matching this schema:
{
  "best_model_name": "Random Forest Classifier",
  "primary_metric_name": "Accuracy",
  "primary_metric_score": 0.88,
  "top_driver_features": ["FeatureA", "FeatureB"],
  "feature_impact_summary": "Clear 2-3 sentence overview of which features dominate predictions.",
  "key_findings": ["Finding 1", "Finding 2"],
  "potential_biases_or_risks": ["Risk 1", "Risk 2"],
  "practical_takeaways": ["Takeaway 1", "Takeaway 2"]
}
"""

    @classmethod
    def explain(cls, summary: AutoMLSummary) -> AIExplainabilityReport:
        """Generate structured AI explanation for an AutoMLSummary.

        Args:
            summary: Completed AutoMLSummary object.

        Returns:
            AIExplainabilityReport.
        """
        best_model = summary.best_model_name
        top_importances = [
            {"feature": fi.feature, "importance": round(fi.importance, 4)}
            for fi in summary.feature_importances[:10]
        ]

        best_eval = next((r for r in summary.leaderboard if r.model_name == best_model), None)
        metric_name = best_eval.primary_metric_name if best_eval else "Score"
        metric_score = best_eval.primary_metric_value if best_eval else 0.0

        summary_payload = {
            "target_column": summary.target_column,
            "task_type": summary.task_type.value,
            "best_model": best_model,
            "primary_metric": f"{metric_name} = {metric_score:.4f}",
            "features_used_count": len(summary.features_used),
            "rows_trained": summary.rows_trained,
            "feature_importances": top_importances,
            "leaderboard": [
                {
                    "model": r.model_name,
                    "score": f"{r.primary_metric_name}: {r.primary_metric_value:.4f}",
                    "fit_time": f"{r.fit_time_seconds}s",
                }
                for r in summary.leaderboard
            ],
        }

        summary_payload = sanitize_for_json(summary_payload)

        if not llm_config.is_configured:
            logger.warning("ExplainabilityService called without configured API key; using fallback")
            return cls._fallback_explanation(summary, metric_name, metric_score, top_importances)

        user_prompt = f"Explain the behavior and feature influence of this best-performing model:\n{safe_json_dumps(summary_payload, indent=2)}"

        messages = [
            {"role": "system", "content": cls._SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        schema_hint = {
            "name": "AIExplainabilityReport",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "best_model_name": {"type": "string"},
                    "primary_metric_name": {"type": "string"},
                    "primary_metric_score": {"type": "number"},
                    "top_driver_features": {"type": "array", "items": {"type": "string"}},
                    "feature_impact_summary": {"type": "string"},
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "potential_biases_or_risks": {"type": "array", "items": {"type": "string"}},
                    "practical_takeaways": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "best_model_name",
                    "primary_metric_name",
                    "primary_metric_score",
                    "top_driver_features",
                    "feature_impact_summary",
                    "key_findings",
                    "potential_biases_or_risks",
                    "practical_takeaways",
                ],
            },
        }

        try:
            svc = LLMService()
            result = svc.generate_structured(messages, response_schema=schema_hint)
            raw = json.loads(result.content)

            return AIExplainabilityReport(
                best_model_name=str(raw.get("best_model_name", best_model)),
                primary_metric_name=str(raw.get("primary_metric_name", metric_name)),
                primary_metric_score=float(raw.get("primary_metric_score", metric_score)),
                top_driver_features=[str(f) for f in raw.get("top_driver_features", [fi["feature"] for fi in top_importances[:5]])],
                feature_impact_summary=str(raw.get("feature_impact_summary", "Model explained.")),
                key_findings=[str(k) for k in raw.get("key_findings", [])],
                potential_biases_or_risks=[str(r) for r in raw.get("potential_biases_or_risks", [])],
                practical_takeaways=[str(t) for t in raw.get("practical_takeaways", [])],
            )

        except LLMError as e:
            logger.error("LLM error during explainability generation: {}", e.message)
            return cls._fallback_explanation(summary, metric_name, metric_score, top_importances)
        except Exception as e:
            logger.exception("Unexpected error in ExplainabilityService: {}", str(e))
            return cls._fallback_explanation(summary, metric_name, metric_score, top_importances)

    @classmethod
    def _fallback_explanation(
        cls, summary: AutoMLSummary, metric_name: str, metric_score: float, top_importances: List[Dict[str, Any]]
    ) -> AIExplainabilityReport:
        """Deterministic fallback explanation when AI is unconfigured."""
        top_feats = [fi["feature"] for fi in top_importances[:5]]
        findings = [
            f"'{summary.best_model_name}' achieved top performance with {metric_name} = {metric_score:.4f}.",
            f"Trained on {summary.rows_trained:,} observations across {len(summary.features_used)} feature attributes.",
        ]
        if top_feats:
            findings.append(f"Leading predictive factors: {', '.join(top_feats)}.")

        return AIExplainabilityReport(
            best_model_name=summary.best_model_name,
            primary_metric_name=metric_name,
            primary_metric_score=metric_score,
            top_driver_features=top_feats,
            feature_impact_summary=f"Predictions are primarily driven by {', '.join(top_feats[:3]) if top_feats else 'the input features'}.",
            key_findings=findings,
            potential_biases_or_risks=["Evaluate model predictions on external out-of-time validation sets."],
            practical_takeaways=["Deploy best model or calibrate decision thresholds for production use."],
        )
