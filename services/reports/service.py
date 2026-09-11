"""Final Executive Report Generator for AutoDS AI Studio.

Compiles end-to-end dataset profiling, data quality audits, cleaning diffs,
exploratory findings, AutoML benchmarks, explainability analysis, and AI insights
into a comprehensive, publication-ready report.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from core.schemas.dataset_profile import DatasetProfile, DatasetAIAnalysis
from core.schemas.cleaning import CleaningExecutionResult, CleaningPlan
from core.schemas.eda import AIEDAAnalysis
from core.schemas.ml import AIExplainabilityReport, AIInsightsReport, AIMLPlan
from models.ml import AutoMLSummary


class ReportGeneratorService:
    """Generates comprehensive executive data science reports."""

    @classmethod
    def generate_markdown_report(
        cls,
        dataset_name: str,
        profile: Optional[DatasetProfile] = None,
        ai_understanding: Optional[DatasetAIAnalysis] = None,
        cleaning_plan: Optional[CleaningPlan] = None,
        cleaning_result: Optional[CleaningExecutionResult] = None,
        eda_analysis: Optional[AIEDAAnalysis] = None,
        ml_plan: Optional[AIMLPlan] = None,
        automl_summary: Optional[AutoMLSummary] = None,
        explainability: Optional[AIExplainabilityReport] = None,
        insights: Optional[AIInsightsReport] = None,
    ) -> str:
        """Compile all available pipeline intelligence into a structured Markdown document.

        Returns:
            Formatted Markdown report string.
        """
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines: List[str] = [
            f"# 📊 AutoDS AI Studio — Executive Data Science Report",
            f"**Dataset:** `{dataset_name}`  ",
            f"**Generated:** `{now_str}`  ",
            f"**Platform:** AutoDS AI Studio (Autonomous Data Science Engine)  ",
            "\n---\n",
        ]

        # 1. Executive Summary
        lines.append("## 1. Executive Summary")
        if insights and insights.headline:
            lines.append(f"### *\"{insights.headline}\"*\n")
            lines.append(f"{insights.executive_summary}\n")
        elif ai_understanding:
            lines.append(f"{ai_understanding.dataset_summary}\n")
        else:
            lines.append(f"Automated data science investigation conducted on dataset `{dataset_name}`.\n")

        # 2. Dataset Overview & Data Quality Health
        lines.append("## 2. Dataset Profile & Data Quality Audit")
        if profile:
            lines.append(
                f"- **Dimensions:** {profile.row_count:,} rows × {profile.column_count:,} columns\n"
                f"- **Duplicate Rows:** {profile.duplicate_row_count:,} ({profile.duplicate_row_percentage:.1f}%)\n"
                f"- **Memory Usage:** {profile.memory_usage_mb:.2f} MB\n"
                f"- **Quality Score:** **{profile.quality_score:.1f} / 100** ({profile.quality_status})\n"
            )
            if profile.detected_issues:
                lines.append("### Detected Quality Issues:")
                for iss in profile.detected_issues:
                    lines.append(f"- **[{iss.severity.upper()}]** `{iss.issue_type}`: {iss.description}")
            else:
                lines.append("✅ Zero critical data quality defects detected.")
        lines.append("\n")

        # 3. Data Cleaning & Transformation Operations
        lines.append("## 3. Data Cleaning & Preprocessing")
        if cleaning_result:
            lines.append(f"**Impact Summary:** {cleaning_result.diff_summary}\n")
            if cleaning_result.operations_applied:
                lines.append("### Executed Cleaning Operations:")
                for op in cleaning_result.operations_applied:
                    lines.append(f"- ✓ {op}")
        else:
            lines.append("No cleaning transformations recorded.")
        lines.append("\n")

        # 4. Exploratory Data Analysis & Visual Insights
        lines.append("## 4. Exploratory Data Analysis (EDA) Insights")
        if eda_analysis:
            lines.append(f"{eda_analysis.overview}\n")
            if eda_analysis.significant_correlations:
                lines.append("### Significant Correlations:")
                for corr in eda_analysis.significant_correlations:
                    lines.append(f"- {corr}")
            if eda_analysis.distribution_insights:
                lines.append("### Feature Distributions:")
                for dist in eda_analysis.distribution_insights:
                    lines.append(f"- {dist}")
        else:
            lines.append("Exploratory Data Analysis not executed.")
        lines.append("\n")

        # 5. Machine Learning & Model Benchmark
        lines.append("## 5. Machine Learning & Model Benchmarking")
        if automl_summary:
            lines.append(
                f"- **Target Column:** `{automl_summary.target_column}`\n"
                f"- **Task Type:** `{automl_summary.task_type.value}`\n"
                f"- **Best Performing Model:** **{automl_summary.best_model_name}**\n"
                f"- **Training Sample Size:** {automl_summary.rows_trained:,} rows\n"
            )
            lines.append("### Model Leaderboard:")
            lines.append("| Model | Score | Fit Time |")
            lines.append("|---|---|---|")
            for m in automl_summary.leaderboard:
                lines.append(f"| {m.model_name} | {m.primary_metric_name}: {m.primary_metric_value:.4f} | {m.fit_time_seconds:.3f}s |")
        else:
            lines.append("Machine learning benchmark not executed.")
        lines.append("\n")

        # 6. Model Explainability & Drivers
        lines.append("## 6. Model Explainability & Key Drivers")
        if explainability:
            lines.append(f"{explainability.feature_impact_summary}\n")
            if explainability.top_driver_features:
                lines.append("### Top Predictive Drivers:")
                for feat in explainability.top_driver_features:
                    lines.append(f"- 🌟 `{feat}`")
            if explainability.key_findings:
                lines.append("### Key Findings:")
                for k in explainability.key_findings:
                    lines.append(f"- {k}")
        else:
            lines.append("Model explainability not computed.")
        lines.append("\n")

        # 7. Strategic Recommendations & Next Actions
        lines.append("## 7. Strategic Recommendations & Next Steps")
        if insights:
            if insights.strategic_recommendations:
                lines.append("### Recommendations:")
                for r in insights.strategic_recommendations:
                    lines.append(f"- 🎯 {r}")
            if insights.critical_risks_and_caveats:
                lines.append("### Risks & Caveats:")
                for c in insights.critical_risks_and_caveats:
                    lines.append(f"- ⚠️ {c}")
            if insights.suggested_next_actions:
                lines.append("### Immediate Next Actions:")
                for a in insights.suggested_next_actions:
                    lines.append(f"- 🚀 {a}")
        else:
            lines.append("- Review data quality benchmarks and validate predictions in staging environments.")

        lines.append("\n---\n*Report compiled by AutoDS AI Studio.*")
        return "\n".join(lines)
