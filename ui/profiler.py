"""AI Dataset Profiler Streamlit UI page for AutoDS AI Studio.

Provides deterministic data profiling and structured AI dataset understanding.
"""

from typing import Optional
import pandas as pd
import streamlit as st

from core.exceptions import AIServiceError
from core.llm.config import llm_config
from core.logging import logger
from core.schemas.dataset_profile import DatasetAIAnalysis, DatasetProfile
from core.state import get_dataset, get_dataset_name, has_dataset
from services.profiler.ai import AIProfilerInterpreter
from services.profiler.service import DatasetProfiler
from services.profiler.validator import DatasetProfileValidator
from ui.theme import render_page_header, render_metric_card, render_section_header, render_empty_state
from utils.formatting import format_number


def show_profiler() -> None:
    """Render the AI Dataset Profiler page."""
    render_page_header(
        title="AI Dataset Profiler",
        description="Automatic structural analysis, data health audit, and LLM semantic dataset understanding.",
        icon="🔍",
    )

    if not has_dataset():
        render_empty_state(
            title="No Dataset Loaded for Profiling",
            description="Please upload a CSV or Excel file on the Upload page to run dataset profiling.",
            icon="📁",
        )
        return

    df = get_dataset()
    filename = get_dataset_name() or "dataset.csv"

    if df is None or len(df) == 0:
        st.warning("⚠️ The current dataset is empty.")
        return

    # Cache profile in session state if dataset unchanged
    profile_key = f"dataset_profile_{filename}_{len(df)}_{len(df.columns)}"
    if profile_key not in st.session_state:
        with st.spinner("Calculating deterministic dataset profile..."):
            profile = DatasetProfiler.profile(df, filename=filename)
            is_valid, val_errors = DatasetProfileValidator.validate_profile(profile)
            if not is_valid:
                st.error("❌ Dataset profile validation failed:")
                for err in val_errors[:3]:
                    st.caption(f"• {err}")
            st.session_state[profile_key] = profile
            st.session_state["dataset_profile"] = profile

    profile: DatasetProfile = st.session_state["dataset_profile"]

    # ── Dataset Overview Metrics ──
    render_section_header("Dataset Health Overview", icon="📊")

    total_missing_cells = sum(c.missing_count for c in profile.columns)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Rows", format_number(profile.row_count), icon="📐")
    with col2:
        render_metric_card("Columns", format_number(profile.column_count), icon="📊")
    with col3:
        render_metric_card("Duplicate Rows", f"{format_number(profile.duplicate_row_count)} ({profile.duplicate_row_percentage:.1f}%)", icon="📋")
    with col4:
        render_metric_card("Missing Cells", format_number(total_missing_cells), icon="⚠️")
    with col5:
        score_color = "success" if profile.quality_score >= 80 else ("warning" if profile.quality_score >= 60 else "danger")
        render_metric_card("Quality Score", f"{profile.quality_score:.1f} / 100", delta=profile.quality_status, delta_color=score_color, icon="🛡️")

    st.caption(f"Memory Usage: `{profile.memory_usage_mb:.2f} MB` | Profiled at: `{profile.profiled_at}`")
    st.divider()

    # ── Column Analysis Table ──
    render_section_header("Column Analysis Table", icon="📋")

    table_data = []
    for c in profile.columns:
        table_data.append({
            "Column": c.name,
            "Type": c.dtype,
            "Missing": c.missing_count,
            "Missing %": f"{c.missing_percentage:.1f}%",
            "Unique": c.unique_count,
            "Unique %": f"{c.unique_percentage:.1f}%",
            "Possible ID": "Yes" if c.is_possible_id else "No",
            "High Card": "Yes" if c.is_high_cardinality else "No",
            "Constant": "Yes" if c.is_constant else "No",
        })

    col_df = pd.DataFrame(table_data)
    st.dataframe(col_df, use_container_width=True, hide_index=True)

    # Detailed column expander
    with st.expander("🔍 View Detailed Statistical Breakdown per Column"):
        for c in profile.columns:
            st.markdown(f"#### `{c.name}` ({c.dtype})")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.caption(f"Non-Null: **{c.non_null_count}**")
                st.caption(f"Missing: **{c.missing_count}** ({c.missing_percentage:.1f}%)")
            with m_col2:
                st.caption(f"Unique: **{c.unique_count}** ({c.unique_percentage:.1f}%)")
                st.caption(f"Category: **{'Numeric' if c.is_numeric else ('Categorical' if c.is_categorical else 'Other')}**")
            with m_col3:
                if c.is_numeric:
                    st.caption(f"Min / Max: **{c.min_val} / {c.max_val}**")
                    st.caption(f"Mean / Std: **{c.mean_val} / {c.std_val}**")
                elif c.is_datetime:
                    st.caption(f"Date Range: **{c.min_date} to {c.max_date}**")
                else:
                    st.caption("Numerical Stats: N/A")
            with m_col4:
                if c.top_values:
                    top_str = ", ".join([f"'{k}': {v}" for k, v in list(c.top_values.items())[:3]])
                    st.caption(f"Top Values: {top_str}")
                else:
                    st.caption("Top Values: N/A")
            st.markdown("---")

    st.divider()

    # ── Deterministic Data Quality Issues ──
    render_section_header("Deterministic Data Quality Audit", icon="⚠️")

    if not profile.detected_issues:
        st.success("✅ Excellent dataset health! Zero deterministic quality issues detected.")
    else:
        for issue in profile.detected_issues:
            sev_badge = {
                "critical": "🔴 CRITICAL",
                "high": "🟠 HIGH",
                "medium": "🟡 MEDIUM",
                "low": "🔵 LOW",
            }.get(issue.severity.lower(), "🟡 MEDIUM")

            col_name = f" in `{issue.column}`" if issue.column else ""
            st.warning(f"**{sev_badge}** | **{issue.issue_type.upper()}**{col_name}\n\n{issue.description}")

    st.divider()

    # ── AI Dataset Understanding & Interpretation ──
    render_section_header("AI Dataset Understanding", icon="🤖")
    st.caption("Statistics calculated deterministically by Python; interpreted semantically by AI.")

    analysis_key = f"ai_dataset_analysis_{filename}_{len(df)}_{len(df.columns)}"

    if not llm_config.is_configured:
        st.info("💡 To enable AI Dataset Understanding, configure `OPENROUTER_API_KEY` in your `.env` file.")
    else:
        run_ai = st.button("🧠 Request AI Dataset Interpretation", key="btn_run_ai_profiler", type="primary")

        if run_ai or analysis_key in st.session_state:
            if run_ai:
                with st.spinner("Analyzing dataset profile with AI..."):
                    try:
                        analysis = AIProfilerInterpreter.analyze(profile)
                        st.session_state[analysis_key] = analysis
                        st.session_state["ai_dataset_analysis"] = analysis
                    except Exception as e:
                        st.error(f"❌ AI interpretation failed: {e}")
                        logger.exception("AI interpretation error")

            analysis: Optional[DatasetAIAnalysis] = st.session_state.get(analysis_key)

            if analysis:
                st.markdown("---")

                col_a1, col_a2 = st.columns([3, 1])
                with col_a1:
                    st.markdown(f"### 🌐 Likely Domain: `{analysis.likely_domain}`")
                with col_a2:
                    conf_color = "success" if analysis.confidence_score >= 0.8 else ("warning" if analysis.confidence_score >= 0.6 else "danger")
                    render_metric_card("AI Confidence", f"{analysis.confidence_score:.0%}", delta=analysis.confidence_level, delta_color=conf_color, icon="🧠")

                st.markdown("#### 📝 Executive Summary")
                st.info(analysis.dataset_summary)

                st.markdown("#### 🏥 Quality Assessment")
                st.markdown(analysis.quality_assessment)

                if analysis.important_columns:
                    st.markdown("#### ⭐ Key Columns")
                    st.markdown(" ".join([f"`{col}`" for col in analysis.important_columns]))

                if analysis.detected_issues:
                    st.markdown("#### 🔍 AI-Interpreted Issues & Evidence")
                    for iss in analysis.detected_issues:
                        sev_icon = {
                            "critical": "🔴",
                            "high": "🟠",
                            "medium": "🟡",
                            "low": "🔵",
                        }.get(iss.severity, "🟡")

                        target_str = f" in `{iss.column}`" if iss.column else ""
                        with st.expander(f"{sev_icon} [{iss.severity.upper()}] {iss.issue_type}{target_str}", expanded=True):
                            st.markdown(f"**Explanation:** {iss.explanation}")
                            st.markdown(f"**📌 Profile Evidence:** _{iss.evidence}_")
                            st.markdown(f"**💡 Recommendation:** {iss.recommendation}")

                if analysis.recommended_next_steps:
                    st.markdown("#### 🚀 Recommended Next Steps")
                    for step in analysis.recommended_next_steps:
                        st.markdown(f"• {step}")

                if analysis.warnings:
                    for w in analysis.warnings:
                        st.caption(f"⚠️ {w}")
