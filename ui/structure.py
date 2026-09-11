"""Structure detection UI page for AutoDS AI Studio.

Displays detected structure type, confidence, evidence, recommendations,
and optional AI deep reasoning.
"""

import streamlit as st

from core.config import config
from core.llm.config import llm_config
from core.state import has_dataset, get_dataset
from core.logging import logger
from core.exceptions import StructureDetectionError, AIServiceError
from services.structure.service import StructureService
from services.ai.planner import AIPlanner
from ui.theme import render_page_header, render_metric_card, render_section_header, render_empty_state


def show_structure() -> None:
    """Render the Structure Detection page."""
    render_page_header(
        title="Structure Detection",
        description="Detect structural layouts: standard tabular, multi-header, unpivoted time-series, or key-value hierarchies.",
        icon="🏗️",
    )

    if not has_dataset():
        render_empty_state(
            title="No Dataset Loaded for Structure Analysis",
            description="Please upload a CSV or Excel file on the Upload page to analyze dataset structure.",
            icon="📁",
        )
        return

    df = get_dataset()
    if df is None:
        st.error("Unable to retrieve dataset from session state.")
        return

    # Run detection (cache in session state)
    report = st.session_state.get("structure_report")
    if report is None:
        try:
            with st.spinner("Analyzing dataset structure..."):
                report = StructureService.detect(df)
            st.session_state["structure_report"] = report
        except StructureDetectionError as e:
            st.error(f"❌ {e.message}")
            return
        except Exception as e:
            st.error("❌ Failed to detect structure.")
            logger.exception("Structure detection error: {}", str(e))
            return

    # Control buttons
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("🔄 Re-analyze", use_container_width=True):
            try:
                with st.spinner("Re-analyzing structure..."):
                    report = StructureService.detect(df)
                st.session_state["structure_report"] = report
                st.session_state["ai_structure_plan"] = None
                st.rerun()
            except Exception as e:
                st.error(f"❌ Re-analysis failed: {e}")
                return

    # ── Detected Structure (Deterministic) ──
    render_section_header("Deterministic Structure Assessment", icon="📊")

    col1, col2 = st.columns(2)
    with col1:
        structure_name = report.structure.value.replace("_", " ").title()
        render_metric_card("Detected Layout", structure_name, icon="📐")
    with col2:
        render_metric_card("Confidence Score", f"{report.confidence:.0%}", delta="Rules Confidence", delta_color="success", icon="🎯")

    # ── Evidence ──
    if report.evidence:
        render_section_header("Supporting Evidence", icon="📋")
        for item in report.evidence:
            st.markdown(f"• {item}")

    # ── Reasons ──
    if report.reasons:
        render_section_header("Rule Explanation", icon="💡")
        for reason in report.reasons:
            st.info(reason)

    # ── Recommended Action ──
    if report.recommended_action:
        render_section_header("Recommended Action", icon="🎯")
        st.success(report.recommended_action)

    # ── Warnings ──
    if report.warnings:
        render_section_header("Observations & Warnings", icon="⚠️")
        for warning in report.warnings:
            st.warning(warning)

    # ── All Rule Scores (expandable) ──
    with st.expander("🔍 Detailed Rule Score Breakdown"):
        if report.all_scores:
            for rule_ev in report.all_scores:
                score_pct = f"{rule_ev.score:.0%}"
                label = rule_ev.structure_type.value.replace("_", " ").title()
                st.markdown(f"**{label}**: {score_pct}")
                if rule_ev.evidence:
                    for e in rule_ev.evidence:
                        st.markdown(f"  • {e}")
                st.markdown("---")

    st.divider()

    # ── AI Deep Reasoning Layer ──
    render_section_header("AI Structural Reasoning", icon="🤖")
    st.caption("Use LLM reasoning for ambiguous datasets or subtle dimensional patterns.")

    if not llm_config.is_configured:
        st.info("💡 To enable AI Reasoning, configure `OPENROUTER_API_KEY` in your `.env` file.")
    else:
        if st.button("🧠 Request AI Structure Analysis", key="ai_structure_btn", type="primary"):
            try:
                with st.spinner("AI analyzing dataset profile..."):
                    inspection = st.session_state.get("inspection_report")
                    ai_plan = AIPlanner.analyze_structure(df, inspection=inspection)
                st.session_state["ai_structure_plan"] = ai_plan
            except AIServiceError as e:
                st.error(f"❌ AI Error: {e.message}")
                if e.details:
                    st.caption(e.details)
            except Exception as e:
                st.error(f"❌ Unexpected AI error: {e}")

        ai_plan = st.session_state.get("ai_structure_plan")
        if ai_plan:
            st.markdown(f"**AI Decision:** `{ai_plan.decision}` (Confidence: **{ai_plan.confidence:.0%}**)")
            if ai_plan.reasoning_summary:
                st.info(f"**Reasoning:** {ai_plan.reasoning_summary}")
            if ai_plan.actions:
                st.markdown("**AI Recommended Actions:**")
                for act in ai_plan.actions:
                    st.markdown(f"• **{act.operation}** on {act.target if act.target else 'dataset'}: {act.reason}")
            if ai_plan.warnings:
                for w in ai_plan.warnings:
                    st.warning(f"⚠️ {w}")
