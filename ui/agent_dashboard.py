"""Unified AI Data Scientist Agent Dashboard for AutoDS AI Studio.

Orchestrates data profiling & semantic understanding, automated EDA,
AutoML training, explainability, cross-stage synthesis, executive report generation,
and dataset AI Chatbot.
"""

from typing import Optional
import pandas as pd
import streamlit as st

from core.logging import logger
from core.llm.config import llm_config
from core.state import (
    get_dataset,
    get_dataset_name,
    has_dataset,
    log_agent_decision,
    get_decision_log,
)
from services.profiler.service import DatasetProfiler
from services.profiler.validator import DatasetProfileValidator
from services.profiler.ai import AIProfilerInterpreter
from services.eda.analyzer import AIEDAAnalyst
from services.ml.ai_planner import AIMLPlanner
from services.ml.trainer import AutoMLTrainer
from services.explainability.service import ExplainabilityService
from services.insights.service import InsightsService
from services.reports.service import ReportGeneratorService
from ui.theme import render_page_header, render_section_header, render_empty_state
from utils.formatting import format_number


def show_agent_dashboard() -> None:
    """Render the AI Data Scientist Mission Control Dashboard."""
    render_page_header(
        title="AI Data Scientist Mission Control",
        description="Autonomous agent orchestrator: Python calculates deterministically, AI interprets semantically, and strict validation controls safety.",
        icon="🧠",
    )

    if not has_dataset():
        render_empty_state(
            title="No Dataset Loaded for AI Data Scientist",
            description="Please upload a dataset on the Upload page to activate the autonomous workflow.",
            icon="🧠",
        )
        return

    df = get_dataset()
    dataset_name = get_dataset_name() or "dataset.csv"

    if df is None or len(df) == 0:
        st.warning("⚠️ The active dataset is empty.")
        return

    # ── Workflow Stepper Visual Representation ──
    st.markdown(
        """
        <div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; margin-bottom: 8px;">Autonomous AI Workflow Pipeline</div>
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 0.85rem; font-weight: 600; color: #F8FAFC;">
                <span style="background: rgba(99,102,241,0.2); color: #818CF8; padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(99,102,241,0.4);">1. Profile</span> ➔
                <span style="background: rgba(99,102,241,0.2); color: #818CF8; padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(99,102,241,0.4);">2. Clean</span> ➔
                <span style="background: rgba(99,102,241,0.2); color: #818CF8; padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(99,102,241,0.4);">3. Explore</span> ➔
                <span style="background: rgba(99,102,241,0.2); color: #818CF8; padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(99,102,241,0.4);">4. Train</span> ➔
                <span style="background: rgba(99,102,241,0.2); color: #818CF8; padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(99,102,241,0.4);">5. Explain</span> ➔
                <span style="background: rgba(99,102,241,0.2); color: #818CF8; padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(99,102,241,0.4);">6. Insights</span> ➔
                <span style="background: rgba(99,102,241,0.2); color: #818CF8; padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(99,102,241,0.4);">7. Report</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Agent Decision & Activity Log ──
    with st.expander("📋 Agent Decision & Execution Trail", expanded=False):
        decision_logs = get_decision_log()
        if decision_logs:
            for log_entry in decision_logs:
                st.caption(f"• {log_entry}")
        else:
            st.caption("No agent activity recorded yet.")

    st.divider()

    # ── Workflow Stages Tabs (Centered & Unclipped Tabs) ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Profile & Understanding",
        "📊 EDA & Patterns",
        "⚙️ AutoML Modeling",
        "🔬 Explainability",
        "💡 AI Insights",
        "📄 Executive Report",
    ])

    # TAB 1: Profile & Understanding
    with tab1:
        render_section_header("Dataset Profiling & AI Semantic Understanding", icon="🔍")
        st.caption("Deterministic Python profiling generates structural facts; AI interprets semantic domain and quality risks.")

        if st.session_state.get("dataset_profile") is None:
            if st.button("🚀 Profile Dataset & Generate Facts", type="primary", key="btn_run_profile"):
                with st.spinner("Calculating deterministic statistics and bounds..."):
                    prof = DatasetProfiler.profile(df, filename=dataset_name)
                    is_valid, errors = DatasetProfileValidator.validate_profile(prof)
                    if is_valid:
                        st.session_state["dataset_profile"] = prof
                        st.session_state["current_stage"] = "understanding"
                        log_agent_decision(f"Dataset profiled: {prof.row_count:,} rows, {prof.column_count} cols | Quality Score: {prof.quality_score:.1f}/100 ({prof.quality_status})")
                        st.rerun()
                    else:
                        st.error(f"Validation failed: {', '.join(errors)}")

        prof = st.session_state.get("dataset_profile")
        if prof:
            # AI Understanding Section
            render_section_header("AI Semantic Interpretation", icon="🤖")
            if st.session_state.get("ai_dataset_analysis") is None:
                if st.button("🧠 Request AI Dataset Understanding", key="btn_run_ai_understand"):
                    with st.spinner("Interpreting dataset profile with AI..."):
                        analysis = AIProfilerInterpreter.analyze(prof)
                        st.session_state["ai_dataset_analysis"] = analysis
                        st.session_state["current_stage"] = "eda"
                        log_agent_decision(f"AI Domain inferred: '{analysis.likely_domain}' | Key columns: {', '.join(analysis.important_columns[:3])}")
                        st.rerun()

            analysis = st.session_state.get("ai_dataset_analysis")
            if analysis:
                st.info(f"**Domain:** `{analysis.likely_domain}`\n\n**Executive Summary:** {analysis.dataset_summary}")
                st.markdown(f"**Quality Assessment:** {analysis.quality_assessment}")
                if analysis.priority_issues:
                    st.markdown("**Priority Quality Issues Identified by AI:**")
                    for p_iss in analysis.priority_issues:
                        st.markdown(f"• ⚠️ {p_iss}")

    # TAB 2: Automated EDA
    with tab2:
        render_section_header("Exploratory Data Analysis & Visual Insights", icon="📊")
        st.caption("Computes statistical patterns and prompts AI to interpret correlations and distributions.")

        if st.session_state.get("ai_eda_analysis") is None:
            if st.button("📈 Run Statistical EDA & AI Analysis", type="primary", key="btn_run_eda"):
                with st.spinner("Computing correlations, distributions, and AI interpretation..."):
                    eda_res = AIEDAAnalyst.analyze(df)
                    st.session_state["ai_eda_analysis"] = eda_res
                    st.session_state["current_stage"] = "ml_plan"
                    log_agent_decision(f"EDA completed: {len(eda_res.significant_correlations)} correlations and patterns analyzed")
                    st.rerun()

        eda_res = st.session_state.get("ai_eda_analysis")
        if eda_res:
            st.info(f"**EDA Overview:** {eda_res.overview}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔥 Key Correlations & Patterns")
                for pat in eda_res.key_patterns:
                    st.markdown(f"• {pat}")
                for corr in eda_res.significant_correlations:
                    st.markdown(f"• {corr}")
            with c2:
                st.markdown("#### 📊 Distribution & Modeling Insights")
                for dist in eda_res.distribution_insights:
                    st.markdown(f"• {dist}")
                for rec in eda_res.recommendations_for_modeling:
                    st.markdown(f"• 💡 {rec}")

    # TAB 3: AutoML Modeling
    with tab3:
        render_section_header("AI ML Strategy & AutoML Benchmarking", icon="⚙️")
        st.caption("AI formulates the predictive task; Scikit-Learn trains and evaluates multiple candidate models.")

        if st.session_state.get("ml_plan") is None:
            if st.button("🤖 Generate AI Machine Learning Strategy", type="primary", key="btn_gen_ml_plan"):
                with st.spinner("Evaluating candidate targets and models..."):
                    ml_plan = AIMLPlanner.plan(df, profile=st.session_state.get("dataset_profile"))
                    st.session_state["ml_plan"] = ml_plan
                    log_agent_decision(f"ML Strategy formulated: Target='{ml_plan.target_column}', Task='{ml_plan.recommended_task}'")
                    st.rerun()

        ml_plan = st.session_state.get("ml_plan")
        if ml_plan:
            st.markdown(f"**Recommended Target:** `{ml_plan.target_column}` | **Task:** `{ml_plan.recommended_task}`")
            st.caption(f"**Reasoning:** {ml_plan.target_reasoning}")

            # Option to adjust target before training
            target_opts = list(df.columns)
            selected_target = st.selectbox(
                "Confirmed Target Feature for Modeling:",
                options=target_opts,
                index=target_opts.index(ml_plan.target_column) if ml_plan.target_column in target_opts else 0,
                key="agent_ml_target_select",
            )

            if st.button("🚀 Train & Benchmark AutoML Models", type="primary", key="btn_exec_automl"):
                with st.spinner("Training Random Forest, Gradient Boosting, Linear models..."):
                    try:
                        automl_summary = AutoMLTrainer.train(df, target_column=selected_target)
                        st.session_state["automl_summary"] = automl_summary
                        st.session_state["current_stage"] = "explainability"
                        log_agent_decision(f"AutoML training finished: Best model is '{automl_summary.best_model_name}'")
                        st.success(f"🏆 Best Model: **{automl_summary.best_model_name}**")
                        st.rerun()
                    except Exception as e:
                        st.error(f"AutoML training failed: {e}")

        summary = st.session_state.get("automl_summary")
        if summary:
            render_section_header("Model Leaderboard", icon="🏆")
            leaderboard_data = []
            for r in summary.leaderboard:
                leaderboard_data.append({
                    "Model": r.model_name,
                    "Primary Score": f"{r.primary_metric_value:.4f}",
                    "Train Score": f"{r.train_score:.4f}",
                    "Test Score": f"{r.test_score:.4f}",
                    "Fit Time": f"{r.fit_time_seconds:.3f}s",
                })
            st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True, hide_index=True)

    # TAB 4: Explainability
    with tab4:
        render_section_header("Model Explainability & Driver Analysis", icon="🔬")
        st.caption("Python calculates feature importances; AI explains the predictive drivers in plain language.")

        summary = st.session_state.get("automl_summary")
        if not summary:
            st.info("Please train AutoML models in Stage 3 first.")
        else:
            if st.session_state.get("explainability_results") is None:
                if st.button("🔬 Generate Model Explanations", type="primary", key="btn_gen_explainability"):
                    with st.spinner("Generating feature importance explanations..."):
                        exp_rep = ExplainabilityService.explain(summary)
                        st.session_state["explainability_results"] = exp_rep
                        st.session_state["current_stage"] = "insights"
                        log_agent_decision(f"Explainability generated: Top drivers {', '.join(exp_rep.top_driver_features[:3])}")
                        st.rerun()

            exp_rep = st.session_state.get("explainability_results")
            if exp_rep:
                st.info(f"**Impact Summary:** {exp_rep.feature_impact_summary}")
                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    st.markdown("#### 🌟 Top Driver Features")
                    for feat in exp_rep.top_driver_features:
                        st.markdown(f"• `{feat}`")
                with e_col2:
                    st.markdown("#### 📌 Key Findings & Biases")
                    for k in exp_rep.key_findings:
                        st.markdown(f"• {k}")
                    for b in exp_rep.potential_biases_or_risks:
                        st.caption(f"⚠️ {b}")

    # TAB 5: AI Insights
    with tab5:
        render_section_header("Cross-Stage Executive AI Insights", icon="💡")
        st.caption("Synthesizes end-to-end evidence from profiling, EDA, and model results into strategic takeaways.")

        if st.session_state.get("ai_insights") is None:
            if st.button("💡 Synthesize Executive Insights", type="primary", key="btn_gen_insights"):
                with st.spinner("Synthesizing multi-stage findings..."):
                    insights_rep = InsightsService.synthesize(
                        profile=st.session_state.get("dataset_profile"),
                        cleaning_result=st.session_state.get("cleaning_results"),
                        eda_analysis=st.session_state.get("ai_eda_analysis"),
                        automl_summary=st.session_state.get("automl_summary"),
                        explainability=st.session_state.get("explainability_results"),
                    )
                    st.session_state["ai_insights"] = insights_rep
                    st.session_state["current_stage"] = "report"
                    log_agent_decision(f"Executive insights synthesized: '{insights_rep.headline}'")
                    st.rerun()

        insights_rep = st.session_state.get("ai_insights")
        if insights_rep:
            st.markdown(f"### *\"{insights_rep.headline}\"*")
            st.info(insights_rep.executive_summary)

            i_col1, i_col2 = st.columns(2)
            with i_col1:
                st.markdown("#### 🎯 Strategic Recommendations")
                for r in insights_rep.strategic_recommendations:
                    st.markdown(f"• {r}")
            with i_col2:
                st.markdown("#### ⚠️ Critical Risks & Next Steps")
                for c in insights_rep.critical_risks_and_caveats:
                    st.markdown(f"• {c}")
                for a in insights_rep.suggested_next_actions:
                    st.markdown(f"• 🚀 {a}")

    # TAB 6: Final Executive Report
    with tab6:
        render_section_header("Final Executive Report", icon="📄")
        st.caption("Comprehensive, publication-ready report compiling all deterministic facts and AI insights.")

        if st.button("📄 Generate Final Executive Report", type="primary", key="btn_gen_report"):
            with st.spinner("Compiling full executive report..."):
                md_report = ReportGeneratorService.generate_markdown_report(
                    dataset_name=dataset_name,
                    profile=st.session_state.get("dataset_profile"),
                    ai_understanding=st.session_state.get("ai_dataset_analysis"),
                    cleaning_plan=st.session_state.get("cleaning_plan"),
                    cleaning_result=st.session_state.get("cleaning_results"),
                    eda_analysis=st.session_state.get("ai_eda_analysis"),
                    ml_plan=st.session_state.get("ml_plan"),
                    automl_summary=st.session_state.get("automl_summary"),
                    explainability=st.session_state.get("explainability_results"),
                    insights=st.session_state.get("ai_insights"),
                )
                st.session_state["final_report"] = md_report
                log_agent_decision("Final Executive Report generated and ready for export")
                st.success("✅ Executive Report generated!")

        final_rep = st.session_state.get("final_report")
        if final_rep:
            st.divider()
            st.download_button(
                label="📥 Download Executive Report (Markdown)",
                data=final_rep,
                file_name=f"AutoDS_Report_{dataset_name}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.markdown(final_rep)

