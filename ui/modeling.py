"""AutoML Modeling UI page for AutoDS AI Studio.

Trains, benchmarks, and evaluates machine learning models on the clean dataset.
"""

from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px

from core.state import has_dataset, get_dataset, get_dataset_name
from core.logging import logger
from models.ml import MLTaskType, AutoMLSummary
from services.ml.planner import MLPlanner
from services.ml.trainer import AutoMLTrainer
from ui.theme import render_page_header, render_metric_card, render_section_header, render_empty_state
from utils.formatting import format_number


def show_modeling() -> None:
    """Render the AutoML Modeling page."""
    render_page_header(
        title="AutoML Modeling",
        description="Automated machine learning pipeline: candidate target suggestion, ML readiness report, model benchmark leaderboard, and predictive feature importance rankings.",
        icon="⚙️",
    )

    if not has_dataset():
        render_empty_state(
            title="No Dataset Loaded for AutoML",
            description="Please upload a dataset on the Upload page to build machine learning models.",
            icon="⚙️",
        )
        return

    df = get_dataset()
    dataset_name = get_dataset_name() or "dataset"

    if df is None:
        st.error("Unable to retrieve dataset from session state.")
        return

    # ── Target & Task Setup ──
    render_section_header("ML Problem Setup & Target Selection", icon="🎯")
    candidate_targets = MLPlanner.suggest_target_columns(df)

    if not candidate_targets:
        st.warning("⚠️ No suitable target columns found in dataset.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        target_col = st.selectbox("Select Target Variable (Y):", options=candidate_targets, index=0)

    with col2:
        detected_task = MLPlanner.detect_task_type(df, target_col)
        task_options = [
            MLTaskType.BINARY_CLASSIFICATION,
            MLTaskType.MULTICLASS_CLASSIFICATION,
            MLTaskType.REGRESSION,
        ]
        task_type = st.selectbox(
            "Task Type:",
            options=task_options,
            index=task_options.index(detected_task),
            format_func=lambda x: x.value.replace("_", " ").title(),
        )

    with col3:
        test_split = st.slider("Test Split Size:", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    # ── ML Readiness Report Preview ──
    readiness = MLPlanner.generate_ml_readiness_report(df, target_col)
    t_qual = readiness["target_quality"]
    f_anal = readiness["feature_analysis"]

    with st.expander("📋 View ML Readiness & Feature Quality Report", expanded=False):
        st.markdown("### Target Variable Inspection")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Total Rows", f"{readiness['total_rows']:,}")
        with m2:
            render_metric_card("Valid Target Rows", f"{t_qual['valid_count']:,}")
        with m3:
            render_metric_card("Missing Target Rows", f"{t_qual['missing_count']:,}")
        with m4:
            render_metric_card("Target Unique Values", f"{t_qual['unique_count']:,}")

        if not t_qual["is_valid"]:
            st.error(f"❌ **Target Unfit for Modeling:** {t_qual['reason']}")

        st.markdown("### Predictive Feature Categorization")
        st.markdown(
            f"- **Usable Predictive Features:** `{f_anal['usable_features_count']}`\n"
            f"  - Numeric features: `{len(f_anal['numeric_cols'])}` (`{', '.join(f_anal['numeric_cols']) if f_anal['numeric_cols'] else 'None'}`)\n"
            f"  - Categorical features: `{len(f_anal['categorical_cols'])}` (`{', '.join(f_anal['categorical_cols']) if f_anal['categorical_cols'] else 'None'}`)\n"
            f"  - Datetime features: `{len(f_anal['datetime_cols'])}` (`{', '.join(f_anal['datetime_cols']) if f_anal['datetime_cols'] else 'None'}`)\n"
            f"  - Free-text features: `{len(f_anal['text_cols'])}` (`{', '.join(f_anal['text_cols']) if f_anal['text_cols'] else 'None'}`)"
        )

        if f_anal["dropped_features"]:
            st.markdown("#### Excluded Features (Identifiers & Zero-Variance)")
            dropped_df = pd.DataFrame([
                {"Column": k, "Exclusion Reason": v} for k, v in f_anal["dropped_features"].items()
            ])
            st.dataframe(dropped_df, use_container_width=True, hide_index=True)

    # ── Train Button ──
    if st.button("🚀 Train & Benchmark Models", type="primary", use_container_width=True):
        try:
            with st.spinner(f"Training models for target '{target_col}'..."):
                summary = AutoMLTrainer.train(
                    df=df,
                    target_column=target_col,
                    task_type=task_type,
                    test_size=test_split,
                )
            st.session_state["automl_summary"] = summary
            if summary.leaderboard:
                st.toast(f"✅ Trained and benchmarked {len(summary.leaderboard)} models!")
            else:
                st.error("Training aborted due to dataset or target limitations.")
        except Exception as e:
            logger.exception("AutoML Training failed: {}", str(e))
            st.error(f"❌ Training failed: {str(e)}")

    st.divider()

    # ── Leaderboard & Results ──
    summary: Optional[AutoMLSummary] = st.session_state.get("automl_summary")
    if summary and summary.target_column == target_col:
        if summary.warnings:
            for w in summary.warnings:
                st.warning(f"⚠️ **ML Warning:** {w}")

        if summary.leaderboard:
            render_section_header("Model Benchmark Leaderboard", icon="🏆")
            st.markdown(
                f"**Top Performing Algorithm:** ⭐ `{summary.best_model_name}` | "
                f"**Baseline Model:** 🎯 `{summary.baseline_model_name}` ({summary.baseline_metric_value:.4f})"
            )

            # Leaderboard Table
            board_rows = []
            for i, res in enumerate(summary.leaderboard, 1):
                role = "🎯 Baseline" if res.is_baseline else ("⭐ Best" if i == 1 else "Model")
                row_dict = {
                    "Rank": f"#{i}",
                    "Role": role,
                    "Model": res.model_name,
                    f"Primary Metric ({res.primary_metric_name})": f"{res.primary_metric_value:.4f}",
                    "Train Score": f"{res.train_score:.4f}",
                    "CV Mean": f"{res.cv_score_mean:.4f} ± {res.cv_score_std:.4f}" if res.cv_score_mean else "N/A",
                    "Fit Time": f"{res.fit_time_seconds:.3f}s",
                }
                for k, v in res.metrics.items():
                    if k not in (res.primary_metric_name, "CV R² Mean", "CV R² Std", "CV Accuracy Mean"):
                        row_dict[k] = f"{v:.4f}"
                board_rows.append(row_dict)

            board_df = pd.DataFrame(board_rows)
            st.dataframe(board_df, use_container_width=True, hide_index=True)

            # Leaderboard Bar Chart
            lead_chart_df = pd.DataFrame([
                {
                    "model_name": res.model_name,
                    "primary_metric_value": res.primary_metric_value,
                    "is_baseline": "Baseline" if res.is_baseline else "Candidate Model",
                }
                for res in summary.leaderboard
            ])
            fig_bar = px.bar(
                lead_chart_df,
                x="model_name",
                y="primary_metric_value",
                color="is_baseline",
                color_discrete_map={"Candidate Model": "#3B82F6", "Baseline": "#64748B"},
                labels={"model_name": "Model", "primary_metric_value": summary.leaderboard[0].primary_metric_name},
                title=f"Model Performance vs Baseline ({summary.leaderboard[0].primary_metric_name})",
                template="plotly_dark",
            )
            fig_bar.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()

            # ── Feature Importances ──
            if summary.feature_importances:
                render_section_header("Predictive Feature Importance", icon="🔍")
                st.caption("ℹ️ Feature importance reflects predictive association and decision influence within the model, not direct causal effect.")
                imp_df = pd.DataFrame([
                    {"Feature": item.feature, "Importance": item.importance}
                    for item in summary.feature_importances[:20]
                ])
                fig_imp = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top Predictive Features",
                    color="Importance",
                    color_continuous_scale="Viridis",
                    template="plotly_dark",
                )
                fig_imp.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B", yaxis={"autorange": "reversed"})
                st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("👆 Click **'Train & Benchmark Models'** above to run automated modeling on your clean dataset.")
