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
        description="Automated machine learning pipeline: candidate target suggestion, model benchmark leaderboard, and predictive feature importance rankings.",
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
            st.toast(f"✅ Trained and benchmarked {len(summary.leaderboard)} models!")
        except Exception as e:
            logger.exception("AutoML Training failed: {}", str(e))
            st.error(f"❌ Training failed: {str(e)}")

    st.divider()

    # ── Leaderboard & Results ──
    summary: Optional[AutoMLSummary] = st.session_state.get("automl_summary")
    if summary and summary.target_column == target_col:
        render_section_header("Model Benchmark Leaderboard", icon="🏆")
        st.markdown(f"**Top Performing Algorithm:** ⭐ `{summary.best_model_name}`")

        # Leaderboard Table
        board_rows = []
        for i, res in enumerate(summary.leaderboard, 1):
            row_dict = {
                "Rank": f"#{i}",
                "Model": res.model_name,
                f"Primary Metric ({res.primary_metric_name})": f"{res.primary_metric_value:.4f}",
                "Train Score": f"{res.train_score:.4f}",
                "Fit Time": f"{res.fit_time_seconds:.3f}s",
            }
            for k, v in res.metrics.items():
                if k != res.primary_metric_name:
                    row_dict[k] = f"{v:.4f}"
            board_rows.append(row_dict)

        board_df = pd.DataFrame(board_rows)
        st.dataframe(board_df, use_container_width=True, hide_index=True)

        # Leaderboard Bar Chart
        lead_chart_df = pd.DataFrame([
            {
                "model_name": res.model_name,
                "primary_metric_value": res.primary_metric_value,
            }
            for res in summary.leaderboard
        ])
        fig_bar = px.bar(
            lead_chart_df,
            x="model_name",
            y="primary_metric_value",
            color="primary_metric_value",
            color_continuous_scale="Blues",
            labels={"model_name": "Model", "primary_metric_value": summary.leaderboard[0].primary_metric_name},
            title=f"Model Performance Comparison ({summary.leaderboard[0].primary_metric_name})",
            template="plotly_dark",
        )
        fig_bar.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # ── Feature Importances ──
        if summary.feature_importances:
            render_section_header("Predictive Feature Importance", icon="🔍")
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
