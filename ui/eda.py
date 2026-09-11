"""Automated EDA & Visuals UI page for AutoDS AI Studio."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from core.state import has_dataset, get_dataset, get_dataset_name
from core.logging import logger
from services.eda.service import EDAService
from ui.theme import render_page_header, render_metric_card, render_section_header, render_empty_state
from utils.dataframe import safe_numeric_columns, safe_categorical_columns


def show_eda() -> None:
    """Render the Automated EDA & Visuals page."""
    render_page_header(
        title="Automated EDA & Visual Insights",
        description="Comprehensive exploratory analysis: missingness profile, feature correlation matrix, distribution plots, and bivariate scatter graphs.",
        icon="📈",
    )

    if not has_dataset():
        render_empty_state(
            title="No Dataset Loaded for EDA",
            description="Please upload a CSV or Excel file on the Upload page to visualize data.",
            icon="📁",
        )
        return

    df = get_dataset()
    dataset_name = get_dataset_name() or "dataset"

    if df is None:
        st.error("Unable to retrieve dataset from session state.")
        return

    tab_overview, tab_corr, tab_dist, tab_scatter = st.tabs([
        "📋 Quick Profile",
        "🔥 Correlation Heatmap",
        "📈 Feature Distributions",
        "🌐 Bivariate Scatter & Boxplots",
    ])

    # ── Tab 1: Quick Profile ──
    with tab_overview:
        render_section_header("Data Matrix Summary", icon="📋")
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Numeric Features", str(len(safe_numeric_columns(df))), icon="🔢")
        with col2:
            render_metric_card("Categorical Features", str(len(safe_categorical_columns(df))), icon="🔤")
        with col3:
            render_metric_card("Total Missing Cells", str(int(df.isna().sum().sum())), icon="⚠️")

        render_section_header("Missing Data Distribution", icon="📊")
        null_counts = df.isna().sum()
        null_df = pd.DataFrame({
            "Column": list(null_counts.index),
            "Missing Count": null_counts.to_numpy(),
            "Missing Percentage": (null_counts.to_numpy() / len(df)) * 100,
        }).sort_values(by="Missing Count", ascending=False)

        if null_df["Missing Count"].sum() > 0:
            fig_null = px.bar(
                null_df[null_df["Missing Count"] > 0],
                x="Column",
                y="Missing Percentage",
                title="Missing Data Percentage by Column",
                labels={"Missing Percentage": "Missing %"},
                color="Missing Percentage",
                color_continuous_scale="Reds",
                template="plotly_dark",
            )
            fig_null.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B")
            st.plotly_chart(fig_null, use_container_width=True)
        else:
            st.success("✅ Clean dataset! Zero missing values detected across all columns.")

    # ── Tab 2: Correlation Heatmap ──
    with tab_corr:
        render_section_header("Pearson Feature Correlation Matrix", icon="🔥")
        corr_matrix = EDAService.get_correlation_matrix(df)
        if corr_matrix is not None and not corr_matrix.empty:
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Feature Correlation Heatmap",
                template="plotly_dark",
            )
            fig_corr.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Fewer than 2 numeric columns available for correlation analysis.")

    # ── Tab 3: Feature Distributions ──
    with tab_dist:
        render_section_header("Univariate Feature Distributions", icon="📈")
        selected_col = st.selectbox("Select Feature to Visualize:", options=list(df.columns), key="eda_dist_col")
        if selected_col:
            if pd.api.types.is_numeric_dtype(df[selected_col]):
                col_data = df[selected_col].dropna()
                fig_hist = px.histogram(
                    col_data,
                    x=col_data.name,
                    marginal="box",
                    title=f"Distribution of '{selected_col}'",
                    color_discrete_sequence=["#6366F1"],
                    template="plotly_dark",
                )
                fig_hist.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                top_cats = df[selected_col].value_counts().head(25)
                fig_bar = px.bar(
                    x=top_cats.index.astype(str),
                    y=top_cats.values,
                    labels={"x": selected_col, "y": "Count"},
                    title=f"Top Categories in '{selected_col}'",
                    color=top_cats.values,
                    color_continuous_scale="Viridis",
                    template="plotly_dark",
                )
                fig_bar.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B")
                st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tab 4: Bivariate Analysis ──
    with tab_scatter:
        render_section_header("Bivariate Relationships & Scatter Analysis", icon="🌐")
        num_cols = safe_numeric_columns(df)
        if len(num_cols) >= 2:
            col_x, col_y, col_color = st.columns(3)
            with col_x:
                x_axis = st.selectbox("X-Axis:", options=num_cols, index=0, key="scatter_x")
            with col_y:
                y_axis = st.selectbox("Y-Axis:", options=num_cols, index=min(1, len(num_cols) - 1), key="scatter_y")
            with col_color:
                cat_options = ["None"] + [c for c in df.columns if df[c].nunique() <= 10]
                color_by = st.selectbox("Color By (Optional):", options=cat_options, index=0, key="scatter_color")

            color_arg = color_by if color_by != "None" else None

            # Prepare clean numeric data for scatter plot
            valid_cols = [x_axis, y_axis]
            if color_arg and color_arg not in valid_cols:
                valid_cols.append(color_arg)

            clean_scatter_df = df[valid_cols].copy()
            clean_scatter_df = clean_scatter_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_axis, y_axis])

            enable_trendline = color_arg is None and len(clean_scatter_df) >= 5

            try:
                fig_scatter = px.scatter(
                    clean_scatter_df,
                    x=x_axis,
                    y=y_axis,
                    color=color_arg,
                    title=f"'{y_axis}' vs '{x_axis}'",
                    trendline="ols" if enable_trendline else None,
                    opacity=0.75,
                    template="plotly_dark",
                )
                fig_scatter.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B")
            except Exception as exc:
                logger.warning("OLS trendline generation failed for '{}' vs '{}': {}", y_axis, x_axis, str(exc))
                st.caption("ℹ️ Trendline unavailable for the selected columns.")
                fig_scatter = px.scatter(
                    clean_scatter_df,
                    x=x_axis,
                    y=y_axis,
                    color=color_arg,
                    title=f"'{y_axis}' vs '{x_axis}'",
                    trendline=None,
                    opacity=0.75,
                    template="plotly_dark",
                )
                fig_scatter.update_layout(paper_bgcolor="#1E293B", plot_bgcolor="#1E293B")

            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("At least 2 numeric columns are required for bivariate scatter analysis.")
