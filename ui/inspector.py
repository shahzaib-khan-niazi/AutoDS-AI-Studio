"""Inspector UI page for AutoDS AI Studio.

Displays dataset health, column details, statistics, and quality assessment.
"""

import streamlit as st
import pandas as pd

from core.state import has_dataset, get_dataset
from core.logging import logger
from services.inspector.service import InspectorService
from ui.theme import render_page_header, render_metric_card, render_section_header, render_empty_state
from utils.formatting import format_number, format_percentage


def show_inspector() -> None:
    """Render the Dataset Inspector page."""
    render_page_header(
        title="Dataset Inspector",
        description="Detailed structural analysis, data type verification, missingness breakdown, statistical distribution, and quality scoring.",
        icon="🔍",
    )

    if not has_dataset():
        render_empty_state(
            title="No Dataset Loaded for Inspection",
            description="Please upload a CSV or Excel file on the Upload page to run dataset inspection.",
            icon="📁",
        )
        return

    df = get_dataset()
    if df is None:
        st.error("Unable to retrieve dataset from session state.")
        return

    # Run inspection (cache in session state)
    report = st.session_state.get("inspection_report")
    if report is None:
        try:
            with st.spinner("Inspecting dataset structure & quality..."):
                report = InspectorService.inspect(df)
            st.session_state["inspection_report"] = report
        except Exception as e:
            st.error("❌ Failed to inspect dataset.")
            logger.exception("Inspection error: {}", str(e))
            return

    # Re-inspect button
    col_hdr, col_btn = st.columns([4, 1])
    with col_btn:
        if st.button("🔄 Re-inspect", use_container_width=True):
            try:
                with st.spinner("Re-inspecting dataset..."):
                    report = InspectorService.inspect(df)
                st.session_state["inspection_report"] = report
                st.rerun()
            except Exception as e:
                st.error(f"❌ Re-inspection failed: {e}")
                return

    # ── Dataset Health Metrics ──
    render_section_header("Dataset Health Overview", icon="📊")
    quality = report.quality

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score_val = f"{quality.quality_score:.0%}"
        delta_col = "success" if quality.quality_score >= 0.8 else ("warning" if quality.quality_score >= 0.6 else "danger")
        render_metric_card("Quality Score", score_val, delta="Calculated Quality", delta_color=delta_col, icon="🛡️")
    with col2:
        render_metric_card("Row Count", format_number(report.row_count), icon="📐")
    with col3:
        render_metric_card("Column Count", format_number(report.column_count), icon="📊")
    with col4:
        render_metric_card("Missing Cells", format_number(quality.missing_cells), icon="⚠️")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        render_metric_card("Duplicate Rows", format_number(quality.duplicate_rows), icon="📋")
    with col6:
        render_metric_card("Empty Rows", format_number(quality.empty_rows), icon="🚫")
    with col7:
        render_metric_card("Empty Columns", format_number(len(quality.empty_columns)), icon="🗑️")
    with col8:
        render_metric_card("Mixed-Type Cols", format_number(len(quality.mixed_type_columns)), icon="🔀")

    # Quality notes
    if quality.quality_notes:
        with st.expander("📝 Quality Score Breakdown & Deductions"):
            for note in quality.quality_notes:
                st.markdown(f"• {note}")

    st.divider()

    # ── Column Details ──
    render_section_header("Column Details & Schema", icon="📋")
    if report.column_details:
        col_data: dict[str, list[object]] = {
            "Column": [],
            "Type": [],
            "Non-Null": [],
            "Null %": [],
            "Unique": [],
            "Flags": [],
        }
        for detail in report.column_details:
            col_data["Column"].append(detail.name)
            col_data["Type"].append(detail.dtype)
            col_data["Non-Null"].append(detail.non_null_count)
            col_data["Null %"].append(format_percentage(detail.null_percentage))
            col_data["Unique"].append(detail.unique_count)

            flags: list[str] = []
            if detail.is_potential_id:
                flags.append("🔑 ID")
            if detail.is_constant:
                flags.append("📌 Constant")
            if detail.is_mixed_type:
                flags.append("⚠️ Mixed")
            col_data["Flags"].append(" ".join(flags) if flags else "")

        st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)

    st.divider()

    # ── Statistics ──
    render_section_header("Statistical Breakdown", icon="📈")

    tab_numeric, tab_categorical, tab_datetime = st.tabs(
        ["🔢 Numeric Features", "🔤 Categorical Features", "📅 Datetime Features"]
    )

    with tab_numeric:
        if report.numeric_stats:
            num_data: dict[str, list[object]] = {
                "Column": [],
                "Count": [],
                "Mean": [],
                "Median": [],
                "Std": [],
                "Min": [],
                "Max": [],
                "Skewness": [],
            }
            for s in report.numeric_stats:
                num_data["Column"].append(s.column)
                num_data["Count"].append(s.count)
                num_data["Mean"].append(f"{s.mean:.4f}")
                num_data["Median"].append(f"{s.median:.4f}")
                num_data["Std"].append(f"{s.std:.4f}")
                num_data["Min"].append(f"{s.min_value:.4f}")
                num_data["Max"].append(f"{s.max_value:.4f}")
                num_data["Skewness"].append(f"{s.skewness:.4f}")
            st.dataframe(pd.DataFrame(num_data), use_container_width=True, hide_index=True)
        else:
            st.info("No numeric columns detected in this dataset.")

    with tab_categorical:
        if report.categorical_stats:
            cat_data: dict[str, list[object]] = {
                "Column": [],
                "Count": [],
                "Unique": [],
                "Top Value": [],
                "Top Frequency": [],
            }
            for s in report.categorical_stats:
                cat_data["Column"].append(s.column)
                cat_data["Count"].append(s.count)
                cat_data["Unique"].append(s.unique)
                cat_data["Top Value"].append(s.top)
                cat_data["Top Frequency"].append(s.top_frequency)
            st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)
        else:
            st.info("No categorical columns detected in this dataset.")

    with tab_datetime:
        if report.datetime_stats:
            dt_data: dict[str, list[object]] = {
                "Column": [],
                "Count": [],
                "Min Date": [],
                "Max Date": [],
            }
            for s in report.datetime_stats:
                dt_data["Column"].append(s.column)
                dt_data["Count"].append(s.count)
                dt_data["Min Date"].append(s.min_date)
                dt_data["Max Date"].append(s.max_date)
            st.dataframe(pd.DataFrame(dt_data), use_container_width=True, hide_index=True)
        else:
            st.info("No datetime columns detected in this dataset.")

    st.divider()

    # ── Issues Summary ──
    render_section_header("Data Quality Issues Audit", icon="⚠️")
    issues_found = False

    if quality.suspicious_column_names:
        issues_found = True
        st.warning(f"**Suspicious column names:** {', '.join(quality.suspicious_column_names)}")

    if quality.empty_columns:
        issues_found = True
        st.warning(f"**Empty columns:** {', '.join(quality.empty_columns)}")

    if quality.constant_columns:
        issues_found = True
        st.info(f"**Constant columns:** {', '.join(quality.constant_columns)}")

    if quality.mixed_type_columns:
        issues_found = True
        st.warning(f"**Mixed-type columns:** {', '.join(quality.mixed_type_columns)}")

    if quality.high_cardinality_columns:
        issues_found = True
        st.info(f"**High-cardinality columns:** {', '.join(quality.high_cardinality_columns)}")

    if quality.possible_id_columns:
        issues_found = True
        st.info(f"**Possible ID columns:** {', '.join(quality.possible_id_columns)}")

    if not issues_found:
        st.success("✅ Clean dataset health! No significant data quality issues detected.")
