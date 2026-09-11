"""Dataset Repair UI page for AutoDS AI Studio.

Displays:
1. ⚡ Autonomous 1-Click Auto-Clean & Preprocess (End-to-End Autonomous Agent)
2. 🤖 AI Guided Repair Planner
3. 🔍 Deterministic Quick Repairs
4. 🛠️ Granular Repair & Imputation Tools (with Categorical & Datatype Normalization Mapping Preview)
"""

import streamlit as st
import pandas as pd
from typing import Optional

from core.config import config
from core.llm.config import llm_config
from core.state import (
    has_dataset,
    get_dataset,
    get_original_dataset,
    get_dataset_name,
    update_dataset,
)
from core.logging import logger
from core.exceptions import AIServiceError
from models.repair import RepairAction, RepairOperation, ConfidenceLevel, IssueTaxonomy
from services.repair.service import RepairService
from services.repair.standardize import get_proposed_standardizations
from services.repair.dates import analyze_date_column, DateConvention, format_date_series, DEFAULT_DISPLAY_FORMAT
from services.repair.outliers import detect_impossible_values, handle_outliers
from services.repair.structural import detect_multi_value_cells
from services.repair.columns import (
    validate_column_rename,
    rename_single_column,
    standardize_name,
    detect_duplicate_columns,
    resolve_duplicate_columns,
    detect_unnamed_columns,
    remove_column,
)

from services.pipeline.autonomous import AutonomousPipelineService, AutonomousPipelineResult
from services.ai.planner import AIPlanner
from ui.theme import render_page_header, render_metric_card, render_section_header, render_empty_state
from utils.dataframe import make_arrow_safe_preview, safe_numeric_columns
from utils.formatting import format_number


def show_repair() -> None:
    """Render the Dataset Repair page."""
    render_page_header(
        title="Autonomous Clean & Repair",
        description="End-to-end dataset preprocessing: autonomous multi-issue cleaning, AI reasoning repair planner, whitespace stripping, date standardization, and categorical inconsistency normalization.",
        icon="🛠️",
    )

    if not has_dataset():
        render_empty_state(
            title="No Dataset Loaded for Repair",
            description="Please upload a CSV or Excel file on the Upload page to run dataset repair.",
            icon="📁",
        )
        return

    df = get_dataset()
    original_df = get_original_dataset()
    dataset_name = get_dataset_name() or "dataset"

    if df is None or original_df is None:
        st.error("Unable to retrieve dataset from session state.")
        return

    # Ensure repair history exists in session state
    if "repair_records" not in st.session_state or st.session_state["repair_records"] is None:
        st.session_state["repair_records"] = []

    # ── Dataset Overview ──
    render_section_header("Active Dataset Dimensions", icon="📊")
    col1, col2 = st.columns(2)
    with col1:
        render_metric_card("Original Uploaded", f"{format_number(len(original_df))} rows × {format_number(len(original_df.columns))} cols", icon="📥")
    with col2:
        render_metric_card("Current Working Dataset", f"{format_number(len(df))} rows × {format_number(len(df.columns))} cols", icon="⚙️")

    st.divider()

    # ── HERO FEATURE 1: ⚡ Autonomous 1-Click Auto-Clean & Preprocess ──
    st.markdown(
        """
        <div class="saas-card" style="border-left: 4px solid #6366F1;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">⚡ Autonomous 1-Click Auto-Preprocessing</div>
            <div style="font-size: 0.9rem; color: #94A3B8; margin-bottom: 16px; line-height: 1.5;">
                The autonomous pipeline automatically inspects your dataset, identifies all data quality and structural problems, standardizes inconsistent text/dates/dtypes, formulates a validated repair plan, executes transformations via safe Python operations, and verifies final quality scores.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_auto_btn, col_auto_opt = st.columns([2, 2])
    with col_auto_btn:
        run_auto = st.button("🚀 Run Autonomous Preprocessing Pipeline", type="primary", use_container_width=True)
    with col_auto_opt:
        use_ai_opt = st.checkbox("Leverage AI for reasoning & planning", value=config.ai.is_configured, disabled=not config.ai.is_configured)

    if run_auto:
        try:
            with st.spinner("🤖 Autonomous Agent analyzing dataset, repairing issues, and validating..."):
                cleaned_df, pipeline_res = AutonomousPipelineService.run_auto_preprocess(
                    df=df,
                    use_ai_if_available=use_ai_opt,
                )

            if pipeline_res.success:
                update_dataset(cleaned_df)
                st.session_state["inspection_report"] = None
                st.session_state["structure_report"] = None
                st.session_state["repair_records"].extend(pipeline_res.records)
                st.session_state["autonomous_result"] = pipeline_res

                st.toast(f"🎉 Autonomous Preprocessing completed in {pipeline_res.execution_time_seconds}s!")
                st.rerun()
            else:
                st.error("Autonomous pipeline encountered errors during validation.")
        except Exception as e:
            logger.exception("Autonomous pipeline failed: {}", str(e))
            st.error(f"❌ Autonomous pipeline failed: {str(e)}")

    # Show latest autonomous result summary if available
    auto_res: Optional[AutonomousPipelineResult] = st.session_state.get("autonomous_result")
    if auto_res:
        render_section_header("Autonomous Audit Report", icon="📋")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            render_metric_card("Initial Quality", f"{auto_res.initial_quality_score:.0%}")
        with col_s2:
            render_metric_card("Final Clean Quality", f"{auto_res.final_quality_score:.0%}", delta=f"+{auto_res.quality_improvement:.0%}", delta_color="success")
        with col_s3:
            render_metric_card("Actions Applied", str(len(auto_res.actions_executed)))
        with col_s4:
            render_metric_card("Runtime", f"{auto_res.execution_time_seconds}s")

        with st.expander("🔍 Detailed Transformation Log", expanded=True):
            for bullet in auto_res.summary_bullet_points:
                st.markdown(f"• {bullet}")

    st.divider()

    # ── HERO FEATURE 2: 🤖 AI Guided Repair Planner ──
    render_section_header("AI Guided Repair Planner", icon="🤖")
    st.caption("Ask the AI to inspect schema profiles and formulate custom multi-step repair plans.")

    if not llm_config.is_configured:
        st.info("💡 Configure `OPENROUTER_API_KEY` in `.env` to enable AI Repair Planning.")
    else:
        col_ai1, _ = st.columns([2, 3])
        with col_ai1:
            if st.button("🧠 Generate Custom AI Plan", key="gen_ai_plan_btn", use_container_width=True):
                try:
                    with st.spinner("AI analyzing profile and creating repair plan..."):
                        inspection = st.session_state.get("inspection_report")
                        structure = st.session_state.get("structure_report")
                        ai_plan = AIPlanner.generate_repair_plan(df, inspection=inspection, structure=structure)
                    st.session_state["ai_repair_plan"] = ai_plan
                except AIServiceError as e:
                    st.error(f"❌ AI Error: {e.message}")
                    if e.details:
                        st.caption(e.details)
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

        ai_plan = st.session_state.get("ai_repair_plan")
        if ai_plan:
            st.markdown(f"**AI Decision:** `{ai_plan.decision}` *(Confidence: {ai_plan.confidence:.0%})*")
            if ai_plan.reasoning_summary:
                st.info(f"**Reasoning:** {ai_plan.reasoning_summary}")

            if ai_plan.actions:
                st.markdown("**Proposed Validated Actions:**")
                for act in ai_plan.actions:
                    target_str = f" on `{', '.join(act.target)}`" if act.target else ""
                    st.markdown(f"• `{act.operation}`{target_str}: {act.reason}")

                if st.button("🚀 Execute Validated AI Plan", type="primary"):
                    exec_actions = AIPlanner.convert_ai_actions_to_repair_actions(ai_plan)
                    if exec_actions:
                        _apply_repairs(df, exec_actions)
                        st.session_state["ai_repair_plan"] = None
                        st.rerun()
                    else:
                        st.warning("No executable actions in plan.")
            else:
                st.success("AI concluded no further repairs are necessary.")

            if ai_plan.warnings:
                for w in ai_plan.warnings:
                    st.warning(f"⚠️ {w}")

    st.divider()

    # ── HERO FEATURE 3: 🔍 Deterministic Quick Repairs & Impact Preview ──
    render_section_header("Deterministic Quick Repairs & Impact Preview", icon="🔍")
    suggested_actions = RepairService.auto_detect_repairs(df)

    if suggested_actions:
        preview = RepairService.generate_repair_preview(df, suggested_actions)

        st.markdown(
            f"Detected **{len(suggested_actions)} safe deterministic issue(s)**. "
            "Inspect preview impact metrics and itemized changes before applying."
        )

        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            render_metric_card("Rows Affected", str(preview["rows_affected"]), icon="📄")
        with col_p2:
            render_metric_card("Values Changed", str(preview["values_changed"]), icon="✏️")
        with col_p3:
            render_metric_card("Columns Changed", str(len(preview["columns_changed"])), icon="📊")
        with col_p4:
            loss = preview["potential_data_loss"]
            render_metric_card("Potential Data Loss", str(loss), icon="⚠️", delta_color="inverse" if loss > 0 else "off")

        if preview["potential_data_loss"] > 0:
            st.warning(f"⚠️ Potential data loss detected: {preview['potential_data_loss']} cell(s) could become null under proposed actions.")

        if preview["datatype_changes"]:
            dt_bullets = [f"`{col}`: {change['from']} → {change['to']}" for col, change in preview["datatype_changes"].items()]
            st.info(f"**Datatype upgrades:** {', '.join(dt_bullets)}")

        # Itemized proposal table
        if preview["preview_table"]:
            with st.expander("👁️ Detailed Proposed Actions Table", expanded=True):
                st.dataframe(pd.DataFrame(preview["preview_table"]), use_container_width=True, hide_index=True)

        if st.button("⚡ Apply All Safe Repairs", type="primary", use_container_width=True, key="btn_apply_all_safe"):
            _apply_repairs(df, suggested_actions)
            st.rerun()
    else:
        st.success("✅ No standard deterministic quality issues detected in current dataset.")

    st.divider()

    # ── HERO FEATURE 4: 🛠️ Granular Repair & Imputation Tools ──
    render_section_header("Granular Repair & Imputation Tools", icon="🛠️")
    tab_norm, tab_clean, tab_impute, tab_types, tab_struct, tab_rename, tab_reset = st.tabs([
        "🔤 Categorical Normalization",
        "🧹 Basic Cleaning",
        "🩹 Missing Value Imputation",
        "🔢 Type Conversion & Outliers",
        "📐 Reshape / Structural",
        "🏷️ Rename Column",
        "🔄 Reset",
    ])

    # ── Tab 1: Categorical Normalization Preview ──
    with tab_norm:
        st.markdown("##### 🔤 Generalized Semantic & Categorical Normalization")
        st.caption(
            "Algorithmic normalization across formatting variants, typos, frequency-asymmetric misspellings, "
            "and dynamic abbreviations without domain hardcoding."
        )

        proposals = get_proposed_standardizations(df)
        if proposals:
            safe_count = sum(1 for p in proposals if p.get("confidence", 0) >= 0.95)
            high_count = sum(1 for p in proposals if 0.85 <= p.get("confidence", 0) < 0.95)
            review_count = sum(1 for p in proposals if p.get("confidence", 0) < 0.85)
            ambig_proposals = [p for p in proposals if p.get("is_ambiguous")]

            # Metric overview
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Total Mappings", str(len(proposals)), icon="🔤")
            with m2:
                render_metric_card("Safe (≥95%)", str(safe_count), icon="🟢")
            with m3:
                render_metric_card("High (85-94%)", str(high_count), icon="🔵")
            with m4:
                render_metric_card("Review (<85%)", str(review_count), icon="🟠")

            if ambig_proposals:
                st.warning(
                    f"⚠️ **{len(ambig_proposals)} Ambiguous abbreviation/variation(s) detected!** "
                    "These have multiple valid interpretations and require user review."
                )
                with st.expander("🔍 View Ambiguous Abbreviations & Candidate Meanings", expanded=False):
                    for ap in ambig_proposals:
                        cand_str = ", ".join(f"`{c}`" for c in ap.get("candidate_meanings", []))
                        st.markdown(f"• **`{ap['column']}`**: `{ap['original_value']}` could refer to {cand_str}")

            # Format preview dataframe with visual indicators
            preview_rows = []
            for p in proposals:
                conf = p.get("confidence", 0.0)
                if conf >= 0.95:
                    tier_str = f"🟢 Safe ({conf:.0%})"
                elif conf >= 0.85:
                    tier_str = f"🔵 High ({conf:.0%})"
                elif conf >= 0.70:
                    tier_str = f"🟠 Review ({conf:.0%})"
                else:
                    tier_str = f"🔴 Low ({conf:.0%})"

                issue_type = p.get("issue_type", "FORMAT")
                ambig_mark = " ⚠️ Ambiguous" if p.get("is_ambiguous") else ""

                preview_rows.append({
                    "Column": p["column"],
                    "Original Value": p["original_value"],
                    "Normalized Value": p["normalized_value"] if p["normalized_value"] is not None else "<NaN (Missing)>",
                    "Tier & Confidence": tier_str,
                    "Taxonomy": f"{issue_type}{ambig_mark}",
                    "Method": p["method"],
                    "Reason": p["reason"],
                    "Affected Rows": p["affected_rows"],
                })

            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

            btn_c1, btn_c2, _ = st.columns([2, 2.5, 2])
            with btn_c1:
                if st.button("✨ Apply All Normalizations", type="primary", key="btn_apply_all_norm"):
                    action = RepairAction(
                        operation=RepairOperation.STANDARDIZE_VALUES,
                        reason=f"Normalized {len(proposals)} inconsistent value mapping(s)",
                    )
                    _apply_repairs(df, [action])
                    st.rerun()

            with btn_c2:
                if st.button("🛡️ Apply High-Confidence Only (≥85%)", key="btn_apply_high_conf_norm"):
                    action = RepairAction(
                        operation=RepairOperation.STANDARDIZE_VALUES,
                        parameters={"min_confidence": 0.85},
                        reason=f"Normalized high-confidence (≥85%) value mappings",
                    )
                    _apply_repairs(df, [action])
                    st.rerun()
        else:
            st.success("✅ No categorical value inconsistencies detected by deterministic engine.")

    with tab_clean:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### Duplicate Rows")
            dup_count = int(df.duplicated().sum())
            st.write(f"Current duplicates: **{dup_count}**")
            if st.button("Remove Duplicate Rows", disabled=dup_count == 0):
                action = RepairAction(
                    operation=RepairOperation.REMOVE_DUPLICATE_ROWS,
                    reason="Manual removal of duplicate rows",
                )
                _apply_repairs(df, [action])
                st.rerun()

            st.markdown("##### Empty Rows")
            empty_rows = df.isna().all(axis=1).sum()
            st.write(f"Completely empty rows: **{empty_rows}**")
            if st.button("Drop Empty Rows", disabled=empty_rows == 0):
                action = RepairAction(
                    operation=RepairOperation.DROP_EMPTY_ROWS,
                    reason="Manual removal of empty rows",
                )
                _apply_repairs(df, [action])
                st.rerun()

            st.markdown("##### Cell Whitespace Stripping")
            st.write("Strips leading/trailing spaces & collapses internal spaces in text cells.")
            if st.button("🧹 Strip Cell Whitespace"):
                action = RepairAction(
                    operation=RepairOperation.STRIP_WHITESPACE,
                    reason="Manual whitespace stripping from text cells",
                )
                _apply_repairs(df, [action])
                st.rerun()

        with col_c2:
            st.markdown("##### Column Names")
            st.write("Fixes unnamed columns, strips whitespace, resolves duplicates.")
            if st.button("Sanitize Column Names"):
                action = RepairAction(
                    operation=RepairOperation.RENAME_COLUMNS,
                    reason="Manual sanitization of column names",
                )
                _apply_repairs(df, [action])
                st.rerun()

            st.markdown("##### Empty Columns")
            empty_cols = [c for c in df.columns if df[c].isna().all()]
            st.write(f"Empty columns: **{len(empty_cols)}** ({', '.join(map(str, empty_cols)) if empty_cols else 'None'})")
            if st.button("Drop Empty Columns", disabled=len(empty_cols) == 0):
                action = RepairAction(
                    operation=RepairOperation.DROP_EMPTY_COLUMNS,
                    target=empty_cols,
                    reason="Manual removal of empty columns",
                )
                _apply_repairs(df, [action])
                st.rerun()

            st.markdown("##### Categorical Standardization")
            st.write("Normalizes case variants, spelling differences & city/entity abbreviations.")
            std_target = st.multiselect(
                "Select columns to standardize (leave empty for auto-detect):",
                options=[c for c in df.columns if df[c].dtype == "object"],
                key="std_val_target",
            )
            if st.button("🔤 Standardize Categorical Values"):
                action = RepairAction(
                    operation=RepairOperation.STANDARDIZE_VALUES,
                    target=std_target if std_target else [],
                    reason="Manual standardization of inconsistent categorical values",
                )
                _apply_repairs(df, [action])
                st.rerun()

    with tab_impute:
        st.markdown("##### Missing Value Strategies")
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            impute_strat = st.selectbox(
                "Imputation Strategy:",
                options=["auto", "median", "mean", "mode", "ffill", "bfill", "constant"],
                help="'auto' uses median for numeric, mode for categorical, and ffill for datetime.",
            )
            custom_fill = None
            if impute_strat == "constant":
                custom_fill = st.text_input("Custom Constant Value:", value="0")

        with col_i2:
            cols_to_impute = st.multiselect(
                "Columns to Impute (leave empty for all columns with missing values):",
                options=[c for c in df.columns if df[c].isna().any()],
            )

        if st.button("🩹 Execute Imputation", type="primary"):
            action = RepairAction(
                operation=RepairOperation.FILL_MISSING,
                target=cols_to_impute if cols_to_impute else [],
                parameters={"strategy": impute_strat, "custom_value": custom_fill},
                reason=f"Impute missing values using {impute_strat} strategy",
            )
            _apply_repairs(df, [action])
            st.rerun()

    with tab_types:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("##### Smart Auto Dtype Detection")
            st.write("Automatically converts text columns that are actually numbers, dates, or booleans.")
            if st.button("🪄 Auto-Detect & Fix All Column Dtypes", type="primary"):
                action = RepairAction(
                    operation=RepairOperation.AUTO_DTYPES,
                    reason="Manual execution of smart automatic dtype detection",
                )
                _apply_repairs(df, [action])
                st.rerun()

            st.markdown("---")
            st.markdown("##### Manual Type Conversion")
            numeric_target = st.multiselect(
                "Select columns to convert to Numeric:",
                options=list(df.columns),
                key="num_conv_target",
            )
            if st.button("Convert to Numeric", disabled=len(numeric_target) == 0):
                action = RepairAction(
                    operation=RepairOperation.CONVERT_TYPES,
                    target=numeric_target,
                    parameters={"target_type": "numeric"},
                    reason=f"Convert {len(numeric_target)} column(s) to numeric",
                )
                _apply_repairs(df, [action])
                st.rerun()

            dt_target = st.multiselect(
                "Select columns to convert to Datetime:",
                options=list(df.columns),
                key="dt_conv_target",
            )
            if st.button("Convert to Datetime", disabled=len(dt_target) == 0):
                action = RepairAction(
                    operation=RepairOperation.CONVERT_TYPES,
                    target=dt_target,
                    parameters={"target_type": "datetime"},
                    reason=f"Convert {len(dt_target)} column(s) to datetime",
                )
                _apply_repairs(df, [action])
                st.rerun()

            st.markdown("---")
            st.markdown("##### 📅 Date Standardization — Output: `DD-MM-YY`")
            st.caption(
                "Parses dates from any input format, determines the column convention via **global evidence**, "
                "and outputs a standardized `DD-MM-YY` string column alongside the internal `datetime64` column."
            )

            # Detect candidate date columns (object dtype with date-like separators)
            date_candidates = []
            for col in df.columns:
                if df[col].dtype == "object":
                    sample = df[col].dropna().astype(str).head(30)
                    if any(any(c in v for c in ["-", "/", ".", ":"]) for v in sample):
                        date_candidates.append(col)

            if date_candidates:
                selected_dt_col = st.selectbox(
                    "Inspect Date Candidate Column:",
                    options=date_candidates,
                    key="date_inspect_col",
                )
                if selected_dt_col:
                    analysis = analyze_date_column(df[selected_dt_col])

                    # ── Header info: target format ──
                    fi1, fi2, fi3 = st.columns([2, 1, 1])
                    fi1.markdown(f"**Detected Convention:** `{analysis.inferred_convention}`")
                    fi2.markdown(f"**Target Format:** `DD-MM-YY`")
                    fi3.markdown(f"**Format String:** `{DEFAULT_DISPLAY_FORMAT}`")

                    # ── Metadata & Metric Cards ──
                    st.markdown("**Detected Type:** `Datetime` &nbsp;|&nbsp; **Standard Output Format:** `DD-MM-YY`")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Confidence", f"{analysis.confidence:.0%}")
                    mc2.metric("Valid Dates", f"{analysis.parsed_count}/{analysis.total_non_null}")
                    mc3.metric("Ambiguous Dates", str(analysis.ambiguous_count))
                    mc4.metric("Invalid Dates", str(analysis.invalid_count))

                    # ── Status alerts ──
                    if analysis.mixed_conventions_detected:
                        st.error(
                            "🔀 **MIXED DATE CONVENTIONS DETECTED** — The column contains both DMY and MDY "
                            "evidence. Enforce a convention manually below before applying."
                        )
                    elif analysis.is_ambiguous:
                        st.warning(
                            "⚠️ **AMBIGUOUS DATE COLUMN** — All day/month values ≤ 12; cannot determine "
                            "convention without external evidence. Ambiguous rows will be flagged, not guessed."
                        )
                    else:
                        st.success(
                            f"✅ Convention `{analysis.inferred_convention}` confirmed with "
                            f"`{analysis.confidence:.0%}` confidence. "
                            f"{analysis.ambiguous_count} ambiguous row(s) will be resolved via this convention."
                        )

                    if analysis.has_time:
                        st.info("⏱️ **Time components detected** — date portion extracted; time preserved internally.")
                    if analysis.has_timezone:
                        st.info("🌐 **Timezone offsets detected** — will be UTC-normalised in datetime64.")
                    if analysis.invalid_count > 0:
                        st.error(f"❌ **{analysis.invalid_count}** invalid / impossible date(s) — will become NaT:")
                        for inv in analysis.invalid_samples[:5]:
                            st.caption(f"  • `{inv}`")

                    # ── Row-level preview table (Original Value | Parsed Date | Final Value | Status) ──
                    if analysis.row_diagnostics:
                        preview_rows = []
                        for rd in analysis.row_diagnostics[:20]:
                            raw = rd.get("raw_value", "")
                            cleaned = rd.get("formatted_ddmmyy", "")   # DD-MM-YY string
                            status_val = rd.get("status", "")
                            parsed_iso = rd.get("parsed_iso", "") or "—"

                            if "Resolved via" in rd.get("reason", ""):
                                status_label = "Valid (Resolved)"
                                final_val = cleaned
                            elif status_val == "UNAMBIGUOUS":
                                status_label = "Valid"
                                final_val = cleaned
                            elif status_val == "AMBIGUOUS":
                                status_label = "Ambiguous"
                                final_val = "—"
                            elif status_val == "INVALID":
                                status_label = "Invalid"
                                final_val = "—"
                            elif status_val == "MISSING":
                                status_label = "Missing"
                                final_val = ""
                            else:
                                status_label = status_val
                                final_val = cleaned

                            preview_rows.append({
                                "Original Value": raw,
                                "Parsed Date": parsed_iso[:19].replace("T", " "),
                                "Final Value": final_val,
                                "Status": status_label,
                            })

                        st.markdown(
                            f"**Date Standardization Preview** — "
                            f"Input → `DD-MM-YY` &nbsp;*(first {len(preview_rows)} rows)*"
                        )
                        st.dataframe(preview_rows, use_container_width=True, hide_index=True)

                    # ── Override & Apply ──
                    st.markdown("---")
                    st.markdown("**Convention override** *(optional — use when auto-detection is ambiguous)*")
                    override_conv = st.selectbox(
                        "Enforce Date Convention:",
                        options=["auto", "DMY (Day-Month-Year)", "MDY (Month-Day-Year)", "YMD (Year-Month-Day)", "ISO"],
                        key="date_override_conv",
                    )
                    conv_map = {
                        "auto": None,
                        "DMY (Day-Month-Year)": "DMY",
                        "MDY (Month-Day-Year)": "MDY",
                        "YMD (Year-Month-Day)": "YMD",
                        "ISO": "ISO",
                    }

                    st.markdown(
                        f"**Output:** internal `datetime64[ns]` column **+** "
                        f"`{selected_dt_col}_display` string column formatted as `DD-MM-YY`."
                    )
                    if st.button("📅 Apply Date Standardization", type="primary", key="btn_normalize_dt"):
                        action = RepairAction(
                            operation=RepairOperation.NORMALIZE_DATES,
                            target=[selected_dt_col],
                            parameters={
                                "convention": conv_map[override_conv],
                                "display_format": DEFAULT_DISPLAY_FORMAT,
                            },
                            reason=(
                                f"Standardize date column '{selected_dt_col}' → "
                                f"datetime64 + DD-MM-YY display"
                            ),
                        )
                        _apply_repairs(df, [action])
                        st.rerun()

            else:
                st.info("No date-like columns detected in the current dataset.")


        with col_t2:
            st.markdown("##### 🚨 Impossible Values & Domain Violations")
            st.caption("Checks for domain violations: negative age/quantity, bounded percentages > 100%, and infinite numbers.")
            imp_issues = detect_impossible_values(df)
            if imp_issues:
                st.error(f"Found impossible value issue(s) in **{len(imp_issues)}** column(s):")
                for c_name, iss_list in imp_issues.items():
                    for iss in iss_list:
                        st.markdown(f"• **`{c_name}`**: {iss['reason']} *(severity: {iss['severity']})*")
            else:
                st.success("✅ No domain violations (negative age/qty, % > 100, inf) detected.")

            st.markdown("---")
            st.markdown("##### 🛡️ Robust Outlier Detection & Capping")
            st.caption("Separate statistical anomalies from errors using robust estimators (MAD, IQR, Z-Score, Percentiles).")

            num_cols = safe_numeric_columns(df)
            outlier_target = st.multiselect(
                "Select Numeric Columns (leave empty for all numeric):",
                options=num_cols,
                key="outlier_target_cols",
            )

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                outlier_method = st.selectbox(
                    "Outlier Method:",
                    options=["iqr", "mad", "zscore", "percentile"],
                    format_func=lambda x: {
                        "iqr": "IQR (Interquartile Range)",
                        "mad": "MAD (Median Absolute Deviation)",
                        "zscore": "Z-Score (Standard Deviations)",
                        "percentile": "Percentile (Extreme Quantiles)",
                    }.get(x, x),
                    key="outlier_method_select",
                )
            with col_m2:
                outlier_action = st.selectbox(
                    "Action:",
                    options=["cap", "flag"],
                    format_func=lambda x: "Cap (clamp to boundary)" if x == "cap" else "Flag (audit only)",
                    key="outlier_action_select",
                )

            if outlier_method == "iqr":
                factor_val = st.slider("IQR Multiplier:", min_value=1.0, max_value=3.0, value=1.5, step=0.1, key="slider_iqr")
            elif outlier_method in ("zscore", "mad"):
                factor_val = st.slider(f"{outlier_method.upper()} Threshold (σ / MAD units):", min_value=2.0, max_value=4.5, value=3.0, step=0.25, key=f"slider_{outlier_method}")
            else:  # percentile
                factor_val = st.slider("Percentile Tail Trim (fraction):", min_value=0.005, max_value=0.05, value=0.01, step=0.005, format="%.3f", key="slider_pct")

            if st.button("🛡️ Execute Outlier Handling", type="primary", key="btn_exec_outliers"):
                action = RepairAction(
                    operation=RepairOperation.HANDLE_OUTLIERS,
                    target=outlier_target if outlier_target else [],
                    parameters={"method": outlier_method, "factor": factor_val, "action": outlier_action},
                    reason=f"{outlier_action.title()} extreme outliers using {outlier_method.upper()} (factor={factor_val})",
                )
                _apply_repairs(df, [action])
                st.rerun()

    with tab_struct:
        st.markdown("##### 💥 Unpack / Explode Multi-Value Record Cells")
        st.caption("Detects and explodes rows where multiple values or sub-records are packed into individual cells using delimiters (newlines, commas, pipes, etc.), synchronizing linked columns.")

        detected_mv = detect_multi_value_cells(df)
        if detected_mv and detected_mv.get("multi_value_columns"):
            sync_cols = detected_mv.get("synchronized_columns")
            mv_cols = detected_mv.get("multi_value_columns")
            det_cols: list[str] = [str(c) for c in (sync_cols if sync_cols else (mv_cols or []))]
            delim = detected_mv.get("delimiter", "|")
            is_sync = detected_mv.get("is_synchronized", False)
            sync_label = "Synchronized across columns" if is_sync else "Independent columns"
            st.info(
                f"Detected **{len(det_cols)}** multi-value column(s) ({sync_label}): "
                + ", ".join([f"`{c}`" for c in det_cols])
                + f" (delimiter: `{repr(delim)}`)"
            )
            selected_mv_cols = st.multiselect(
                "Columns to Explode / Unpack (leave empty to use all detected):",
                options=list(df.columns),
                default=det_cols,
                key="explode_mv_cols",
            )
            explode_cols_to_use = list(selected_mv_cols) if selected_mv_cols else det_cols
            if st.button("💥 Explode Multi-Value Cells", type="primary", key="btn_explode_mv"):
                action = RepairAction(
                    operation=RepairOperation.EXPLODE_MULTI_VALUE_CELLS,
                    target=explode_cols_to_use,
                    reason=f"Explode multi-value cells across {len(explode_cols_to_use)} column(s)",
                )
                _apply_repairs(df, [action])
                st.rerun()
        else:
            st.success("✅ No multi-value delimited cells detected.")
            all_str_cols = [c for c in df.columns if df[c].dtype == "object"]
            manual_explode_cols = st.multiselect(
                "Manual Columns to Explode (if you know cells contain delimited lists):",
                options=all_str_cols,
                key="manual_explode_mv_cols",
            )
            if manual_explode_cols and st.button("💥 Explode Selected Columns", key="btn_manual_explode_mv"):
                action = RepairAction(
                    operation=RepairOperation.EXPLODE_MULTI_VALUE_CELLS,
                    target=manual_explode_cols,
                    reason=f"Manual explosion of multi-value cells in {len(manual_explode_cols)} column(s)",
                )
                _apply_repairs(df, [action])
                st.rerun()

        st.markdown("---")
        st.markdown("##### 📑 Repeated Header & Metadata Row Removal")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("###### Repeated Column Headers in Data")
            st.caption("Removes rows inside the data block that repeat the table column names (common in concatenated report exports).")
            if st.button("🗑️ Remove Repeated Headers"):
                action = RepairAction(
                    operation=RepairOperation.REMOVE_REPEATED_HEADERS,
                    reason="Remove repeated header rows inside data block",
                )
                _apply_repairs(df, [action])
                st.rerun()
        with col_r2:
            st.markdown("###### Trailing Summary & Metadata Rows")
            st.caption("Removes total/subtotal, disclaimer, or notes rows at the bottom of exported tables.")
            if st.button("🗑️ Remove Metadata / Summary Rows"):
                action = RepairAction(
                    operation=RepairOperation.REMOVE_METADATA_ROWS,
                    reason="Remove summary/metadata rows at dataset boundaries",
                )
                _apply_repairs(df, [action])
                st.rerun()

        st.markdown("---")
        st.markdown("##### 🔄 Unpivot (Wide to Long)")
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            id_cols = st.multiselect(
                "Identifier Columns (leave empty to auto-detect):",
                options=list(df.columns),
                key="unpivot_id_cols",
            )
            var_name = st.text_input("Variable Column Name:", value="Variable", key="unpivot_var_name")
        with col_u2:
            val_cols = st.multiselect(
                "Value Columns to unpivot (leave empty to use all non-ID):",
                options=[c for c in df.columns if c not in id_cols],
                key="unpivot_val_cols",
            )
            val_name = st.text_input("Value Column Name:", value="Value", key="unpivot_val_name")

        if st.button("Execute Unpivot / Reshape"):
            action = RepairAction(
                operation=RepairOperation.UNPIVOT,
                parameters={
                    "id_columns": id_cols if id_cols else None,
                    "value_columns": val_cols if val_cols else None,
                    "var_name": var_name,
                    "value_name": val_name,
                },
                reason="Manual unpivot to long format",
            )
            _apply_repairs(df, [action])
            st.rerun()

        st.markdown("---")
        st.markdown("##### 🏷️ Flatten Headers")
        h_rows = st.number_input("Number of header rows to flatten/promote:", min_value=1, max_value=max(1, len(df) - 1), value=1)
        if st.button("Flatten / Promote Header Rows"):
            action = RepairAction(
                operation=RepairOperation.FLATTEN_HEADERS,
                parameters={"header_rows": h_rows},
                reason=f"Promote first {h_rows} row(s) to headers",
            )
            _apply_repairs(df, [action])
            st.rerun()

    # ── Tab: Rename Column ──
    with tab_rename:
        st.markdown("##### 🏷️ Rename Column")
        st.caption("Select the column you want to rename.")

        col_options = list(df.columns)
        rename_col_selected = st.selectbox(
            "Select Column:",
            options=col_options,
            key="rename_tab_col_select",
        )

        rename_new_name = st.text_input(
            "New Column Name:",
            value="",
            key="rename_tab_new_name",
            placeholder="Enter new column name...",
        )

        # Show live preview when both fields have values
        if rename_col_selected and rename_new_name and rename_new_name.strip():
            cleaned_new = rename_new_name.strip()
            is_valid, rename_err = validate_column_rename(df, rename_col_selected, cleaned_new)

            col_dtype = str(df[rename_col_selected].dtype) if rename_col_selected in df.columns else "unknown"
            st.markdown(
                f"""
                <div style="background-color: #1E293B; border: 2px solid #3B82F6; border-radius: 8px; padding: 16px; margin: 12px 0;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                        Rename Preview
                    </div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 10px;">
                        <code>{rename_col_selected}</code> &nbsp;→&nbsp; <span style="color: #4ADE80;"><code>{cleaned_new}</code></span>
                    </div>
                    <div style="color: #94A3B8; font-size: 0.85rem; line-height: 1.6;">
                        Rows affected: <strong>0</strong> &nbsp;•&nbsp;
                        Values changed: <strong>0</strong> &nbsp;•&nbsp;
                        Datatype: <strong>{col_dtype} (unchanged)</strong> &nbsp;•&nbsp;
                        Data loss: <strong>None</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not is_valid:
                st.error(f"❌ {rename_err}")
            else:
                if st.button("🏷️ Rename Column", type="primary", key="btn_apply_rename_tab"):
                    try:
                        repaired_df, record = rename_single_column(
                            df=df,
                            old_name=rename_col_selected,
                            new_name=cleaned_new,
                            method="manual",
                            confidence=1.0,
                        )
                        # Post-validation: verify integrity
                        assert len(repaired_df) == len(df), "Row count mismatch after rename"
                        assert len(repaired_df.columns) == len(df.columns), "Column count mismatch after rename"
                        assert cleaned_new in repaired_df.columns, "New column name not found after rename"
                        assert str(repaired_df[cleaned_new].dtype) == str(df[rename_col_selected].dtype), "Dtype changed after rename"

                        update_dataset(repaired_df)
                        st.session_state["repair_records"].append(record)
                        st.toast(f"✅ Renamed '{rename_col_selected}' → '{cleaned_new}' successfully!")
                        st.rerun()
                    except AssertionError as ae:
                        st.error(f"❌ Rename validation failed: {ae}")
                    except Exception as e:
                        st.error(f"❌ Failed to rename column: {str(e)}")
        elif rename_new_name and not rename_new_name.strip():
            st.warning("⚠️ Column name cannot be empty or whitespace only.")

    with tab_reset:
        st.warning("⚠️ Reverts all repairs and restores the original untouched dataset.")
        if st.button("🔄 Reset to Original Dataset", type="secondary"):
            update_dataset(original_df.copy())
            st.session_state["repair_records"] = []
            st.session_state["repair_report"] = None
            st.session_state["inspection_report"] = None
            st.session_state["structure_report"] = None
            st.session_state["ai_repair_plan"] = None
            st.session_state["autonomous_result"] = None
            st.success("Dataset successfully reset to original version.")
            st.rerun()

    st.divider()

    # ── Repair History & Download ──
    render_section_header("Repair Audit Trail & CSV Export", icon="📜")
    records = st.session_state.get("repair_records", [])

    if records:
        history_data = []
        for r in records:
            op_label = r.operation.replace("_", " ").title()
            col_target = getattr(r, "column", None) or (r.details.get("column") if r.details else None) or "—"
            issue_val = getattr(r, "issue_type", None)
            issue_name = issue_val.value if hasattr(issue_val, "value") else (str(issue_val) if issue_val else (r.details.get("issue_type") if r.details else "—"))
            orig_val = getattr(r, "original_value", None) or (r.details.get("original_value") if r.details else None) or "—"
            new_val = getattr(r, "new_value", None) or (r.details.get("new_value") if r.details else None) or "—"
            meth_val = getattr(r, "method", None) or (r.details.get("method") if r.details else None) or "deterministic"
            conf_val = getattr(r, "confidence", 1.0)
            conf_str = f"{conf_val:.0%}" if conf_val is not None else "100%"
            risk_val = getattr(r, "risk_level", None) or ("safe" if (conf_val or 1.0) >= 0.95 else "medium")
            stat_val = getattr(r, "status", None) or ("✅ Applied" if r.success else "❌ Failed")

            detail_str = ""
            if r.details:
                if r.details.get("repair_type") == "column_rename":
                    old = r.details.get("old_name")
                    new = r.details.get("new_name")
                    if old and new:
                        detail_str = f"`{old}` → `{new}`"
                elif r.details.get("repair_type") == "column_drop":
                    detail_str = f"Dropped `{r.details.get('column_dropped')}`"
                elif "reason" in r.details:
                    detail_str = str(r.details["reason"])

            history_data.append({
                "Column": col_target,
                "Issue Taxonomy": issue_name,
                "Operation": op_label,
                "Original": orig_val if orig_val != "—" else (detail_str or "—"),
                "Repaired Value": new_val if new_val != "—" else (detail_str or "—"),
                "Method": meth_val,
                "Confidence": conf_str,
                "Risk": risk_val,
                "Rows": f"{r.rows_before} → {r.rows_after}",
                "Status": stat_val,
            })
        st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)

        with st.expander("🔍 Detailed Audit Log (JSON Records)", expanded=False):
            audit_entries = []
            for r in records:
                entry = {
                    "operation": r.operation,
                    "timestamp": r.timestamp.isoformat() if hasattr(r.timestamp, "isoformat") else str(r.timestamp),
                    "rows_before": r.rows_before,
                    "rows_after": r.rows_after,
                    "columns_before": r.columns_before,
                    "columns_after": r.columns_after,
                    "success": r.success,
                }
                if r.details:
                    entry.update(r.details)
                audit_entries.append(entry)
            st.json(audit_entries)
    else:
        st.info("No repairs have been applied yet.")

    # Repaired Dataset Preview
    render_section_header("Current Working Dataset Preview", icon="👁️")
    preview = make_arrow_safe_preview(df, max_rows=50)
    st.dataframe(preview, use_container_width=True)

    # Download button (all dates exported in standardized DD-MM-YY format)
    csv_bytes = df.to_csv(index=False, date_format="%d-%m-%y").encode("utf-8")
    st.download_button(
        label="📥 Download Clean Dataset (CSV)",
        data=csv_bytes,
        file_name=f"clean_{dataset_name.replace('.xlsx', '.csv').replace('.parquet', '.csv')}",
        mime="text/csv",
    )


def _apply_repairs(df: pd.DataFrame, actions: list[RepairAction]) -> None:
    """Execute a list of repairs, update session state, and display feedback."""
    try:
        with st.spinner("Applying repairs and validating..."):
            repaired_df, result = RepairService.repair(df, actions)

        if result.success:
            update_dataset(repaired_df)
            st.session_state["inspection_report"] = None
            st.session_state["structure_report"] = None
            st.session_state["repair_report"] = result

            if "repair_records" not in st.session_state:
                st.session_state["repair_records"] = []
            st.session_state["repair_records"].extend(result.records)

            st.toast(f"✅ Applied {len(result.actions_applied)} repair(s) successfully!")
        else:
            if result.errors:
                for err in result.errors:
                    st.error(f"❌ {err}")
            if result.warnings:
                for warn in result.warnings:
                    st.warning(f"⚠️ {warn}")

    except Exception as e:
        logger.exception("Failed to execute repair: {}", str(e))
        st.error(f"❌ Repair failed: {str(e)}")
