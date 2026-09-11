"""AI Chatbot page for AutoDS AI Studio.

Enables interactive conversational Q&A grounded on the active uploaded dataset.
Uses OpenRouter LLMService (NVIDIA Nemotron primary + MiniMax fallback).
"""

import json
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from core.llm.config import llm_config
from core.llm.service import LLMService
from core.llm.exceptions import LLMConfigError, LLMServiceError
from core.logging import logger
from core.state import get_dataset, get_dataset_name, has_dataset
from core.utils.json_sanitizer import sanitize_for_json, safe_json_dumps
from ui.theme import render_page_header, render_empty_state
from utils.dataframe import safe_numeric_columns, safe_categorical_columns


def _build_dataset_context(df: pd.DataFrame, dataset_name: str) -> str:
    """Construct a comprehensive yet token-efficient context description of the dataset."""
    row_count = len(df)
    col_count = len(df.columns)
    try:
        dup_count = int(df.duplicated().sum())
    except Exception:
        dup_count = 0
    missing_total = int(df.isna().sum().sum())

    num_cols = safe_numeric_columns(df)
    cat_cols = safe_categorical_columns(df)

    cols_desc = []
    for col in df.columns:
        s = df[col]
        dt = str(s.dtype)
        nulls = s.isna().sum()
        null_pct = (nulls / max(row_count, 1)) * 100
        uniques = s.nunique(dropna=True)

        info = f"- **{col}** ({dt}): {uniques} unique values, {nulls} missing ({null_pct:.1f}%)"
        if col in num_cols:
            s_clean = s.dropna()
            if len(s_clean) > 0:
                info += f", Min={s_clean.min()}, Max={s_clean.max()}, Mean={s_clean.mean():.2f}, Median={s_clean.median():.2f}"
        elif col in cat_cols:
            vc = s.value_counts(dropna=True).head(3)
            top_cats = ", ".join([f"'{k}': {v}" for k, v in vc.items()])
            info += f", Top categories: [{top_cats}]"
        cols_desc.append(info)

    # First 3 sample rows
    sample_records = []
    for _, r in df.head(3).iterrows():
        sample_records.append({str(k): (str(v) if pd.notna(v) else None) for k, v in r.items()})

    context_parts = [
        f"DATASET SUMMARY:",
        f"- Filename: {dataset_name}",
        f"- Total Dimensions: {row_count:,} rows × {col_count} columns",
        f"- Total Missing Cells: {missing_total:,}",
        f"- Duplicate Rows: {dup_count:,}",
        "\nCOLUMN DETAILS:",
        "\n".join(cols_desc),
        "\nSAMPLE ROWS (First 3):",
        safe_json_dumps(sanitize_for_json(sample_records), indent=2),
    ]

    # Include profiler findings if available
    profile = st.session_state.get("dataset_profile")
    if profile:
        context_parts.append(
            f"\nQUALITY AUDIT:\n- Quality Score: {profile.quality_score:.1f}/100 ({profile.quality_status})"
        )
        if profile.detected_issues:
            issues_str = "; ".join([f"{iss.issue_type} in {iss.column or 'dataset'} ({iss.description})" for iss in profile.detected_issues[:5]])
            context_parts.append(f"- Detected Issues: {issues_str}")

    # Include AutoML findings if available
    automl = st.session_state.get("automl_summary")
    if automl:
        context_parts.append(
            f"\nAUTOML MODELING:\n- Target: {automl.target_column} ({automl.task_type.value})\n- Best Model: {automl.best_model_name}"
        )

    return "\n".join(context_parts)


def show_llm_test() -> None:
    """Render the user-facing AI Chatbot page."""
    render_page_header(
        title="AI Chatbot",
        description="Ask questions and discover insights from your uploaded dataset.",
        icon="🧪",
    )

    # ── Check Dataset Availability ──
    if not has_dataset():
        render_empty_state(
            title="No Dataset Loaded for Chatbot",
            description="Upload a dataset on the Upload page to ask questions and discover AI insights.",
            icon="🧪",
        )
        return

    df = get_dataset()
    dataset_name = get_dataset_name() or "dataset.csv"

    if df is None or len(df) == 0:
        st.warning("⚠️ The current dataset is empty.")
        return

    # ── Initialize Chat History ──
    if "dataset_chat_messages" not in st.session_state:
        st.session_state["dataset_chat_messages"] = [
            {
                "role": "assistant",
                "content": f"Hello! I am your AI Dataset Analyst. I've analyzed **`{dataset_name}`** ({len(df):,} rows × {len(df.columns)} columns). What questions can I answer for you about your data?",
            }
        ]

    # ── Suggested Question Prompts (Uniformly Aligned) ──
    st.markdown("**Suggested Questions:**")
    c1, c2, c3, c4 = st.columns(4)
    clicked_prompt = None
    with c1:
        if st.button("📊 Dataset Overview", key="qp_summary", use_container_width=True):
            clicked_prompt = "Provide an executive summary of this dataset, its main columns, and likely domain purpose."
    with c2:
        if st.button("⚠️ Missing & Quality", key="qp_quality", use_container_width=True):
            clicked_prompt = "What are the most notable data quality issues, missing values, and duplicate records in this dataset?"
    with c3:
        if st.button("🔥 Key Relationships", key="qp_corr", use_container_width=True):
            clicked_prompt = "What are the key numerical and categorical relationships in this dataset?"
    with c4:
        if st.button("💡 Important Insights", key="qp_ml", use_container_width=True):
            clicked_prompt = "Give me 5 important business and data insights from this dataset."

    st.markdown("---")

    # ── Clear Chat Button Bar ──
    col_chat_title, col_clear = st.columns([4, 1])
    with col_chat_title:
        st.caption(f"Active Conversation grounded on `{dataset_name}` ({len(df):,} rows × {len(df.columns)} cols)")
    with col_clear:
        if st.button("🗑️ Clear Chat", key="btn_clear_chat", use_container_width=True):
            st.session_state["dataset_chat_messages"] = [
                {
                    "role": "assistant",
                    "content": f"Chat cleared. How can I help you analyze **`{dataset_name}`**?",
                }
            ]
            st.rerun()

    # ── Render Message History ──
    for msg in st.session_state["dataset_chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Handle User Input ──
    user_input = st.chat_input("Ask any question about your dataset...")
    prompt_to_send = clicked_prompt or user_input

    if prompt_to_send:
        # Append User Message
        st.session_state["dataset_chat_messages"].append({
            "role": "user",
            "content": prompt_to_send,
        })
        with st.chat_message("user"):
            st.markdown(prompt_to_send)

        # Query LLM backend
        if not llm_config.is_configured:
            assistant_reply = "⚠️ AI Service is not configured. Please set `OPENROUTER_API_KEY` in your `.env` file to enable chatbot answers."
            st.session_state["dataset_chat_messages"].append({
                "role": "assistant",
                "content": assistant_reply,
            })
            with st.chat_message("assistant"):
                st.markdown(assistant_reply)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing dataset facts and formulating answer..."):
                    dataset_context = _build_dataset_context(df, dataset_name)
                    system_prompt = (
                        "You are the expert AI Data Scientist assistant for AutoDS AI Studio.\n"
                        "You have access to the exact uploaded dataset summary and statistical facts below.\n"
                        "RULES:\n"
                        "1. Answer user questions about the dataset accurately, concisely, and insightfully.\n"
                        "2. Ground all answers strictly in the provided dataset facts.\n"
                        "3. Do not invent columns, metrics, or rows that are not in the context.\n"
                        "4. Format your responses with clear markdown, bullet points, and bold headers where appropriate.\n\n"
                        f"{dataset_context}"
                    )

                    messages = [{"role": "system", "content": system_prompt}]
                    # Add last 6 messages from chat history for conversational context
                    for m in st.session_state["dataset_chat_messages"][-6:]:
                        if m["role"] in ("user", "assistant"):
                            messages.append({"role": m["role"], "content": m["content"]})

                    try:
                        svc = LLMService()
                        result = svc.generate(messages, max_tokens=1500)
                        assistant_reply = result.content

                        st.markdown(assistant_reply)

                        st.session_state["dataset_chat_messages"].append({
                            "role": "assistant",
                            "content": assistant_reply,
                        })

                    except LLMConfigError as e:
                        err_msg = f"❌ Configuration Error: {e.message}"
                        st.error(err_msg)
                        st.session_state["dataset_chat_messages"].append({"role": "assistant", "content": err_msg})
                    except LLMServiceError as e:
                        err_msg = f"❌ AI Service Error: {e.message}"
                        st.error(err_msg)
                        st.session_state["dataset_chat_messages"].append({"role": "assistant", "content": err_msg})
                    except Exception as e:
                        err_msg = f"❌ Unexpected Error: {str(e)}"
                        st.error(err_msg)
                        logger.exception("Dataset chatbot error")
                        st.session_state["dataset_chat_messages"].append({"role": "assistant", "content": err_msg})
