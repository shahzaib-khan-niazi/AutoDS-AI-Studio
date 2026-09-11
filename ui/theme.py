"""Centralized Theme and UI Helper System for AutoDS AI Studio.

Provides reusable dark SaaS design tokens, custom CSS injection, and layout component renderers.
Color Palette:
- Background: #0F172A
- Secondary / Card: #1E293B
- Border: #334155
- Primary: #6366F1
- Secondary Accent: #8B5CF6
- Success: #22C55E
- Warning: #F59E0B
- Danger: #EF4444
- Text: #F8FAFC
- Muted: #94A3B8
"""

import streamlit as st
from typing import Optional
from core.state import has_dataset, get_dataset, get_dataset_name
from utils.formatting import format_number


def apply_custom_css() -> None:
    """Inject custom SaaS dark theme styling into the Streamlit app."""
    css = """
    <style>
    /* Global Background & Font */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #F8FAFC;
    }

    /* SaaS Header Container */
    .saas-header-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
    }
    .saas-header-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .saas-header-desc {
        font-size: 0.95rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.5;
    }

    /* Section Header */
    .saas-section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #F8FAFC;
        margin: 18px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* SaaS Cards */
    .saas-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    /* Metric Cards - Aligned Equal Height & Level */
    .saas-metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        min-height: 105px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-color 0.2s ease-in-out;
    }
    .saas-metric-card:hover {
        border-color: #6366F1;
    }
    .saas-metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .saas-metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .saas-metric-delta {
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .delta-success { color: #22C55E; }
    .delta-warning { color: #F59E0B; }
    .delta-danger { color: #EF4444; }
    .delta-normal { color: #94A3B8; }

    /* Badges */
    .saas-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-success { background-color: rgba(34, 197, 94, 0.15); color: #22C55E; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-warning { background-color: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-danger { background-color: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-primary { background-color: rgba(99, 102, 241, 0.15); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.3); }
    .badge-neutral { background-color: rgba(148, 163, 184, 0.15); color: #94A3B8; border: 1px solid rgba(148, 163, 184, 0.3); }

    /* Empty State Container */
    .saas-empty-state {
        background-color: #1E293B;
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 48px 32px;
        text-align: center;
        margin: 24px 0;
    }
    .saas-empty-icon {
        font-size: 3rem;
        margin-bottom: 12px;
    }
    .saas-empty-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 8px;
    }
    .saas-empty-desc {
        font-size: 0.9rem;
        color: #94A3B8;
        max-width: 480px;
        margin: 0 auto 16px auto;
        line-height: 1.5;
    }

    /* Active Dataset Status Bar */
    .saas-dataset-bar {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Buttons Override */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button[kind="primary"] {
        background-color: #6366F1;
        color: #FFFFFF;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #4F46E5;
        border: none;
    }
    .stButton > button[kind="secondary"] {
        background-color: #1E293B;
        color: #F8FAFC;
        border: 1px solid #334155;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #6366F1;
        color: #6366F1;
    }

    /* Streamlit Tab Styling - Prevent Text Clipping & Ensure Full Visibility */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        min-height: 48px;
        border-bottom: 1px solid #334155;
        padding-bottom: 4px;
        overflow: visible !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto !important;
        min-height: 42px !important;
        white-space: normal !important;
        border-radius: 6px 6px 0 0 !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 8px 14px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        line-height: 1.3 !important;
        overflow: visible !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #6366F1 !important;
        border-bottom: 2px solid #6366F1 !important;
    }

    /* Tables & Dataframes */
    .stDataFrame {
        border: 1px solid #334155;
        border-radius: 8px;
    }

    /* Expander override */
    .stExpander {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_page_header(
    title: str,
    description: str,
    icon: str = "🤖",
    show_dataset_status: bool = True,
) -> None:
    """Render a unified, clean SaaS header for every page."""
    apply_custom_css()
    
    st.markdown(
        f"""
        <div class="saas-header-container">
            <div class="saas-header-title">
                <span>{icon}</span> {title}
            </div>
            <div class="saas-header-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if show_dataset_status and has_dataset():
        render_dataset_summary()


def render_section_header(title: str, icon: str = "", description: Optional[str] = None) -> None:
    """Render a section title with optional icon and description."""
    icon_str = f"{icon} " if icon else ""
    st.markdown(f'<div class="saas-section-header">{icon_str}{title}</div>', unsafe_allow_html=True)
    if description:
        st.caption(description)


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    icon: str = "",
) -> None:
    """Render a custom rounded metric card with clean borders and readable text."""
    icon_html = f"<span style='margin-right: 6px;'>{icon}</span>" if icon else ""
    delta_html = ""
    if delta:
        css_cls = f"delta-{delta_color}"
        delta_html = f'<div class="saas-metric-delta {css_cls}">{delta}</div>'

    st.markdown(
        f"""
        <div class="saas-metric-card">
            <div class="saas-metric-label">{icon_html}{label}</div>
            <div class="saas-metric-value" title="{value}">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(status: str, text: Optional[str] = None) -> str:
    """Return HTML string or render a badge with semantic colors."""
    status_lower = status.lower()
    badge_type = "neutral"
    if status_lower in ("connected", "ready", "completed", "success", "pass", "low"):
        badge_type = "success"
    elif status_lower in ("warning", "processing", "medium"):
        badge_type = "warning"
    elif status_lower in ("failed", "error", "critical", "high"):
        badge_type = "danger"
    elif status_lower in ("primary", "ai"):
        badge_type = "primary"

    display_text = text if text else status.upper()
    return f'<span class="saas-badge badge-{badge_type}">● {display_text}</span>'


def render_dataset_summary() -> None:
    """Render a dataset summary bar if a dataset is currently loaded."""
    if not has_dataset():
        return

    df = get_dataset()
    name = get_dataset_name() or "Active Dataset"
    if df is None:
        return

    rows = format_number(len(df))
    cols = format_number(len(df.columns))
    missing_pct = (df.isna().sum().sum() / max(len(df) * len(df.columns), 1)) * 100

    profile = st.session_state.get("dataset_profile")
    quality_score = f"{profile.quality_score:.0f}%" if profile else "N/A"

    st.markdown(
        f"""
        <div class="saas-dataset-bar">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.2rem;">📁</span>
                <div>
                    <strong style="color: #F8FAFC;">{name}</strong>
                    <div style="font-size: 0.8rem; color: #94A3B8;">{rows} rows × {cols} columns</div>
                </div>
            </div>
            <div style="display: flex; gap: 20px; align-items: center; font-size: 0.85rem;">
                <div><span style="color: #94A3B8;">Missing Cells:</span> <strong style="color: #F8FAFC;">{missing_pct:.1f}%</strong></div>
                <div><span style="color: #94A3B8;">Quality Score:</span> <strong style="color: #6366F1;">{quality_score}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(
    title: str = "No dataset loaded",
    description: str = "Upload a CSV, Excel, or supported dataset to begin analysis.",
    icon: str = "📁",
) -> None:
    """Render a clean empty state card."""
    st.markdown(
        f"""
        <div class="saas-empty-state">
            <div class="saas-empty-icon">{icon}</div>
            <div class="saas-empty-title">{title}</div>
            <div class="saas-empty-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
