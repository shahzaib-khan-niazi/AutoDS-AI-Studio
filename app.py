"""AutoDS AI Studio — Main Streamlit Application Entry Point.

AI-Powered Data Science SaaS Platform:
1. Configure Streamlit page & theme
2. Initialize session state
3. Render professional sidebar navigation & dataset status
4. Dispatch to modular UI components:
   - DATA: Upload, Profiler, Inspector, Structure, Repair, EDA
   - AI: AI Data Scientist, AI Chatbot
   - MACHINE LEARNING: AutoML Modeling
"""

import streamlit as st

from core.constants import APP_NAME, APP_ICON, APP_VERSION
from core.state import initialize_state, has_dataset, get_dataset_name, get_dataset
from core.logging import logger
from ui.theme import apply_custom_css
from ui.upload import show_upload
from ui.agent_dashboard import show_agent_dashboard
from ui.profiler import show_profiler
from ui.inspector import show_inspector
from ui.structure import show_structure
from ui.repair import show_repair
from ui.eda import show_eda
from ui.modeling import show_modeling
from ui.llm_test import show_llm_test
from utils.formatting import format_number


# Page Configuration
st.set_page_config(
    page_title=f"{APP_NAME} — AI Data Science Platform",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State & Inject Styling
initialize_state()
apply_custom_css()


def main() -> None:
    """Main application orchestrator."""
    # Sidebar Header
    st.sidebar.markdown(
        f"""
        <div style="padding: 8px 0 16px 0; border-bottom: 1px solid #334155; margin-bottom: 16px;">
            <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                {APP_ICON} {APP_NAME}
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 500; margin-top: 2px;">
                AI-Powered Data Science Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active Dataset Status Badge in Sidebar
    if has_dataset():
        df = get_dataset()
        name = get_dataset_name() or "dataset"
        if df is not None:
            rows_str = format_number(len(df))
            cols_str = format_number(len(df.columns))
            st.sidebar.markdown(
                f"""
                <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
                    <div style="font-size: 0.75rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;">Active Dataset</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin: 2px 0 6px 0; word-break: break-all;">{name}</div>
                    <div style="font-size: 0.8rem; color: #22C55E;">● Loaded ({rows_str} × {cols_str})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.markdown(
            """
            <div style="background: #1E293B; border: 1px dashed #334155; border-radius: 8px; padding: 12px; margin-bottom: 16px; text-align: center;">
                <div style="font-size: 0.8rem; color: #94A3B8;">📁 No dataset loaded yet</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Clean SaaS Navigation Menu (No duplicate Dataset Chatbot or AI Integration Test)
    nav_options = [
        "📤 Upload Dataset",
        "🔍 AI Dataset Profiler",
        "📊 Dataset Inspector",
        "🏗️ Structure Detection",
        "🛠️ Autonomous Clean & Repair",
        "📈 Automated EDA & Visuals",
        "🧠 AI Data Scientist",
        "⚙️ AutoML Modeling",
        "🧪 AI Chatbot",
    ]

    selected_page = st.sidebar.radio(
        "Navigation",
        nav_options,
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        """
        <div style="border-top: 1px solid #334155; margin-top: 20px; padding-top: 14px; font-size: 0.75rem; color: #94A3B8; line-height: 1.6;">
            <strong>Pipeline Guarantees:</strong><br/>
            • 🐍 Python Deterministic Execution<br/>
            • 🤖 AI Semantic Reasoning<br/>
            • 🛡️ Strict Profile Validation
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Route to Selected Page
    if "Upload" in selected_page:
        show_upload()
    elif "Profiler" in selected_page:
        show_profiler()
    elif "Inspector" in selected_page:
        show_inspector()
    elif "Structure" in selected_page:
        show_structure()
    elif "Repair" in selected_page or "Autonomous" in selected_page:
        show_repair()
    elif "EDA" in selected_page:
        show_eda()
    elif "Modeling" in selected_page:
        show_modeling()
    elif "AI Data Scientist" in selected_page:
        show_agent_dashboard()
    elif "Chatbot" in selected_page:
        show_llm_test()


if __name__ == "__main__":
    main()
