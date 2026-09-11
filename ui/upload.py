"""Upload page UI for AutoDS AI Studio.

Handles file upload widget, loading via UploadService,
and displaying dataset preview with metadata.
"""

import streamlit as st
from core.state import set_dataset, has_dataset, get_dataset_name, get_dataset
from core.logging import logger
from core.constants import DEFAULT_PREVIEW_ROWS, SUPPORTED_EXTENSIONS
from core.exceptions import DatasetLoadError, DatasetValidationError
from services.upload.service import UploadService
from ui.theme import render_page_header, render_metric_card, render_section_header, render_empty_state
from utils.dataframe import make_arrow_safe_preview
from utils.formatting import format_bytes, format_number


def show_upload() -> None:
    """Render the Upload Dataset page."""
    render_page_header(
        title="Upload Dataset",
        description="Upload your structured dataset (CSV, XLSX, XLS, Parquet) to launch automated AI analysis and data science workflows.",
        icon="📁",
        show_dataset_status=False,
    )

    # Main Upload Card Area
    st.markdown(
        f"""
        <div class="saas-card" style="text-align: center; padding: 32px 24px;">
            <div style="font-size: 2.5rem; margin-bottom: 8px;">📤</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #F8FAFC;">Drag & Drop your dataset file here</div>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "parquet"],
        label_visibility="collapsed",
        help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}",
    )

    if uploaded_file is not None:
        _handle_upload(uploaded_file)
    elif not has_dataset():
        render_empty_state(
            title="No Dataset Uploaded Yet",
            description="Select or drop a file above to start profiling, repairing, and analyzing with AI.",
            icon="📁",
        )

    # Show current dataset info if one is loaded
    if has_dataset():
        _show_current_dataset()


def _handle_upload(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> None:
    """Process an uploaded file."""
    filename = uploaded_file.name

    # Avoid reloading the same file on every rerun
    current_name = get_dataset_name()
    if current_name == filename and has_dataset():
        return

    try:
        with st.spinner(f"Loading and validating {filename}..."):
            df, metadata = UploadService.load(uploaded_file, filename)

        # Store in session state
        set_dataset(df, filename, is_original=True)
        st.session_state["dataset_metadata"] = metadata

        st.toast(f"✅ Loaded {filename} successfully!", icon="🎉")

        render_section_header("Upload Summary", icon="📋")
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Rows", format_number(metadata.rows), icon="📐")
        with col2:
            render_metric_card("Columns", format_number(metadata.columns), icon="📊")
        with col3:
            render_metric_card("Memory Usage", format_bytes(metadata.memory_usage_bytes), icon="💾")

        # Show preview
        render_section_header("Dataset Preview", icon="👁️")
        preview = make_arrow_safe_preview(df, max_rows=DEFAULT_PREVIEW_ROWS)
        st.dataframe(preview, use_container_width=True)

    except (DatasetLoadError, DatasetValidationError) as e:
        st.error(f"❌ {e.message}")
        if e.details:
            with st.expander("Technical Error Details"):
                st.code(e.details)
        logger.error("Upload failed: {} — {}", e.message, e.details)

    except Exception as e:
        st.error("❌ An unexpected error occurred while loading the file.")
        logger.exception("Unexpected upload error: {}", str(e))


def _show_current_dataset() -> None:
    """Show information about the currently loaded dataset."""
    st.divider()
    render_section_header("Active Dataset Profile", icon="📊")

    metadata = st.session_state.get("dataset_metadata")
    name = get_dataset_name()
    df = get_dataset()

    if df is not None and name:
        rows = metadata.rows if metadata else len(df)
        cols = metadata.columns if metadata else len(df.columns)
        memory = format_bytes(metadata.memory_usage_bytes) if metadata else f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB"

        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Rows", format_number(rows), icon="📐")
        with col2:
            render_metric_card("Columns", format_number(cols), icon="📊")
        with col3:
            render_metric_card("Memory Size", memory, icon="💾")

        # Column types summary
        with st.expander("🔍 Column Schema & Data Types"):
            if metadata:
                for col_name, dtype in metadata.dtypes.items():
                    st.text(f"  • {col_name}: {dtype}")
            else:
                for col_name, dtype in df.dtypes.items():
                    st.text(f"  • {col_name}: {dtype}")

        # Dataset Preview
        render_section_header("Dataset Preview (First 20 Rows)", icon="👁️")
        preview = make_arrow_safe_preview(df, max_rows=DEFAULT_PREVIEW_ROWS)
        st.dataframe(preview, use_container_width=True)
