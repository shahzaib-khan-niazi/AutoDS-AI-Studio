"""Streamlit session state management for AutoDS AI Studio.

Provides safe initialization and access helpers for session state.
Always uses defensive copies for DataFrames to prevent accidental mutation.
"""

from typing import Any, Optional
import pandas as pd
import streamlit as st
from core.logging import logger


# Keys managed in session state
_STATE_KEYS: dict[str, Any] = {
    "dataset": None,
    "original_dataset": None,
    "dataset_name": None,
    "dataset_metadata": None,
    "dataset_profile": None,
    "ai_dataset_analysis": None,
    "inspection_report": None,
    "structure_report": None,
    "repair_report": None,
    "repair_history": [],
    "ai_plan": None,
    # Agent Workflow Pipeline State Keys
    "current_stage": "upload",
    "decision_log": [],
    "cleaning_plan": None,
    "cleaning_validation": None,
    "cleaning_results": None,
    "eda_results": None,
    "ai_eda_analysis": None,
    "ml_plan": None,
    "automl_summary": None,
    "explainability_results": None,
    "ai_insights": None,
    "final_report": None,
}


def initialize_state() -> None:
    """Initialize all session state keys with defaults.

    Safe to call multiple times — only sets keys that don't exist yet.
    """
    for key, default_value in _STATE_KEYS.items():
        if key not in st.session_state:
            if isinstance(default_value, list):
                st.session_state[key] = list(default_value)
            else:
                st.session_state[key] = default_value


def log_agent_decision(message: str, icon: str = "✓") -> None:
    """Append a concise record to the Agent Activity & Decision Log.

    Args:
        message: Brief description of the action or conclusion.
        icon: Status indicator icon (default '✓').
    """
    if "decision_log" not in st.session_state:
        st.session_state["decision_log"] = []
    
    log_entry = f"{icon} {message}"
    if log_entry not in st.session_state["decision_log"]:
        st.session_state["decision_log"].append(log_entry)
        logger.info("Agent Decision Log: {}", log_entry)


def get_decision_log() -> list[str]:
    """Retrieve the current Agent Activity & Decision Log."""
    return st.session_state.get("decision_log", [])


def set_dataset(df: pd.DataFrame, name: str, is_original: bool = False) -> None:
    """Store a dataset in session state with a defensive copy.

    Args:
        df: The DataFrame to store.
        name: Filename or label for the dataset.
        is_original: If True, also sets the original_dataset (only on first upload).
    """
    st.session_state["dataset"] = df.copy()
    st.session_state["dataset_name"] = name

    if is_original:
        st.session_state["original_dataset"] = df.copy()
        # Reset downstream reports when a new dataset is uploaded
        st.session_state["dataset_profile"] = None
        st.session_state["ai_dataset_analysis"] = None
        st.session_state["inspection_report"] = None
        st.session_state["structure_report"] = None
        st.session_state["repair_report"] = None
        st.session_state["repair_history"] = []
        st.session_state["ai_plan"] = None
        st.session_state["cleaning_plan"] = None
        st.session_state["cleaning_validation"] = None
        st.session_state["cleaning_results"] = None
        st.session_state["eda_results"] = None
        st.session_state["ai_eda_analysis"] = None
        st.session_state["ml_plan"] = None
        st.session_state["automl_summary"] = None
        st.session_state["explainability_results"] = None
        st.session_state["ai_insights"] = None
        st.session_state["final_report"] = None
        st.session_state["current_stage"] = "profile"
        st.session_state["decision_log"] = [f"✓ Dataset '{name}' loaded ({len(df):,} rows × {len(df.columns):,} cols)"]
        logger.info("Original dataset set: {} ({} rows × {} cols)", name, len(df), len(df.columns))


def get_dataset() -> Optional[pd.DataFrame]:
    """Get the current working dataset (defensive copy).

    Returns:
        A copy of the current DataFrame, or None if not loaded.
    """
    df = st.session_state.get("dataset")
    if df is not None and isinstance(df, pd.DataFrame):
        return df.copy()
    return None


def get_original_dataset() -> Optional[pd.DataFrame]:
    """Get the original uploaded dataset (defensive copy).

    Returns:
        A copy of the original DataFrame, or None if not loaded.
    """
    df = st.session_state.get("original_dataset")
    if df is not None and isinstance(df, pd.DataFrame):
        return df.copy()
    return None


def get_dataset_name() -> Optional[str]:
    """Get the current dataset filename."""
    name = st.session_state.get("dataset_name")
    if isinstance(name, str):
        return name
    return None


def update_dataset(df: pd.DataFrame) -> None:
    """Update the working dataset after a repair (defensive copy).

    Does NOT touch original_dataset.

    Args:
        df: The repaired DataFrame.
    """
    st.session_state["dataset"] = df.copy()
    logger.info("Working dataset updated ({} rows × {} cols)", len(df), len(df.columns))


def clear_dataset() -> None:
    """Clear all dataset-related state."""
    for key in _STATE_KEYS:
        if isinstance(_STATE_KEYS[key], list):
            st.session_state[key] = []
        else:
            st.session_state[key] = None
    logger.info("Dataset state cleared")


def has_dataset() -> bool:
    """Check if a dataset is currently loaded."""
    df = st.session_state.get("dataset")
    return df is not None and isinstance(df, pd.DataFrame)
