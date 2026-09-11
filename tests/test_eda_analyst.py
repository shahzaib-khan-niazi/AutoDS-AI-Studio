"""Unit tests for AIEDAAnalyst (services/eda/analyzer.py)."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from core.schemas.eda import AIEDAAnalysis
from services.eda.analyzer import AIEDAAnalyst


def test_eda_analyst_deterministic_summary() -> None:
    """EDA analyst computes accurate statistics without LLM."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
        "cat": ["A", "B", "A", "B", "A"],
    })

    summary, corr_items = AIEDAAnalyst._compute_eda_summary(df)

    assert summary["total_rows"] == 5
    assert summary["total_columns"] == 3
    assert len(corr_items) == 1
    assert corr_items[0].correlation == 1.0


def test_eda_analyst_fallback_analysis() -> None:
    """EDA analyst returns clean fallback when LLM is not configured."""
    df = pd.DataFrame({
        "x": [10.0, 20.0, 30.0],
        "y": [1.0, 2.0, 3.0],
    })

    with patch("services.eda.analyzer.llm_config") as mock_cfg:
        mock_cfg.is_configured = False
        res = AIEDAAnalyst.analyze(df)

        assert isinstance(res, AIEDAAnalysis)
        assert len(res.key_patterns) > 0
        assert len(res.significant_correlations) > 0
