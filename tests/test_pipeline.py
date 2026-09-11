"""Tests for Autonomous Preprocessing Pipeline."""

import numpy as np
import pandas as pd
import pytest

from services.pipeline.autonomous import AutonomousPipelineService


def test_autonomous_preprocessing_pipeline() -> None:
    # Create raw dataset with multiple issues:
    # 1. Duplicate rows
    # 2. Unnamed column name
    # 3. Completely empty column
    # 4. High-null column (>80% missing)
    # 5. Missing values in numeric column
    # 6. Extreme outlier value
    df_raw = pd.DataFrame({
        "Unnamed: 0": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "category": ["A", "A", "B", "A", "C", "B", "C", "A", "B", "C", "A", "B", "C"],
        "num_val": [10.0, 10.0, 20.0, np.nan, 25.0, 30.0, 15.0, 22.0, 28.0, 18.0, 35.0, 24.0, 1000.0],  # 1000 is outlier
        "all_null": [np.nan] * 13,
        "high_null": [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],

    })

    cleaned_df, result = AutonomousPipelineService.run_auto_preprocess(df_raw, use_ai_if_available=False)

    assert result.success is True
    # Duplicate removed (13 -> 12)
    assert len(cleaned_df) == 12
    # Unnamed column renamed
    assert "Unnamed: 0" not in cleaned_df.columns
    # Empty column dropped
    assert "all_null" not in cleaned_df.columns
    # High-null column dropped
    assert "high_null" not in cleaned_df.columns
    # Missing values imputed (zero remaining nulls)
    assert cleaned_df.isna().sum().sum() == 0
    # Outlier capped
    assert cleaned_df["num_val"].max() < 1000.0
    # Quality improved
    assert result.final_quality_score >= result.initial_quality_score
    assert len(result.summary_bullet_points) > 0
