"""Tests for AutoML Planner and Trainer on clean data."""

import numpy as np
import pandas as pd
import pytest

from models.ml import MLTaskType
from services.ml.planner import MLPlanner
from services.ml.trainer import AutoMLTrainer


def test_ml_planner_detection() -> None:
    df_cls = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    })
    task = MLPlanner.detect_task_type(df_cls, "target")
    assert task == MLTaskType.BINARY_CLASSIFICATION

    df_reg = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "price": [100.5, 200.3, 305.1, 410.2, 502.8, 620.4, 715.0, 809.9],
    })
    task_reg = MLPlanner.detect_task_type(df_reg, "price")
    assert task_reg == MLTaskType.REGRESSION


def test_automl_trainer_classification() -> None:
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.normal(50000, 15000, size=n),
        "city": np.random.choice(["NY", "SF", "LA", "Chicago"], size=n),
        "subscribed": np.random.choice(["Yes", "No"], size=n),
    })

    summary = AutoMLTrainer.train(df, target_column="subscribed")

    assert summary.task_type in (MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION)
    assert len(summary.leaderboard) >= 3
    assert summary.best_model_name != ""
    assert len(summary.feature_importances) > 0
    # Metrics check
    best = summary.leaderboard[0]
    assert "Accuracy" in best.metrics
    assert best.primary_metric_value > 0.0


def test_automl_trainer_regression() -> None:
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "size_sqft": np.random.uniform(500, 3000, size=n),
        "bedrooms": np.random.randint(1, 6, size=n),
        "price": np.random.uniform(100000, 800000, size=n),
    })

    summary = AutoMLTrainer.train(df, target_column="price")

    assert summary.task_type == MLTaskType.REGRESSION
    assert len(summary.leaderboard) >= 3
    best = summary.leaderboard[0]
    assert "R² Score" in best.metrics
    assert "RMSE" in best.metrics
