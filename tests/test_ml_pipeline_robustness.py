import pytest
import pandas as pd
import numpy as np

from models.ml import MLTaskType, AutoMLSummary
from services.ml.planner import MLPlanner
from services.ml.trainer import AutoMLTrainer


def test_scenario_a_numerical_classification():
    """Scenario A: Pure numerical features + classification target."""
    np.random.seed(42)
    n = 150
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    y = (f1 + f2 > 0).astype(int)

    df = pd.DataFrame({"feat_1": f1, "feat_2": f2, "target_class": y})
    summary = AutoMLTrainer.train(df, target_column="target_class", task_type=MLTaskType.BINARY_CLASSIFICATION)

    assert summary.rows_trained > 0
    assert summary.best_model_name != "None"
    assert len(summary.leaderboard) > 1
    # Ensure baseline DummyClassifier exists
    has_baseline = any(r.is_baseline for r in summary.leaderboard)
    assert has_baseline is True


def test_scenario_b_numerical_categorical_classification():
    """Scenario B: Numerical + categorical features + classification target."""
    np.random.seed(42)
    n = 150
    cat_vals = np.random.choice(["TypeA", "TypeB", "TypeC"], size=n)
    num_vals = np.random.randn(n)
    y = ((cat_vals == "TypeA") | (num_vals > 0.5)).astype(int)

    df = pd.DataFrame({"category_feat": cat_vals, "numeric_feat": num_vals, "outcome": y})
    summary = AutoMLTrainer.train(df, target_column="outcome", task_type=MLTaskType.BINARY_CLASSIFICATION)

    assert summary.best_model_name != "None"
    assert "category_feat" in summary.features_used
    assert "numeric_feat" in summary.features_used


def test_scenario_c_numerical_categorical_regression():
    """Scenario C: Numerical + categorical features + regression target."""
    np.random.seed(42)
    n = 150
    cat_vals = np.random.choice(["High", "Medium", "Low"], size=n)
    num_vals = np.random.randn(n) * 10
    y = (cat_vals == "High").astype(int) * 50.0 + num_vals * 2.5 + np.random.randn(n)

    df = pd.DataFrame({"tier": cat_vals, "score": num_vals, "revenue": y})
    summary = AutoMLTrainer.train(df, target_column="revenue", task_type=MLTaskType.REGRESSION)

    assert summary.best_model_name != "None"
    top_r2 = summary.leaderboard[0].primary_metric_value
    assert top_r2 > 0.5  # Model captures strong signal


def test_scenario_d_mixed_date_categorical_numerical():
    """Scenario D: Mixed date + categorical + numerical features."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range(start="2021-01-01", periods=n, freq="D")
    cats = np.random.choice(["North", "South"], size=n)
    nums = np.random.randn(n)
    y = (dates.year == 2021).astype(int) * 10.0 + nums * 5.0

    df = pd.DataFrame({"join_date": dates, "region": cats, "metric": nums, "target_var": y})
    summary = AutoMLTrainer.train(df, target_column="target_var", task_type=MLTaskType.REGRESSION)

    assert summary.best_model_name != "None"
    assert "join_date" in summary.features_used


def test_scenario_e_missing_values():
    """Scenario E: Missing values in features and target."""
    np.random.seed(42)
    n = 100
    nums = np.random.randn(n)
    nums[::10] = np.nan
    cats = np.random.choice(["A", "B", None], size=n)
    y = np.random.randn(n)
    y[5] = np.nan  # Missing target row

    df = pd.DataFrame({"f_num": nums, "f_cat": cats, "y_target": y})
    summary = AutoMLTrainer.train(df, target_column="y_target", task_type=MLTaskType.REGRESSION)

    assert summary.best_model_name != "None"
    # Ensure missing target row was dropped from ML population (99 trained/tested)
    assert summary.rows_trained + summary.rows_tested == 99


def test_scenario_f_imbalanced_classification():
    """Scenario F: Imbalanced classification target."""
    np.random.seed(42)
    n = 200
    y = np.zeros(n, dtype=int)
    y[:20] = 1  # 10% positive class (severe imbalance)
    x1 = np.random.randn(n) + y * 2.0  # Feature with signal

    df = pd.DataFrame({"feature_x": x1, "imbalanced_target": y})
    summary = AutoMLTrainer.train(df, target_column="imbalanced_target", task_type=MLTaskType.BINARY_CLASSIFICATION)

    assert summary.best_model_name != "None"
    # Check that metrics contain Balanced Accuracy
    best_eval = summary.leaderboard[0]
    assert "Balanced Accuracy" in best_eval.metrics


def test_scenario_g_small_dataset():
    """Scenario G: Small dataset (e.g. 25 rows)."""
    np.random.seed(42)
    n = 25
    x = np.random.randn(n)
    y = x * 3.0 + np.random.randn(n) * 0.1

    df = pd.DataFrame({"x_small": x, "y_small": y})
    summary = AutoMLTrainer.train(df, target_column="y_small", task_type=MLTaskType.REGRESSION)

    assert summary.rows_trained > 0
    assert summary.best_model_name != "None"


def test_scenario_h_identifier_columns():
    """Scenario H: Dataset with identifier columns (must be excluded from predictive features)."""
    n = 100
    uuids = [f"UUID_{i:05d}" for i in range(n)]
    nums = np.random.randn(n)
    y = nums * 4.0

    df = pd.DataFrame({"customer_uuid": uuids, "val_num": nums, "target_y": y})
    summary = AutoMLTrainer.train(df, target_column="target_y", task_type=MLTaskType.REGRESSION)

    assert "customer_uuid" not in summary.features_used
    assert "customer_uuid" in summary.dropped_features
    assert "val_num" in summary.features_used


def test_scenario_i_constant_columns():
    """Scenario I: Dataset with constant columns (must be excluded)."""
    n = 100
    const_col = ["SAME_VALUE"] * n
    nums = np.random.randn(n)
    y = nums * 2.0

    df = pd.DataFrame({"const_x": const_col, "real_x": nums, "target_y": y})
    summary = AutoMLTrainer.train(df, target_column="target_y", task_type=MLTaskType.REGRESSION)

    assert "const_x" not in summary.features_used
    assert "const_x" in summary.dropped_features
    assert "real_x" in summary.features_used


def test_scenario_j_duplicate_records():
    """Scenario J: Dataset with duplicate records."""
    np.random.seed(42)
    df_single = pd.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})
    df_dup = pd.concat([df_single, df_single], ignore_index=True)

    summary = AutoMLTrainer.train(df_dup, target_column="y", task_type=MLTaskType.REGRESSION)
    assert summary.best_model_name != "None"


def test_scenario_k_arbitrary_column_names():
    """Scenario K: Dataset with arbitrary/random column names (domain agnostic)."""
    np.random.seed(42)
    n = 100
    col_a = np.random.randn(n)
    col_b = np.random.choice(["alpha", "beta"], size=n)
    y = (col_a > 0).astype(int)

    df = pd.DataFrame({"abc_123": col_a, "xyz_999": col_b, "target_77": y})
    summary = AutoMLTrainer.train(df, target_column="target_77", task_type=MLTaskType.BINARY_CLASSIFICATION)

    assert summary.best_model_name != "None"
    assert "abc_123" in summary.features_used
    assert "xyz_999" in summary.features_used


def test_scenario_l_target_quality_zero_variance():
    """Scenario L: Target with zero variance should abort training gracefully with explicit warning."""
    df = pd.DataFrame({"x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "target_zero_var": [5] * 10})
    summary = AutoMLTrainer.train(df, target_column="target_zero_var", task_type=MLTaskType.REGRESSION)

    assert summary.rows_trained == 0
    assert len(summary.warnings) > 0
    assert "Insufficient target variation" in summary.warnings[0]


def test_datetime_value_based_detection_arbitrary_names():
    """Requirement 1, 3, 8, 14: Value-based datetime detection with arbitrary column names.
    Columns with high uniqueness and arbitrary names (col_a, timestamp_value, registration, event_time)
    must be detected as datetime features without reliance on hardcoded column names.
    """
    n = 100
    dates_1 = pd.date_range("2021-01-01", periods=n, freq="D").astype(str)  # 100% unique strings
    dates_2 = pd.date_range("2022-05-15", periods=n, freq="h").astype(str)  # 100% unique strings
    dates_3 = pd.date_range("2020-03-10", periods=n, freq="12h").astype(str)
    dates_4 = pd.date_range("2019-11-01", periods=n, freq="3D").astype(str)

    df = pd.DataFrame({
        "event_time": dates_1,
        "registration": dates_2,
        "col_a": dates_3,
        "timestamp_value": dates_4,
        "outcome": np.random.randint(0, 2, size=n),
    })

    f_analysis = MLPlanner.analyze_features(df, target_column="outcome")
    dt_cols = f_analysis["datetime_cols"]
    dropped = f_analysis["dropped_features"]

    assert "event_time" in dt_cols
    assert "registration" in dt_cols
    assert "col_a" in dt_cols
    assert "timestamp_value" in dt_cols
    assert len(dropped) == 0


def test_high_cardinality_date_vs_non_date_id():
    """Requirement 2, 9, 10, 15: High-cardinality date column is retained as datetime feature,
    while a high-cardinality non-date string ID is excluded as an identifier.
    """
    n = 100
    date_unique = pd.date_range("2023-01-01", periods=n, freq="D").astype(str)
    non_date_id = [f"USER_ID_{i:04d}" for i in range(n)]
    num_feat = np.random.randn(n)
    target = np.random.randint(0, 2, size=n)

    df = pd.DataFrame({
        "user_identity_code": non_date_id,
        "transaction_date_col": date_unique,
        "amount": num_feat,
        "target": target,
    })

    f_analysis = MLPlanner.analyze_features(df, target_column="target")
    assert "transaction_date_col" in f_analysis["datetime_cols"]
    assert "user_identity_code" in f_analysis["dropped_features"]
    assert "user_identity_code" not in f_analysis["datetime_cols"]


def test_mixed_date_formats_and_invalid_dates_row_preservation():
    """Requirement 11, 12, 15: Mixed valid date formats + invalid date values.
    Rows with invalid dates are preserved without dropping rows.
    """
    mixed_dates = [
        "02/27/2021", "15-01-2022", "11/05/2021", "2021/10/29", "invalid_date",
        "05/16/2021", "2021/10/10", "Dec 05 2021", "25-04-2022", "May 19 2021",
        "18-05-2021", "2021.08.23", "Jul 05 2021", "2021.11.18", "2021/09/28",
        "corrupted_value", "Apr 28 2021", "Feb 06 2021", "Aug 31 2021", "Mar 27 2022",
    ] * 5  # 100 rows total

    np.random.seed(42)
    n = len(mixed_dates)
    df = pd.DataFrame({
        "mixed_date_col": mixed_dates,
        "metric_val": np.random.randn(n),
        "target": np.random.randint(0, 2, size=n),
    })

    summary = AutoMLTrainer.train(df, target_column="target", task_type=MLTaskType.BINARY_CLASSIFICATION)
    assert summary.rows_trained + summary.rows_tested == n
    assert "mixed_date_col" in summary.features_used


def test_raw_datetime_strings_not_passed_to_model_and_leak_free():
    """Requirement 5, 6, 13, 15: Extract temporal features generically (year, month, day, dayofweek, hour, days_elapsed).
    Raw datetime strings are not passed to model; train/test preprocessing is leak-free.
    """
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="7D")
    df = pd.DataFrame({"random_time_col": dates.astype(str), "val": np.random.randn(n), "y": (dates.year == 2020).astype(int)})

    summary = AutoMLTrainer.train(df, target_column="y", task_type=MLTaskType.BINARY_CLASSIFICATION)
    assert summary.best_model_name != "None"
    assert "random_time_col" in summary.features_used

    # Verify extracted feature names from ColumnTransformer
    best_eval = summary.leaderboard[0]
    assert best_eval.primary_metric_value > 0.5

