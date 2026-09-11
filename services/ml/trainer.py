"""AutoML Model Trainer — trains, evaluates, and ranks multiple scikit-learn models on clean data."""

import time
from typing import Any, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from core.logging import logger
from models.ml import MLTaskType, ModelEvaluationResult, FeatureImportance, AutoMLSummary
from services.ml.planner import MLPlanner
from utils.dataframe import safe_numeric_columns, safe_categorical_columns


class AutoMLTrainer:
    """Automated Machine Learning Trainer using Scikit-Learn."""

    @classmethod
    def train(
        cls,
        df: pd.DataFrame,
        target_column: str,
        task_type: Optional[MLTaskType] = None,
        test_size: float = 0.2,
    ) -> AutoMLSummary:
        """Train multiple models on clean dataset, evaluate, and rank them.

        Args:
            df: Clean DataFrame.
            target_column: Name of target column.
            task_type: Optional task type (auto-detected if None).
            test_size: Test split ratio (default 0.2).

        Returns:
            AutoMLSummary with leaderboard and best model.
        """
        logger.info("Starting AutoML training on target: '{}'", target_column)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        if task_type is None:
            task_type = MLPlanner.detect_task_type(df, target_column)

        # 1. Feature Engineering & Preprocessing
        X, y, feature_names = cls._prepare_features(df, target_column, task_type)

        if len(X) < 10 or len(feature_names) == 0:
            raise ValueError("Insufficient data or features for ML training (minimum 10 rows and 1 feature required)")

        # 2. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 3. Model Candidates
        models = cls._get_model_candidates(task_type)
        leaderboard: list[ModelEvaluationResult] = []
        feature_importances: list[FeatureImportance] = []

        for model_name, model_inst in models.items():
            try:
                t0 = time.time()
                model_inst.fit(X_train, y_train)
                fit_time = round(time.time() - t0, 3)

                y_pred_train = model_inst.predict(X_train)
                y_pred_test = model_inst.predict(X_test)

                eval_result = cls._evaluate_model(
                    model_name=model_name,
                    task_type=task_type,
                    y_train=y_train,
                    y_test=y_test,
                    y_pred_train=y_pred_train,
                    y_pred_test=y_pred_test,
                    fit_time=fit_time,
                )
                leaderboard.append(eval_result)

                # Extract feature importance from tree-based models if available
                if hasattr(model_inst, "feature_importances_") and not feature_importances:
                    raw_importances = model_inst.feature_importances_
                    for feat, imp in zip(feature_names, raw_importances):
                        feature_importances.append(
                            FeatureImportance(feature=feat, importance=round(float(imp), 4))
                        )
                    feature_importances.sort(key=lambda x: x.importance, reverse=True)

            except Exception as e:
                logger.warning("Model '{}' training failed: {}", model_name, str(e))

        # Rank Leaderboard
        if task_type == MLTaskType.REGRESSION:
            leaderboard.sort(key=lambda r: r.primary_metric_value, reverse=True)  # Higher R2 is better
        else:
            leaderboard.sort(key=lambda r: r.primary_metric_value, reverse=True)  # Higher Accuracy/F1 is better

        best_model_name = leaderboard[0].model_name if leaderboard else "None"

        summary = AutoMLSummary(
            target_column=target_column,
            task_type=task_type,
            rows_trained=len(X_train),
            features_used=feature_names,
            leaderboard=leaderboard,
            best_model_name=best_model_name,
            feature_importances=feature_importances,
        )

        logger.info("AutoML training complete. Best model: '{}'", best_model_name)
        return summary

    @classmethod
    def _prepare_features(
        cls, df: pd.DataFrame, target_column: str, task_type: MLTaskType
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Encode categorical features and scale numeric features."""
        data = df.dropna(subset=[target_column]).copy()

        y_raw = data[target_column]
        if task_type in (MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION):
            le = LabelEncoder()
            y = np.asarray(le.fit_transform(y_raw.astype(str)))
        else:
            y = np.asarray(pd.to_numeric(y_raw, errors="coerce").fillna(0.0).values)

        feature_df = data.drop(columns=[target_column]).copy()
        encoded_cols: list[str] = []
        arrays_to_concat: list[np.ndarray] = []

        # Process Numeric Columns
        numeric_cols = [c for c in safe_numeric_columns(feature_df)]
        if numeric_cols:
            num_data = feature_df[numeric_cols].fillna(0.0).values
            scaler = StandardScaler()
            scaled_num = scaler.fit_transform(num_data)
            arrays_to_concat.append(scaled_num)
            encoded_cols.extend(numeric_cols)

        # Process Categorical Columns (One-Hot or Ordinal)
        cat_cols = [c for c in feature_df.columns if c not in numeric_cols]
        for c in cat_cols:
            if feature_df[c].nunique() <= 20:
                dummies = pd.get_dummies(feature_df[c], prefix=c, drop_first=True)
                if not dummies.empty:
                    arrays_to_concat.append(dummies.values.astype(float))
                    encoded_cols.extend(list(dummies.columns))
            else:
                le_feat = LabelEncoder()
                enc_vals = np.asarray(le_feat.fit_transform(feature_df[c].astype(str))).reshape(-1, 1)
                arrays_to_concat.append(enc_vals.astype(float))
                encoded_cols.append(c)

        if arrays_to_concat:
            X = np.hstack(arrays_to_concat)
        else:
            X = np.empty((len(data), 0))

        return X, y, encoded_cols

    @staticmethod
    def _get_model_candidates(task_type: MLTaskType) -> dict[str, Any]:
        """Return candidate models based on task type."""
        if task_type == MLTaskType.REGRESSION:
            return {
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "Ridge Regression": Ridge(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            }
        else:
            return {
                "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
            }

    @staticmethod
    def _evaluate_model(
        model_name: str,
        task_type: MLTaskType,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_pred_train: np.ndarray,
        y_pred_test: np.ndarray,
        fit_time: float,
    ) -> ModelEvaluationResult:
        """Compute evaluation metrics for regression or classification."""
        if task_type == MLTaskType.REGRESSION:
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            mae = mean_absolute_error(y_test, y_pred_test)

            metrics = {
                "R² Score": round(test_r2, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
            }

            return ModelEvaluationResult(
                model_name=model_name,
                task_type=task_type,
                metrics=metrics,
                train_score=train_r2,
                test_score=test_r2,
                primary_metric_name="R² Score",
                primary_metric_value=test_r2,
                fit_time_seconds=fit_time,
            )
        else:
            train_acc = float(accuracy_score(y_train, y_pred_train))
            test_acc = float(accuracy_score(y_test, y_pred_test))
            f1 = float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))
            prec = float(precision_score(y_test, y_pred_test, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred_test, average="weighted", zero_division=0))

            metrics = {
                "Accuracy": round(test_acc, 4),
                "F1 Score (Weighted)": round(f1, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
            }

            return ModelEvaluationResult(
                model_name=model_name,
                task_type=task_type,
                metrics=metrics,
                train_score=train_acc,
                test_score=test_acc,
                primary_metric_name="Accuracy",
                primary_metric_value=test_acc,
                fit_time_seconds=fit_time,
            )
