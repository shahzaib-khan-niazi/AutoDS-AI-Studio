"""AutoML Model Trainer — Leak-free Scikit-Learn Pipelines with Baselines and Cross-Validation."""

import time
from typing import Any, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    KFold,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from core.logging import logger
from models.ml import (
    MLTaskType,
    ModelEvaluationResult,
    FeatureImportance,
    AutoMLSummary,
)
from services.ml.planner import MLPlanner


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible transformer to extract year, month, day, dayofweek, hour, and days_elapsed from datetime columns."""

    def __init__(self, feature_names: Optional[list[str]] = None):
        self.feature_names = feature_names or []
        self.output_feature_names_: list[str] = []
        self.max_dates_: dict[str, Any] = {}

    def fit(self, X, y=None):
        self.output_feature_names_ = []
        self.max_dates_ = {}
        if isinstance(X, pd.DataFrame):
            X_df = X
            cols = list(X.columns)
        else:
            cols = self.feature_names if self.feature_names else [f"date_{i}" for i in range(X.shape[1] if hasattr(X, "shape") else 1)]
            X_df = pd.DataFrame(X, columns=cols)

        for col in cols:
            try:
                dt_s = pd.to_datetime(X_df[col], errors="coerce", utc=True)
            except Exception:
                dt_s = pd.Series(pd.NaT, index=X_df.index)

            valid_dt = dt_s.dropna()
            if not valid_dt.empty:
                self.max_dates_[col] = valid_dt.max()
            else:
                self.max_dates_[col] = pd.Timestamp("2025-01-01", tz="UTC")

            self.output_feature_names_.extend([
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_dayofweek",
                f"{col}_hour",
                f"{col}_days_elapsed",
            ])
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            cols = self.feature_names if self.feature_names else [f"date_{i}" for i in range(X.shape[1] if hasattr(X, "shape") else 1)]
            X_df = pd.DataFrame(X, columns=cols[:X.shape[1]] if hasattr(X, "shape") else None)

        extracted_parts = []
        for col in X_df.columns:
            try:
                dt_s = pd.to_datetime(X_df[col], errors="coerce", utc=True)
            except Exception:
                dt_s = pd.Series(pd.NaT, index=X_df.index)

            ref_dt = self.max_dates_.get(col, pd.Timestamp("2025-01-01", tz="UTC"))

            year = dt_s.dt.year.astype(float).values
            month = dt_s.dt.month.astype(float).values
            day = dt_s.dt.day.astype(float).values
            dayofweek = dt_s.dt.dayofweek.astype(float).values
            hour = dt_s.dt.hour.astype(float).values

            try:
                elapsed = (ref_dt - dt_s).dt.total_seconds() / 86400.0
                days_elapsed = elapsed.astype(float).values
            except Exception:
                days_elapsed = np.full(len(dt_s), np.nan)

            extracted_parts.append(np.column_stack([
                year,
                month,
                day,
                dayofweek,
                hour,
                days_elapsed,
            ]))

        if extracted_parts:
            return np.hstack(extracted_parts)
        return np.empty((len(X_df), 0))

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_feature_names_)


class AutoMLTrainer:
    """Automated Machine Learning Trainer using Scikit-Learn Pipelines."""

    @classmethod
    def train(
        cls,
        df: pd.DataFrame,
        target_column: str,
        task_type: Optional[MLTaskType] = None,
        test_size: float = 0.2,
    ) -> AutoMLSummary:
        """Train multiple candidate models + baselines using leak-free ColumnTransformer pipelines.

        Args:
            df: Source DataFrame.
            target_column: Target variable name.
            task_type: Optional task type (detected if None).
            test_size: Test split ratio (default 0.2).

        Returns:
            AutoMLSummary object with readiness report, leaderboard, and feature importances.
        """
        logger.info("Starting leak-free AutoML training on target: '{}'", target_column)

        # 1. Target Quality Check
        target_quality = MLPlanner.check_target_quality(df, target_column)
        if not target_quality["is_valid"]:
            logger.warning("Target quality check failed for '{}': {}", target_column, target_quality["reason"])
            return AutoMLSummary(
                target_column=target_column,
                task_type=task_type or MLTaskType.REGRESSION,
                rows_trained=0,
                rows_tested=0,
                warnings=[target_quality["reason"]],
            )

        # 2. Filter Target Missingness (Only remove missing targets from ML population)
        data = df.dropna(subset=[target_column]).copy()
        if len(data) < 10:
            return AutoMLSummary(
                target_column=target_column,
                task_type=task_type or MLTaskType.REGRESSION,
                rows_trained=0,
                rows_tested=0,
                warnings=["Insufficient non-null target rows for machine learning (minimum 10 required)"],
            )

        # Detect Task Type
        if task_type is None:
            task_type = MLPlanner.detect_task_type(data, target_column)

        # 3. Feature Readiness & Categorization Analysis
        feature_analysis = MLPlanner.analyze_features(data, target_column)
        readiness_report = MLPlanner.generate_ml_readiness_report(data, target_column)

        numeric_cols = feature_analysis["numeric_cols"]
        categorical_cols = feature_analysis["categorical_cols"]
        datetime_cols = feature_analysis["datetime_cols"]
        text_cols = feature_analysis["text_cols"]
        dropped_features = feature_analysis["dropped_features"]

        usable_features = numeric_cols + categorical_cols + datetime_cols + text_cols
        if not usable_features:
            return AutoMLSummary(
                target_column=target_column,
                task_type=task_type,
                rows_trained=0,
                rows_tested=0,
                dropped_features=dropped_features,
                ml_readiness_report=readiness_report,
                warnings=["No usable predictive features remaining after excluding identifiers and constant columns"],
            )

        # Prepare X and y
        X_df = data[usable_features].copy()
        y_raw = data[target_column]

        # Ensure numeric columns are strictly numeric dtype (coerce non-numeric strings to NaN for imputer)
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                X_df[col] = pd.to_numeric(X_df[col].astype(str).str.replace(r"[$,%]", "", regex=True), errors="coerce")

        # Ensure datetime columns are strictly datetime dtype via robust parser
        for col in datetime_cols:
            if not pd.api.types.is_datetime64_any_dtype(X_df[col]):
                try:
                    from services.repair.dates import classify_and_parse_single_date
                    parsed_vals = [classify_and_parse_single_date(v)[1] for v in X_df[col]]
                    X_df[col] = pd.to_datetime(parsed_vals, errors="coerce", utc=True)
                except Exception:
                    X_df[col] = pd.to_datetime(X_df[col], errors="coerce", utc=True)

        # Target Encoding
        label_encoder: Optional[LabelEncoder] = None
        if task_type in (MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw.astype(str))
        else:
            y = pd.to_numeric(y_raw, errors="coerce").fillna(0.0).values.astype(float)

        # 4. Train-Test Split (with Stratification for Classification)
        stratify_arr = None
        if task_type in (MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION):
            class_counts = pd.Series(y).value_counts()
            if (class_counts >= 2).all() and len(class_counts) > 1:
                stratify_arr = y

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=test_size, random_state=42, stratify=stratify_arr
        )

        # Check Class Imbalance
        is_imbalanced = False
        if task_type in (MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION):
            c_counts = pd.Series(y_train).value_counts()
            if len(c_counts) > 1 and (c_counts.max() / max(c_counts.min(), 1)) > 4.0:
                is_imbalanced = True

        # 5. Build ColumnTransformer Preprocessor (Fit on Train ONLY inside Pipeline)
        preprocessor = cls._build_preprocessor(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            datetime_cols=datetime_cols,
            text_cols=text_cols,
        )

        # 6. Candidate Models & Baselines
        candidates = cls._get_model_candidates(task_type, is_imbalanced=is_imbalanced)
        leaderboard: list[ModelEvaluationResult] = []
        fitted_pipelines: dict[str, Pipeline] = {}

        # Cross-Validation Strategy
        if task_type in (MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION):
            min_c = int(pd.Series(y_train).value_counts().min())
            n_folds = max(2, min(5, min_c))
            cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_scoring = "accuracy"
        else:
            n_folds = max(2, min(5, len(X_train) // 10))
            cv_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_scoring = "r2"

        baseline_metric_val = 0.0
        baseline_model_name = ""

        for model_name, (model_inst, is_baseline) in candidates.items():
            try:
                # Full Pipeline: Preprocessor + Model
                full_pipeline = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("model", model_inst),
                ])

                t0 = time.time()
                full_pipeline.fit(X_train, y_train)
                fit_time = round(time.time() - t0, 3)

                fitted_pipelines[model_name] = full_pipeline

                # Predictions
                y_pred_train = full_pipeline.predict(X_train)
                y_pred_test = full_pipeline.predict(X_test)

                # Predict Proba for Classification if available
                y_proba_test = None
                if task_type in (MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION):
                    if hasattr(full_pipeline, "predict_proba"):
                        try:
                            y_proba_test = full_pipeline.predict_proba(X_test)
                        except Exception:
                            y_proba_test = None

                # Cross Validation on Training Set
                try:
                    cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv_splitter, scoring=cv_scoring)
                    cv_mean = float(np.mean(cv_scores))
                    cv_std = float(np.std(cv_scores))
                except Exception as cv_err:
                    logger.debug("CV calculation skipped for '{}': {}", model_name, str(cv_err))
                    cv_mean, cv_std = 0.0, 0.0

                # Evaluate Metrics
                eval_result = cls._evaluate_model(
                    model_name=model_name,
                    task_type=task_type,
                    y_train=y_train,
                    y_test=y_test,
                    y_pred_train=y_pred_train,
                    y_pred_test=y_pred_test,
                    y_proba_test=y_proba_test,
                    fit_time=fit_time,
                    cv_mean=cv_mean,
                    cv_std=cv_std,
                    is_baseline=is_baseline,
                )
                leaderboard.append(eval_result)

                if is_baseline:
                    baseline_model_name = model_name
                    baseline_metric_val = eval_result.primary_metric_value

            except Exception as e:
                logger.warning("Candidate model '{}' training failed: {}", model_name, str(e))

        # Sort Leaderboard
        if task_type == MLTaskType.REGRESSION:
            leaderboard.sort(key=lambda r: (not r.is_baseline, r.primary_metric_value), reverse=True)
        else:
            leaderboard.sort(key=lambda r: (not r.is_baseline, r.primary_metric_value), reverse=True)

        best_model_name = leaderboard[0].model_name if leaderboard else "None"
        best_metric_val = leaderboard[0].primary_metric_value if leaderboard else 0.0
        best_pipeline = fitted_pipelines.get(best_model_name)

        # Extract Feature Importances from the best model (or fallback to any non-baseline fitted pipeline)
        feature_importances: list[FeatureImportance] = []
        if best_pipeline:
            feature_importances = cls._extract_feature_importances(best_pipeline)

        if not feature_importances:
            for m_name, pipe in fitted_pipelines.items():
                if m_name != baseline_model_name:
                    fi_list = cls._extract_feature_importances(pipe)
                    if fi_list:
                        feature_importances = fi_list
                        break

        warnings_list: list[str] = []
        if best_metric_val <= baseline_metric_val + 0.02 and len(leaderboard) > 1:
            warnings_list.append(
                f"Model performance ({best_model_name} score {best_metric_val:.4f}) is close to baseline ({baseline_model_name} {baseline_metric_val:.4f}) — dataset may have weak predictive signal."
            )

        summary = AutoMLSummary(
            target_column=target_column,
            task_type=task_type,
            rows_trained=len(X_train),
            rows_tested=len(X_test),
            features_used=usable_features,
            dropped_features=dropped_features,
            leaderboard=leaderboard,
            best_model_name=best_model_name,
            baseline_model_name=baseline_model_name,
            baseline_metric_value=baseline_metric_val,
            best_pipeline=best_pipeline,
            feature_importances=feature_importances,
            ml_readiness_report=readiness_report,
            warnings=warnings_list,
        )

        logger.info("AutoML training complete. Best model: '{}' (score: {:.4f})", best_model_name, best_metric_val)
        return summary

    @classmethod
    def _build_preprocessor(
        cls,
        numeric_cols: list[str],
        categorical_cols: list[str],
        datetime_cols: list[str],
        text_cols: list[str],
    ) -> ColumnTransformer:
        """Construct leak-free ColumnTransformer pipeline for all feature types."""
        transformers = []

        if numeric_cols:
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipeline, numeric_cols))

        if categorical_cols:
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipeline, categorical_cols))

        if datetime_cols:
            dt_pipeline = Pipeline(steps=[
                ("extractor", DatetimeFeatureExtractor(feature_names=datetime_cols)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("dt", dt_pipeline, datetime_cols))

        if text_cols:
            for t_col in text_cols:
                text_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                    ("tfidf", TfidfVectorizer(max_features=30, stop_words="english")),
                ])
                transformers.append((f"text_{t_col}", text_pipeline, t_col))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    @staticmethod
    def _get_model_candidates(
        task_type: MLTaskType, is_imbalanced: bool = False
    ) -> dict[str, tuple[Any, bool]]:
        """Return candidate models and baselines: name -> (model_instance, is_baseline)."""
        cw = "balanced" if is_imbalanced else None

        if task_type == MLTaskType.REGRESSION:
            return {
                "Dummy Regressor (Baseline)": (DummyRegressor(strategy="mean"), True),
                "Ridge Regression": (Ridge(alpha=1.0), False),
                "Random Forest Regressor": (RandomForestRegressor(n_estimators=100, random_state=42), False),
                "Gradient Boosting Regressor": (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
                "HistGradientBoosting Regressor": (HistGradientBoostingRegressor(random_state=42), False),
                "Decision Tree Regressor": (DecisionTreeRegressor(random_state=42), False),
            }
        else:
            return {
                "Dummy Classifier (Baseline)": (DummyClassifier(strategy="most_frequent"), True),
                "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42, class_weight=cw), False),
                "Random Forest Classifier": (RandomForestClassifier(n_estimators=100, random_state=42, class_weight=cw), False),
                "Gradient Boosting Classifier": (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
                "HistGradientBoosting Classifier": (HistGradientBoostingClassifier(random_state=42, class_weight=cw), False),
                "Decision Tree Classifier": (DecisionTreeClassifier(random_state=42, class_weight=cw), False),
            }

    @staticmethod
    def _evaluate_model(
        model_name: str,
        task_type: MLTaskType,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_pred_train: np.ndarray,
        y_pred_test: np.ndarray,
        y_proba_test: Optional[np.ndarray],
        fit_time: float,
        cv_mean: float,
        cv_std: float,
        is_baseline: bool,
    ) -> ModelEvaluationResult:
        """Compute comprehensive evaluation metrics for regression or classification."""
        if task_type == MLTaskType.REGRESSION:
            train_r2 = float(r2_score(y_train, y_pred_train))
            test_r2 = float(r2_score(y_test, y_pred_test))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            mae = float(mean_absolute_error(y_test, y_pred_test))

            metrics = {
                "R² Score": round(test_r2, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "CV R² Mean": round(cv_mean, 4),
                "CV R² Std": round(cv_std, 4),
            }

            return ModelEvaluationResult(
                model_name=model_name,
                task_type=task_type,
                metrics=metrics,
                train_score=round(train_r2, 4),
                test_score=round(test_r2, 4),
                cv_score_mean=round(cv_mean, 4),
                cv_score_std=round(cv_std, 4),
                primary_metric_name="R² Score",
                primary_metric_value=round(test_r2, 4),
                fit_time_seconds=fit_time,
                is_baseline=is_baseline,
            )
        else:
            train_acc = float(accuracy_score(y_train, y_pred_train))
            test_acc = float(accuracy_score(y_test, y_pred_test))
            bal_acc = float(balanced_accuracy_score(y_test, y_pred_test))
            f1 = float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))
            prec = float(precision_score(y_test, y_pred_test, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred_test, average="weighted", zero_division=0))

            roc_auc_val = 0.0
            if y_proba_test is not None:
                try:
                    if len(np.unique(y_test)) == 2 and y_proba_test.shape[1] >= 2:
                        roc_auc_val = float(roc_auc_score(y_test, y_proba_test[:, 1]))
                    elif len(np.unique(y_test)) > 2:
                        roc_auc_val = float(roc_auc_score(y_test, y_proba_test, multi_class="ovr", average="weighted"))
                except Exception:
                    roc_auc_val = 0.0

            metrics = {
                "Accuracy": round(test_acc, 4),
                "Balanced Accuracy": round(bal_acc, 4),
                "F1 Score (Weighted)": round(f1, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "CV Accuracy Mean": round(cv_mean, 4),
            }
            if roc_auc_val > 0:
                metrics["ROC-AUC"] = round(roc_auc_val, 4)

            return ModelEvaluationResult(
                model_name=model_name,
                task_type=task_type,
                metrics=metrics,
                train_score=round(train_acc, 4),
                test_score=round(test_acc, 4),
                cv_score_mean=round(cv_mean, 4),
                cv_score_std=round(cv_std, 4),
                primary_metric_name="Balanced Accuracy" if len(np.unique(y_test)) > 2 else "Accuracy",
                primary_metric_value=round(test_acc, 4),
                fit_time_seconds=fit_time,
                is_baseline=is_baseline,
            )

    @classmethod
    def _extract_feature_importances(cls, pipeline: Pipeline) -> list[FeatureImportance]:
        """Safely extract feature importance rankings from a trained Scikit-Learn Pipeline."""
        try:
            preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
            model = pipeline.named_steps["model"]

            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
            else:
                return []

            importances = None
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                coefs = model.coef_
                importances = np.abs(coefs).mean(axis=0) if coefs.ndim > 1 else np.abs(coefs)

            if importances is None or len(importances) != len(feature_names):
                return []

            fi_list = []
            for raw_name, imp in zip(feature_names, importances):
                # Clean prefix from ColumnTransformer (e.g. num__age -> age, cat__cat_col_B -> cat_col_B)
                clean_name = str(raw_name)
                for prefix in ("num__", "cat__", "dt__", "text__"):
                    if clean_name.startswith(prefix):
                        clean_name = clean_name[len(prefix):]
                fi_list.append(FeatureImportance(feature=clean_name, importance=round(float(imp), 4)))

            fi_list.sort(key=lambda x: x.importance, reverse=True)
            return fi_list[:30]
        except Exception as e:
            logger.debug("Feature importance extraction skipped: {}", str(e))
            return []
