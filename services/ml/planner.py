"""ML Planner — detects task type, validates target quality, and analyzes features for AutoML."""

from typing import Any, Optional
import numpy as np
import pandas as pd

from core.logging import logger
from models.ml import MLTaskType
from utils.dataframe import is_numeric_column


class MLPlanner:
    """Helper for planning ML tasks on clean datasets with strict feature analysis."""

    @classmethod
    def detect_task_type(cls, df: pd.DataFrame, target_column: str) -> MLTaskType:
        """Automatically detect if the target column represents classification or regression."""
        if target_column not in df.columns:
            return MLTaskType.REGRESSION

        series = df[target_column].dropna()
        if len(series) == 0:
            return MLTaskType.REGRESSION

        n_unique = series.nunique()

        # Binary classification
        if n_unique == 2:
            return MLTaskType.BINARY_CLASSIFICATION

        # Non-numeric string/category is always classification
        if not is_numeric_column(df, target_column):
            return MLTaskType.MULTICLASS_CLASSIFICATION

        # Discrete numeric targets with low cardinality
        if n_unique <= 15 and (n_unique / max(len(series), 1)) < 0.05:
            return MLTaskType.MULTICLASS_CLASSIFICATION

        return MLTaskType.REGRESSION

    @classmethod
    def check_target_quality(cls, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        """Validate target column quality, missingness, and variation before modeling."""
        if target_column not in df.columns:
            return {
                "is_valid": False,
                "reason": f"Target column '{target_column}' does not exist in dataset",
                "missing_count": len(df),
                "valid_count": 0,
                "unique_count": 0,
            }

        total_rows = len(df)
        valid_series = df[target_column].dropna()
        missing_count = total_rows - len(valid_series)
        valid_count = len(valid_series)
        unique_count = valid_series.nunique() if valid_count > 0 else 0

        if valid_count < 10:
            return {
                "is_valid": False,
                "reason": f"Insufficient data: only {valid_count} non-null target values available (minimum 10 required)",
                "missing_count": missing_count,
                "valid_count": valid_count,
                "unique_count": unique_count,
            }

        if unique_count <= 1:
            return {
                "is_valid": False,
                "reason": "Insufficient target variation for reliable modeling (target contains only 1 unique value)",
                "missing_count": missing_count,
                "valid_count": valid_count,
                "unique_count": unique_count,
            }

        if is_numeric_column(df, target_column):
            num_vals = pd.to_numeric(valid_series, errors="coerce").dropna()
            if len(num_vals) > 0:
                var_val = float(np.var(num_vals))
                if var_val == 0.0:
                    return {
                        "is_valid": False,
                        "reason": "Insufficient target variation for reliable modeling (target variance is 0.0)",
                        "missing_count": missing_count,
                        "valid_count": valid_count,
                        "unique_count": unique_count,
                    }

        class_distribution = {}
        if not is_numeric_column(df, target_column) or unique_count <= 20:
            class_counts = valid_series.value_counts().to_dict()
            class_distribution = {str(k): int(v) for k, v in class_counts.items()}

        return {
            "is_valid": True,
            "reason": "Target passed quality and variation checks",
            "missing_count": missing_count,
            "valid_count": valid_count,
            "unique_count": unique_count,
            "class_distribution": class_distribution,
        }

    @classmethod
    def analyze_features(cls, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        """Categorize features into numeric, categorical, datetime, text, and identifier/constant dropped features."""
        total_rows = len(df)
        feature_df = df.drop(columns=[target_column], errors="ignore")

        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        datetime_cols: list[str] = []
        text_cols: list[str] = []
        dropped_features: dict[str, str] = {}

        for col in feature_df.columns:
            s = feature_df[col]
            valid_s = s.dropna()
            n_unique = valid_s.nunique()

            # 1. Constant Column Check
            if n_unique <= 1:
                dropped_features[col] = "Constant feature (zero variance)"
                continue

            # 2. Datetime Column Check (datetime dtype or parseable mixed dates via analyze_date_column)
            is_dt = pd.api.types.is_datetime64_any_dtype(s)
            if not is_dt and (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)):
                try:
                    from services.repair.dates import analyze_date_column
                    diag = analyze_date_column(s)
                    total_valid = diag.unambiguous_count + diag.ambiguous_count
                    if diag.total_non_null > 0 and (total_valid / max(diag.total_non_null, 1)) >= 0.5:
                        is_dt = True
                except Exception:
                    sample_dt = pd.to_datetime(valid_s.head(50), errors="coerce")
                    if len(sample_dt) > 0 and sample_dt.notna().sum() / len(sample_dt) >= 0.5:
                        is_dt = True

            if is_dt:
                datetime_cols.append(col)
                continue

            # 3. Numeric Column Check
            is_num = is_numeric_column(df, col)
            if not is_num and (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
                clean_num_s = pd.to_numeric(valid_s.astype(str).str.replace(r"[$,%]", "", regex=True), errors="coerce")
                if len(clean_num_s) > 0 and clean_num_s.notna().sum() / len(clean_num_s) >= 0.75:
                    is_num = True

            if is_num:
                numeric_cols.append(col)
                continue

            # 4. High-Uniqueness Identifier Check (for non-numeric string/category columns)
            uniq_ratio = n_unique / max(len(valid_s), 1)
            col_lower = str(col).lower()
            is_id_keyword = any(kw in col_lower for kw in ["id", "uuid", "guid", "index", "key", "code"])

            if total_rows >= 20 and (uniq_ratio > 0.95 or is_id_keyword):
                dropped_features[col] = "Likely identifier column (high uniqueness)"
                continue

            # 5. Free-Form Text Column Check
            if n_unique > 50:
                sample_str = valid_s.astype(str).head(50)
                avg_len = sample_str.str.len().mean()
                avg_words = sample_str.str.split().str.len().mean()
                if avg_len > 30 or avg_words > 3:
                    text_cols.append(col)
                    continue

            # 6. Categorical Column Check
            categorical_cols.append(col)

        return {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "text_cols": text_cols,
            "dropped_features": dropped_features,
            "total_features_analyzed": len(feature_df.columns),
            "usable_features_count": len(numeric_cols) + len(categorical_cols) + len(datetime_cols) + len(text_cols),
        }

    @classmethod
    def generate_ml_readiness_report(cls, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        """Generate a complete ML readiness report for UI display and audit logging."""
        task_type = cls.detect_task_type(df, target_column)
        target_info = cls.check_target_quality(df, target_column)
        feature_info = cls.analyze_features(df, target_column)

        report = {
            "target_column": target_column,
            "task_type": task_type.value,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "target_quality": target_info,
            "feature_analysis": feature_info,
            "is_ready_for_ml": target_info["is_valid"] and feature_info["usable_features_count"] > 0,
        }

        if not target_info["is_valid"]:
            report["not_ready_reason"] = target_info["reason"]
        elif feature_info["usable_features_count"] == 0:
            report["not_ready_reason"] = "No usable predictive features found after excluding identifiers and constant columns"

        return report

    @classmethod
    def suggest_target_columns(cls, df: pd.DataFrame) -> list[str]:
        """Suggest candidate target columns from the dataset using value characteristics."""
        candidates: list[str] = []
        for col in df.columns:
            s = df[col].dropna()
            if len(s) < 10:
                continue
            # Exclude high-cardinality ID columns
            uniq_ratio = s.nunique() / max(len(s), 1)
            col_lower = str(col).lower()
            if uniq_ratio > 0.95 and len(s) > 20:
                if "id" in col_lower or "uuid" in col_lower or "guid" in col_lower or "key" in col_lower:
                    continue
            # Must have variation
            if s.nunique() <= 1:
                continue
            candidates.append(col)

        # Sort with target-like keyword priority as secondary preference
        priority_keywords = ["target", "label", "price", "churn", "salary", "revenue", "class", "outcome", "survived", "status", "score"]
        candidates.sort(
            key=lambda c: any(kw in c.lower() for kw in priority_keywords),
            reverse=True,
        )
        return candidates
