"""Deterministic Dataset Profiler for AutoDS AI Studio.

Calculates comprehensive structural, statistical, and quality metrics
using Python/Pandas. Performs NO LLM calls itself.
"""

from datetime import datetime, timezone
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from core.constants import HIGH_MISSING_THRESHOLD, HIGH_CARDINALITY_THRESHOLD
from core.logging import logger
from core.schemas.dataset_profile import (
    ColumnProfile,
    DatasetProfile,
    DatasetQualityIssue,
)


class DatasetProfiler:
    """Calculates deterministic profile metadata from a Pandas DataFrame."""

    @classmethod
    def profile(
        cls, df: pd.DataFrame, filename: Optional[str] = None
    ) -> DatasetProfile:
        """Generate a complete DatasetProfile from a DataFrame.

        Args:
            df: Pandas DataFrame to analyze.
            filename: Optional dataset filename or identifier.

        Returns:
            Validated DatasetProfile object.
        """
        filename_str = filename or "dataset.csv"

        if df is None or not isinstance(df, pd.DataFrame):
            logger.error("DatasetProfiler received non-DataFrame input")
            return cls._empty_profile(filename_str)

        row_count = len(df)
        column_count = len(df.columns)

        if row_count == 0 or column_count == 0:
            logger.warning(
                "DatasetProfiler received empty DataFrame ({} rows x {} cols)",
                row_count,
                column_count,
            )
            return cls._empty_profile(filename_str, row_count=row_count, column_count=column_count)

        # Duplicate row calculation
        try:
            duplicate_row_count = int(df.duplicated().sum())
        except Exception:
            duplicate_row_count = 0
        duplicate_row_percentage = (
            round((duplicate_row_count / row_count) * 100.0, 2)
            if row_count > 0
            else 0.0
        )

        # Memory usage
        try:
            memory_bytes = df.memory_usage(deep=True).sum()
            memory_mb = round(float(memory_bytes) / (1024.0 * 1024.0), 2)
        except Exception:
            memory_mb = 0.0

        # Profile each column
        column_profiles: List[ColumnProfile] = []
        missing_columns: List[str] = []
        constant_columns: List[str] = []
        possible_id_columns: List[str] = []
        high_cardinality_columns: List[str] = []
        suspicious_columns: List[str] = []
        mixed_type_columns: List[str] = []
        detected_issues: List[DatasetQualityIssue] = []

        for col in df.columns:
            col_prof = cls._profile_column(df, col, row_count)
            column_profiles.append(col_prof)

            # Accumulate column category tags
            if col_prof.missing_count > 0:
                missing_columns.append(col_prof.name)
            if col_prof.is_constant:
                constant_columns.append(col_prof.name)
            if col_prof.is_possible_id:
                possible_id_columns.append(col_prof.name)
            if col_prof.is_high_cardinality:
                high_cardinality_columns.append(col_prof.name)

            # Check for mixed types in object/string series
            series = df[col]
            if pd.api.types.is_object_dtype(series):
                non_null_vals = series.dropna()
                if len(non_null_vals) > 0:
                    types_set = {type(v).__name__ for v in non_null_vals.iloc[:500]}
                    if len(types_set) > 1:
                        mixed_type_columns.append(col_prof.name)

            # Check for suspicious columns (100% missing, or non-finite values)
            if col_prof.missing_percentage == 100.0:
                suspicious_columns.append(col_prof.name)
            elif col_prof.is_numeric:
                try:
                    num_series = pd.to_numeric(series, errors="coerce")
                    if np.isinf(num_series).any():
                        suspicious_columns.append(col_prof.name)
                except Exception:
                    pass

        # Detect quality issues
        detected_issues = cls._detect_quality_issues(
            row_count=row_count,
            duplicate_row_count=duplicate_row_count,
            duplicate_row_percentage=duplicate_row_percentage,
            column_profiles=column_profiles,
        )

        # Calculate deterministic Quality Score (0 - 100)
        quality_score, quality_status = cls._calculate_quality_score(
            row_count=row_count,
            column_count=column_count,
            duplicate_row_percentage=duplicate_row_percentage,
            column_profiles=column_profiles,
        )

        profiled_at = datetime.now(timezone.utc).isoformat()

        profile = DatasetProfile(
            filename=filename_str,
            row_count=row_count,
            column_count=column_count,
            duplicate_row_count=duplicate_row_count,
            duplicate_row_percentage=duplicate_row_percentage,
            memory_usage_mb=memory_mb,
            quality_score=quality_score,
            quality_status=quality_status,
            columns=column_profiles,
            missing_columns=missing_columns,
            constant_columns=constant_columns,
            possible_id_columns=possible_id_columns,
            high_cardinality_columns=high_cardinality_columns,
            suspicious_columns=suspicious_columns,
            mixed_type_columns=mixed_type_columns,
            detected_issues=detected_issues,
            profiled_at=profiled_at,
        )

        logger.info(
            "DatasetProfile generated for '{}': {} rows x {} cols | Quality Score={:.1f} ({})",
            filename_str,
            row_count,
            column_count,
            quality_score,
            quality_status,
        )

        return profile

    @classmethod
    def _profile_column(
        cls, df: pd.DataFrame, col_name: Any, row_count: int
    ) -> ColumnProfile:
        """Profile a single DataFrame column."""
        name_str = str(col_name)
        series = df[col_name]
        raw_dtype = str(series.dtype)

        missing_count = int(series.isna().sum())
        non_null_count = row_count - missing_count
        missing_percentage = (
            round((missing_count / row_count) * 100.0, 2) if row_count > 0 else 0.0
        )

        try:
            unique_count = int(series.nunique(dropna=True))
        except Exception:
            unique_count = 0

        unique_percentage = (
            round((unique_count / row_count) * 100.0, 2) if row_count > 0 else 0.0
        )

        # Column Type Flags
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        is_boolean = pd.api.types.is_bool_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        is_categorical = (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ) and not is_boolean and not is_datetime

        is_constant = unique_count <= 1

        # Heuristic for possible ID column
        is_possible_id = cls._check_possible_id(name_str, unique_count, row_count, is_numeric)

        # High cardinality heuristic
        is_high_cardinality = False
        if is_categorical or pd.api.types.is_object_dtype(series):
            if unique_count > 30 and (unique_count / max(row_count, 1)) >= 0.50:
                is_high_cardinality = True

        # Compute type-specific statistics
        min_val = None
        max_val = None
        mean_val = None
        median_val = None
        std_val = None
        standard_deviation = None
        q1_val = None
        q3_val = None
        top_values = None
        top_value_counts = None
        cardinality = unique_count if (is_categorical or pd.api.types.is_object_dtype(series)) else None
        min_date = None
        max_date = None

        if is_numeric and non_null_count > 0:
            try:
                # Filter out non-finite values (inf, -inf, nan) before calculating numeric statistics
                finite_series = series.replace([np.inf, -np.inf], np.nan).dropna()
                if len(finite_series) > 0:
                    min_val = cls._safe_float(finite_series.min())
                    max_val = cls._safe_float(finite_series.max())
                    mean_val = cls._safe_float(finite_series.mean())
                    median_val = cls._safe_float(finite_series.median())
                    std_val = cls._safe_float(finite_series.std()) if len(finite_series) > 1 else 0.0
                    standard_deviation = std_val
                    q1_val = cls._safe_float(finite_series.quantile(0.25))
                    q3_val = cls._safe_float(finite_series.quantile(0.75))
            except Exception as e:
                logger.debug("Error computing numeric stats for '{}': {}", name_str, str(e))

        if is_datetime and non_null_count > 0:
            try:
                min_date_val = series.min()
                max_date_val = series.max()
                if pd.notna(min_date_val):
                    min_date = str(min_date_val)
                if pd.notna(max_date_val):
                    max_date = str(max_date_val)
            except Exception:
                pass

        if (is_categorical or is_boolean or not is_numeric) and non_null_count > 0:
            try:
                vc = series.value_counts(dropna=True).head(5)
                top_values = {str(k): int(v) for k, v in vc.items()}
                top_value_counts = top_values
            except Exception:
                top_values = None
                top_value_counts = None

        # Semantic Datatype Inference (Section 3 & 4)
        from services.repair.auto_dtypes import infer_semantic_datatype
        type_inf = infer_semantic_datatype(series)

        return ColumnProfile(
            name=name_str,
            dtype=cls._friendly_dtype_name(series.dtype),
            pandas_dtype=raw_dtype,
            non_null_count=non_null_count,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            is_numeric=is_numeric,
            is_categorical=is_categorical,
            is_datetime=is_datetime,
            is_boolean=is_boolean,
            is_constant=is_constant,
            is_possible_id=is_possible_id,
            is_high_cardinality=is_high_cardinality,
            inferred_semantic_type=type_inf.detected_type,
            sample_values=type_inf.sample_values,
            suspicious_values=type_inf.suspicious_values,
            format_patterns=type_inf.format_patterns,
            possible_datatype=type_inf.suggested_action,
            datatype_confidence=type_inf.confidence,
            datatype_evidence=type_inf.evidence,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            median_val=median_val,
            std_val=std_val,
            standard_deviation=standard_deviation,
            q1_val=q1_val,
            q3_val=q3_val,
            top_values=top_values,
            top_value_counts=top_value_counts,
            cardinality=cardinality,
            min_date=min_date,
            max_date=max_date,
        )

    @classmethod
    def _check_possible_id(
        cls, name: str, unique_count: int, row_count: int, is_numeric: bool
    ) -> bool:
        """Heuristic check if a column represents a primary key or unique identifier."""
        name_lower = name.lower()
        id_patterns = [r"\bid\b", r"_id$", r"^id_", r"uuid", r"guid", r"index", r"code", r"number", r"no$"]
        matches_name = any(re.search(pat, name_lower) for pat in id_patterns)

        if row_count > 0:
            uniqueness_ratio = unique_count / row_count
            if matches_name and uniqueness_ratio >= 0.80:
                return True
            if not is_numeric and uniqueness_ratio >= 0.98 and unique_count > 10:
                return True

        return False

    @classmethod
    def _detect_quality_issues(
        cls,
        row_count: int,
        duplicate_row_count: int,
        duplicate_row_percentage: float,
        column_profiles: List[ColumnProfile],
    ) -> List[DatasetQualityIssue]:
        """Detect deterministic data quality issues."""
        issues: List[DatasetQualityIssue] = []

        # Duplicate rows issue
        if duplicate_row_count > 0:
            severity = "critical" if duplicate_row_percentage > 20.0 else ("high" if duplicate_row_percentage > 5.0 else "medium")
            issues.append(
                DatasetQualityIssue(
                    issue_type="duplicate_rows",
                    severity=severity,
                    description=f"Dataset contains {duplicate_row_count} duplicate rows ({duplicate_row_percentage:.1f}%).",
                    metric_value=f"{duplicate_row_percentage:.1f}%",
                )
            )

        # Column-level issues
        for col in column_profiles:
            if col.missing_count > 0:
                if col.missing_percentage > 50.0:
                    sev = "critical"
                elif col.missing_percentage > 20.0:
                    sev = "high"
                elif col.missing_percentage > 5.0:
                    sev = "medium"
                else:
                    sev = "low"
                issues.append(
                    DatasetQualityIssue(
                        issue_type="missing_values",
                        column=col.name,
                        severity=sev,
                        description=f"Column '{col.name}' has {col.missing_count} missing values ({col.missing_percentage:.1f}%).",
                        metric_value=f"{col.missing_percentage:.1f}%",
                    )
                )

            if col.is_constant:
                issues.append(
                    DatasetQualityIssue(
                        issue_type="constant_column",
                        column=col.name,
                        severity="high",
                        description=f"Column '{col.name}' is constant with only {col.unique_count} unique value(s).",
                        metric_value=f"{col.unique_count} unique",
                    )
                )

            if col.is_high_cardinality:
                issues.append(
                    DatasetQualityIssue(
                        issue_type="high_cardinality",
                        column=col.name,
                        severity="medium",
                        description=f"Categorical column '{col.name}' has high cardinality ({col.unique_count} unique values).",
                        metric_value=f"{col.unique_count} unique",
                    )
                )

        return issues

    @classmethod
    def _calculate_quality_score(
        cls,
        row_count: int,
        column_count: int,
        duplicate_row_percentage: float,
        column_profiles: List[ColumnProfile],
    ) -> Tuple[float, str]:
        """Calculate a transparent, deterministic quality score between 0 and 100."""
        if row_count == 0 or column_count == 0:
            return 0.0, "Critical"

        score = 100.0

        # 1. Missing cells penalty (max 30 points)
        total_cells = row_count * column_count
        total_missing = sum(c.missing_count for c in column_profiles)
        overall_missing_pct = (total_missing / total_cells) * 100.0 if total_cells > 0 else 0.0
        missing_penalty = min(30.0, overall_missing_pct * 1.5)

        # Extra penalty for severely missing columns (>50% missing)
        severely_missing_cols = sum(1 for c in column_profiles if c.missing_percentage > 50.0)
        missing_penalty += min(15.0, severely_missing_cols * 5.0)
        score -= min(35.0, missing_penalty)

        # 2. Duplicate rows penalty (max 25 points)
        dup_penalty = min(25.0, duplicate_row_percentage * 1.2)
        score -= dup_penalty

        # 3. Constant columns penalty (max 20 points)
        constant_cols = sum(1 for c in column_profiles if c.is_constant)
        constant_penalty = min(20.0, constant_cols * 5.0)
        score -= constant_penalty

        # 4. High cardinality penalty (max 15 points)
        high_card_cols = sum(1 for c in column_profiles if c.is_high_cardinality)
        high_card_penalty = min(15.0, high_card_cols * 3.0)
        score -= high_card_penalty

        # Clamp score to [0, 100]
        final_score = round(max(0.0, min(100.0, score)), 1)

        if final_score >= 90.0:
            status = "Excellent"
        elif final_score >= 75.0:
            status = "Good"
        elif final_score >= 60.0:
            status = "Fair"
        elif final_score >= 40.0:
            status = "Poor"
        else:
            status = "Critical"

        return final_score, status

    @staticmethod
    def _friendly_dtype_name(dtype: Any) -> str:
        """Return a user-friendly data type name."""
        s = str(dtype).lower()
        if "int" in s:
            return "Integer"
        elif "float" in s:
            return "Float"
        elif "bool" in s:
            return "Boolean"
        elif "datetime" in s:
            return "Datetime"
        elif "category" in s:
            return "Category"
        elif "object" in s or "string" in s:
            return "String/Text"
        return str(dtype)

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        """Safely convert a pandas scalar to float, returning None if NaN/Inf."""
        if pd.isna(val) or val is None:
            return None
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return None
            return round(f, 4)
        except (ValueError, TypeError):
            return None

    @classmethod
    def _empty_profile(
        cls, filename: str, row_count: int = 0, column_count: int = 0
    ) -> DatasetProfile:
        """Build a fallback DatasetProfile for an empty or invalid dataset."""
        return DatasetProfile(
            filename=filename,
            row_count=row_count,
            column_count=column_count,
            duplicate_row_count=0,
            duplicate_row_percentage=0.0,
            memory_usage_mb=0.0,
            quality_score=0.0,
            quality_status="Critical",
            columns=[],
            missing_columns=[],
            constant_columns=[],
            possible_id_columns=[],
            high_cardinality_columns=[],
            suspicious_columns=[],
            mixed_type_columns=[],
            detected_issues=[
                DatasetQualityIssue(
                    issue_type="empty_dataset",
                    severity="critical",
                    description="The dataset contains zero rows or zero columns.",
                )
            ],
            profiled_at=datetime.now(timezone.utc).isoformat(),
        )
