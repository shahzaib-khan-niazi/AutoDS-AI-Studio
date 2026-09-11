"""Autonomous Preprocessing Pipeline Service.

Executes end-to-end dataset preprocessing completely autonomously in progressive stages:
Stage 1: Structural reshaping (if multi-header / wide / pivot)
Stage 2: Deterministic cleaning (duplicates, column sanitization, empty rows/cols, high-null cols, mixed types)
Stage 3: Dtype-aware imputation & outlier handling
Stage 4: Validation & Quality Score audit
"""

import time
from typing import Any, Optional
import pandas as pd
from pydantic import BaseModel, Field

from core.logging import logger
from core.config import config
from models.repair import RepairAction, RepairOperation, RepairRecord, RepairResult
from models.inspection import InspectionResult
from models.structure import StructureResult, StructureType
from services.inspector.service import InspectorService
from services.structure.service import StructureService
from services.repair.service import RepairService
from services.repair.structural import (
    flatten_headers,
    unpivot,
    detect_multi_value_cells,
    explode_multi_value_cells,
    remove_repeated_headers,
    remove_metadata_rows,
)
from services.ai.planner import AIPlanner
from utils.dataframe import safe_copy


class AutonomousPipelineResult(BaseModel):
    """Result summary of the autonomous preprocessing pipeline."""

    success: bool = True
    initial_rows: int = 0
    final_rows: int = 0
    initial_columns: int = 0
    final_columns: int = 0
    initial_quality_score: float = 0.0
    final_quality_score: float = 0.0
    quality_improvement: float = 0.0
    actions_executed: list[str] = Field(default_factory=list)
    records: list[RepairRecord] = Field(default_factory=list)
    summary_bullet_points: list[str] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class AutonomousPipelineService:
    """Service for autonomous 1-click dataset preprocessing."""

    @classmethod
    def run_auto_preprocess(
        cls,
        df: pd.DataFrame,
        use_ai_if_available: bool = True,
    ) -> tuple[pd.DataFrame, AutonomousPipelineResult]:
        """Run the complete autonomous preprocessing pipeline.

        Args:
            df: Raw input DataFrame.
            use_ai_if_available: Whether to query AI planner if API key is configured.

        Returns:
            Tuple of (Cleaned DataFrame, AutonomousPipelineResult).
        """
        start_time = time.time()
        logger.info("Initiating Autonomous Auto-Preprocessing on {}×{} dataset", len(df), len(df.columns))

        current_df = safe_copy(df)
        all_records: list[RepairRecord] = []
        actions_applied: list[str] = []
        all_warnings: list[str] = []

        # 1. Initial Inspection & Structure Detection
        initial_inspection = InspectorService.inspect(current_df)
        initial_structure = StructureService.detect(current_df)
        initial_score = initial_inspection.quality.quality_score

        # ── STAGE 1: Structural Reshaping & Record Unpacking ──
        # 1a. Check for packed multi-value delimiter cells (e.g. Category | Amount)
        multi_val_detection = detect_multi_value_cells(current_df)
        if multi_val_detection and multi_val_detection.get("confidence", 0) >= 0.85:
            logger.info("Stage 1: Exploding packed multi-value records...")
            current_df, exp_rec = explode_multi_value_cells(
                current_df,
                columns=multi_val_detection.get("synchronized_columns", multi_val_detection.get("multi_value_columns")),
                delimiter=multi_val_detection.get("delimiter", "|"),
                fill_mismatched=True,
            )
            all_records.append(exp_rec)
            actions_applied.append("explode_multi_value_cells")
            if exp_rec.warnings:
                all_warnings.extend(exp_rec.warnings)

        # 1b. Check for repeated headers in data body
        if len(current_df) > 2:
            current_df, rep_hdr_rec = remove_repeated_headers(current_df)
            if rep_hdr_rec.details.get("dropped_count", 0) > 0:
                all_records.append(rep_hdr_rec)
                actions_applied.append("remove_repeated_headers")

        # 1c. Check for summary / metadata trailing rows
        if len(current_df) > 2:
            current_df, meta_rec = remove_metadata_rows(current_df)
            if meta_rec.details.get("dropped_count", 0) > 0:
                all_records.append(meta_rec)
                actions_applied.append("remove_metadata_rows")

        # 1c. Multi-header / Wide-table unpivot
        if initial_structure.structure == StructureType.MULTI_HEADER and initial_structure.confidence >= 0.5:
            logger.info("Stage 1: Flattening multi-level headers...")
            current_df, flat_rec = flatten_headers(current_df, header_rows=1)
            all_records.append(flat_rec)
            actions_applied.append("flatten_headers")

            # Check if now wide table needing unpivot
            post_flat_struct = StructureService.detect(current_df)
            if post_flat_struct.structure in (StructureType.WIDE_TABLE, StructureType.PIVOT_TABLE, StructureType.CROSSTAB):
                first_col = current_df.columns[0]
                current_df, unp_rec = unpivot(
                    current_df,
                    id_columns=[first_col],
                    var_name="Category",
                    value_name="Value",
                    split_delimiter=" - ",
                    drop_null_values=True,
                )
                all_records.append(unp_rec)
                actions_applied.append("unpivot")

        elif initial_structure.structure in (StructureType.WIDE_TABLE, StructureType.PIVOT_TABLE, StructureType.CROSSTAB) and initial_structure.confidence >= 0.6:
            logger.info("Stage 1: Unpivoting wide table...")
            first_col = current_df.columns[0]
            current_df, unp_rec = unpivot(
                current_df,
                id_columns=[first_col],
                var_name="Category",
                value_name="Value",
                split_delimiter=" - ",
                drop_null_values=True,
            )
            all_records.append(unp_rec)
            actions_applied.append("unpivot")

        # ── STAGE 2: Deterministic / AI Cleaning Pipeline ──
        # Re-inspect on the normalized schema
        stage2_inspection = InspectorService.inspect(current_df)
        stage2_structure = StructureService.detect(current_df)

        stage2_actions = cls._formulate_stage2_plan(
            df=current_df,
            inspection=stage2_inspection,
            structure=stage2_structure,
            use_ai=use_ai_if_available and config.ai.is_configured,
        )

        if stage2_actions:
            logger.info("Stage 2: Executing {} cleaning action(s)...", len(stage2_actions))
            current_df, rep_res = RepairService.repair(current_df, stage2_actions)
            all_records.extend(rep_res.records)
            actions_applied.extend(rep_res.actions_applied)
            all_warnings.extend(rep_res.warnings)

        # ── STAGE 3: Final Inspection & Quality Gain ──
        final_inspection = InspectorService.inspect(current_df)
        final_score = final_inspection.quality.quality_score
        improvement = max(0.0, final_score - initial_score)

        # ── STAGE 4: Summary Bullets ──
        bullets = cls._generate_summary_bullets(
            df_orig=df,
            df_clean=current_df,
            initial_inspection=initial_inspection,
            final_inspection=final_inspection,
            records=all_records,
        )

        elapsed = round(time.time() - start_time, 2)
        logger.info(
            "Autonomous Preprocessing completed in {}s | Quality: {:.0%} -> {:.0%}",
            elapsed, initial_score, final_score
        )

        result = AutonomousPipelineResult(
            success=True,
            initial_rows=len(df),
            final_rows=len(current_df),
            initial_columns=len(df.columns),
            final_columns=len(current_df.columns),
            initial_quality_score=initial_score,
            final_quality_score=final_score,
            quality_improvement=improvement,
            actions_executed=actions_applied,
            records=all_records,
            summary_bullet_points=bullets,
            execution_time_seconds=elapsed,
            warnings=all_warnings,
        )

        return current_df, result

    @classmethod
    def evaluate_ai_plan_for_auto_repair(
        cls,
        df: pd.DataFrame,
        ai_plan: Any,
        confidence_threshold: float = 0.85,
    ) -> list[RepairAction]:
        """Evaluate AI proposed actions against the 6 auto-repair criteria and deterministic evidence."""
        ai_repair_actions = AIPlanner.convert_ai_actions_to_repair_actions(ai_plan)
        if not ai_repair_actions:
            return []

        validated_ai_actions: list[RepairAction] = []
        from services.repair.auto_dtypes import _is_identifier_column
        from services.repair.dates import analyze_date_column

        for act in ai_repair_actions:
            targets = act.target or []
            is_safe_to_auto_apply = True

            for t in targets:
                if t not in df.columns:
                    is_safe_to_auto_apply = False
                    break

                # Guard against converting or modifying identifiers
                if act.operation in (RepairOperation.CONVERT_TYPES, RepairOperation.AUTO_DTYPES, RepairOperation.STANDARDIZE_VALUES):
                    if _is_identifier_column(df[t]):
                        logger.info("Rejected AI action for identifier column '{}'", t)
                        is_safe_to_auto_apply = False
                        break

                # Guard against ambiguous dates
                if act.operation in (RepairOperation.NORMALIZE_DATES, RepairOperation.CONVERT_TYPES):
                    date_diag = analyze_date_column(df[t])
                    if date_diag.is_ambiguous:
                        logger.warning("Rejected AI date conversion for strictly ambiguous column '{}'", t)
                        is_safe_to_auto_apply = False
                        break

            if is_safe_to_auto_apply and act.confidence >= confidence_threshold:
                validated_ai_actions.append(act)
            else:
                logger.info("Flagged uncertain/ambiguous AI action for review: {}", act.operation.value)

        return validated_ai_actions

    @classmethod
    def _formulate_stage2_plan(
        cls,
        df: pd.DataFrame,
        inspection: InspectionResult,
        structure: StructureResult,
        use_ai: bool,
    ) -> list[RepairAction]:
        """Formulate a prioritized list of cleaning actions for the normalized schema."""
        plan_actions: list[RepairAction] = []

        # If AI is available and configured, consult AI for planning, but validate against deterministic evidence!
        # RULE 1: Never auto-repair based on AI confidence alone.
        # AUTO-REPAIR only when:
        # 1. Deterministic evidence is strong
        # 2. AI agrees when AI is required
        # 3. No ambiguity exists
        # 4. No meaningful data loss is detected
        # 5. Validation passes
        # 6. Confidence >= configured threshold
        if use_ai:
            try:
                ai_plan = AIPlanner.generate_repair_plan(df, inspection=inspection, structure=structure)
                validated_ai_actions = cls.evaluate_ai_plan_for_auto_repair(df, ai_plan)
                if validated_ai_actions:
                    logger.info("Using {} validated AI actions (gated by deterministic evidence)", len(validated_ai_actions))
                    return validated_ai_actions
            except Exception as e:
                logger.warning("AI planning fallback to deterministic auto-plan: {}", str(e))

        # Deterministic Plan (Gated by 6 Auto-Repair Criteria):
        quality = inspection.quality

        # Step 1: Remove duplicate rows
        if quality.duplicate_rows > 0:
            plan_actions.append(
                RepairAction(
                    operation=RepairOperation.REMOVE_DUPLICATE_ROWS,
                    reason=f"Eliminate {quality.duplicate_rows} duplicate row(s)",
                )
            )

        # Step 2: Sanitize column names
        if quality.suspicious_column_names:
            plan_actions.append(
                RepairAction(
                    operation=RepairOperation.RENAME_COLUMNS,
                    reason="Fix unnamed, duplicate, or unstripped column names",
                )
            )

        # Step 3: Drop empty rows
        if quality.empty_rows > 0:
            plan_actions.append(
                RepairAction(
                    operation=RepairOperation.DROP_EMPTY_ROWS,
                    reason=f"Drop {quality.empty_rows} completely empty row(s)",
                )
            )

        # Step 4: Drop empty columns
        if quality.empty_columns:
            plan_actions.append(
                RepairAction(
                    operation=RepairOperation.DROP_EMPTY_COLUMNS,
                    target=quality.empty_columns,
                    reason=f"Drop {len(quality.empty_columns)} completely empty column(s)",
                )
            )

        # Step 5: Drop high-missing columns (>80% missing)
        total_rows = len(df)
        if total_rows > 10:
            high_null = [
                c for c in df.columns
                if (df[c].isna().sum() / total_rows) >= 0.8 and not df[c].isna().all()
            ]
            if high_null:
                plan_actions.append(
                    RepairAction(
                        operation=RepairOperation.DROP_HIGH_MISSING_COLUMNS,
                        target=high_null,
                        parameters={"threshold": 0.8},
                        reason=f"Drop {len(high_null)} column(s) with >80% missing data",
                    )
                )

        # Step 6: Strip whitespace from all string values
        has_whitespace_issues = False
        for col in df.columns:
            if df[col].dtype == "object":
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    str_vals = non_null.astype(str)
                    if (str_vals != str_vals.str.strip()).any() or str_vals.str.contains(r"  +", regex=True).any():
                        has_whitespace_issues = True
                        break
        if has_whitespace_issues:
            plan_actions.append(
                RepairAction(
                    operation=RepairOperation.STRIP_WHITESPACE,
                    reason="Strip leading/trailing whitespace and collapse multiple spaces",
                )
            )

        # Step 7: Standardize inconsistent categorical values
        from services.repair.standardize import detect_inconsistent_columns
        inconsistent_cols = detect_inconsistent_columns(df)
        if inconsistent_cols:
            plan_actions.append(
                RepairAction(
                    operation=RepairOperation.STANDARDIZE_VALUES,
                    target=inconsistent_cols,
                    reason=f"Standardize {len(inconsistent_cols)} column(s) with inconsistent values",
                )
            )

        # Step 8: Auto-detect and fix wrong dtypes
        from services.repair.auto_dtypes import detect_mistyped_columns
        mistyped = detect_mistyped_columns(df)
        if mistyped:
            plan_actions.append(
                RepairAction(
                    operation=RepairOperation.AUTO_DTYPES,
                    target=[m["column"] for m in mistyped],
                    reason=f"Auto-correct dtypes for {len(mistyped)} mistyped column(s)",
                )
            )

        # Step 9: Convert mixed/object columns that are actually numeric or datetime
        for col_detail in inspection.column_details:
            if col_detail.is_mixed_type and col_detail.name in df.columns:
                from services.structure.utils import numeric_ratio, datetime_ratio
                num_r = numeric_ratio(df, col_detail.name)
                if num_r > 0.8:
                    plan_actions.append(
                        RepairAction(
                            operation=RepairOperation.CONVERT_TYPES,
                            target=[col_detail.name],
                            parameters={"target_type": "numeric"},
                            reason=f"Convert mixed column '{col_detail.name}' to numeric ({num_r:.0%} parseable)",
                        )
                    )
                else:
                    dt_r = datetime_ratio(df, col_detail.name)
                    if dt_r > 0.8:
                        plan_actions.append(
                            RepairAction(
                                operation=RepairOperation.CONVERT_TYPES,
                                target=[col_detail.name],
                                parameters={"target_type": "datetime"},
                                reason=f"Convert mixed column '{col_detail.name}' to datetime ({dt_r:.0%} parseable)",
                            )
                        )

        # Step 10: Impute remaining missing values
        if quality.missing_cells > 0:
            plan_actions.append(
                RepairAction(
                    operation=RepairOperation.FILL_MISSING,
                    parameters={"strategy": "auto"},
                    reason="Dtype-aware imputation (median for numeric, mode for categorical)",
                )
            )

        # Step 11: Outlier capping for numeric features
        plan_actions.append(
            RepairAction(
                operation=RepairOperation.HANDLE_OUTLIERS,
                parameters={"method": "iqr", "factor": 1.5, "action": "cap"},
                reason="Statistical IQR capping on extreme numeric outliers",
            )
        )

        return plan_actions

    @staticmethod
    def _generate_summary_bullets(
        df_orig: pd.DataFrame,
        df_clean: pd.DataFrame,
        initial_inspection: InspectionResult,
        final_inspection: InspectionResult,
        records: list[RepairRecord],
    ) -> list[str]:
        """Generate human-readable summary bullets of all automated repairs."""
        bullets: list[str] = []

        for rec in records:
            if not rec.success:
                continue
            op = rec.operation
            det = rec.details

            if op == "remove_duplicate_rows":
                n_dup = det.get("duplicates_removed", rec.rows_before - rec.rows_after)
                if n_dup > 0:
                    bullets.append(f"Removed **{n_dup}** duplicate row(s)")
            elif op == "rename_columns":
                n_renamed = det.get("total_renamed", 0)
                if n_renamed > 0:
                    bullets.append(f"Sanitized & resolved **{n_renamed}** problematic column name(s)")
            elif op == "drop_empty_rows":
                n_empty = det.get("empty_rows_dropped", 0)
                if n_empty > 0:
                    bullets.append(f"Dropped **{n_empty}** completely empty row(s)")
            elif op == "drop_empty_columns":
                cols = det.get("empty_columns_dropped", [])
                if cols:
                    bullets.append(f"Dropped **{len(cols)}** entirely null column(s): `{', '.join(cols)}`")
            elif op == "drop_high_missing_columns":
                cols = det.get("dropped_columns", [])
                if cols:
                    bullets.append(f"Dropped **{len(cols)}** column(s) with >80% missing data: `{', '.join(cols)}`")
            elif op == "fill_missing":
                filled = det.get("cells_filled", 0)
                if filled > 0:
                    bullets.append(f"Imputed **{filled}** missing cell(s) across {len(det.get('columns_imputed', {}))} columns")
            elif op == "handle_outliers":
                n_out = det.get("total_outliers_handled", 0)
                if n_out > 0:
                    bullets.append(f"Capped **{n_out}** extreme statistical outlier value(s)")
            elif op == "convert_types":
                conv = det.get("converted", [])
                if conv:
                    bullets.append(f"Converted **{len(conv)}** column(s) to safe `{det.get('target_type')}` type: `{', '.join(conv)}`")
            elif op == "unpivot":
                bullets.append(f"Reshaped wide table to tidy format ({rec.rows_before}×{rec.columns_before} -> {rec.rows_after}×{rec.columns_after})")
            elif op == "flatten_headers":
                bullets.append("Promoted top metadata row(s) to formal table headers")
            elif op == "strip_whitespace":
                n_cells = det.get("total_cells_modified", 0)
                n_cols = len(det.get("columns_cleaned", {}))
                if n_cells > 0:
                    bullets.append(f"Stripped whitespace from **{n_cells}** cell(s) across {n_cols} column(s)")
            elif op == "standardize_values":
                n_cells = det.get("total_cells_modified", 0)
                n_cols = len(det.get("columns_standardized", {}))
                if n_cells > 0:
                    bullets.append(f"Standardized **{n_cells}** inconsistent value(s) across {n_cols} column(s)")
            elif op == "auto_dtypes":
                n_conv = det.get("total_converted", 0)
                conversions = det.get("conversions", {})
                if n_conv > 0:
                    conv_strs = [f"`{c}`→{info['to']}" for c, info in conversions.items()]
                    bullets.append(f"Auto-corrected dtypes for **{n_conv}** column(s): {', '.join(conv_strs)}")

        init_score = initial_inspection.quality.quality_score
        final_score = final_inspection.quality.quality_score
        bullets.append(f"Elevated Overall Data Quality Score: **{init_score:.0%}** -> **{final_score:.0%}**")

        return bullets
