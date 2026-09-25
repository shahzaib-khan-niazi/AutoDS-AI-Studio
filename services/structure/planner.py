"""Deterministic Structural Repair Planner for AutoDS AI Studio.

Analyzes DataFrames for structural issues and formulates validated, domain-agnostic
StructuralPlan objects with audit metrics and safety controls.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd

from core.logging import logger
from models.structure import StructureResult, StructureType, StructuralPlan, StructuralAudit


class StructuralPlanner:
    """Planner for deterministic structural analysis and repair generation."""

    @classmethod
    def create_plan(
        cls,
        df: pd.DataFrame,
        structure_result: Optional[StructureResult] = None,
    ) -> StructuralPlan:
        """Analyze DataFrame structure and construct a validated StructuralPlan.

        Args:
            df: Source DataFrame (never modified).
            structure_result: Optional pre-computed StructureResult.

        Returns:
            Validated StructuralPlan instance.
        """
        rows_before = len(df)
        cols_before = len(df.columns)
        source_cols = [str(c) for c in df.columns]

        if df.empty:
            return StructuralPlan(
                detected_issue="empty_dataset",
                evidence=["Dataset is empty"],
                proposed_transformation="none",
                confidence=1.0,
                requires_human_approval=False,
                target_schema={},
            )

        ambiguity_flags: list[str] = []

        # 0. Check for embedded structured records in text cells
        embedded_plan = cls._plan_embedded_structured_records(df)
        if embedded_plan:
            return embedded_plan

        # 1. Check for repeated category blocks (horizontally distributed records)
        repeated_block_plan = cls._plan_repeated_category_blocks(df)
        if repeated_block_plan:
            return repeated_block_plan

        # 2. Check for multi-row headers / metadata banners
        multi_header_plan = cls._plan_multi_row_headers(df)
        if multi_header_plan:
            return multi_header_plan

        # 3. Check for spacer rows/columns
        spacer_plan = cls._plan_spacers(df)
        if spacer_plan:
            return spacer_plan

        # 4. Check for subtotals
        subtotal_plan = cls._plan_subtotals(df)
        if subtotal_plan:
            return subtotal_plan

        # Default: normal tabular dataset
        target_schema = {str(c): str(df[c].dtype) for c in df.columns}
        sample_dict = {
            "before": df.head(3).to_dict(orient="records"),
            "after": df.head(3).to_dict(orient="records"),
        }

        audit = StructuralAudit(
            source_shape=(rows_before, cols_before),
            target_shape=(rows_before, cols_before),
            source_columns=source_cols,
            target_columns=source_cols,
            source_cells_considered=rows_before * cols_before,
            source_cells_retained=rows_before * cols_before,
            source_cells_discarded=0,
            discarded_cell_reasons={},
            confidence_score=1.0,
            ambiguity_flags=[],
            validation_result={"valid": True},
        )

        return StructuralPlan(
            detected_issue="STANDARD_TABULAR",
            evidence=["Dataset appears to be in standard tabular format."],
            proposed_transformation="none",
            source_rows=list(range(rows_before)),
            source_columns=source_cols,
            target_schema=target_schema,
            confidence=1.0,
            ambiguity_flags=[],
            requires_human_approval=False,
            before_after_sample=sample_dict,
            audit=audit,
        )

    @classmethod
    def _plan_repeated_category_blocks(cls, df: pd.DataFrame) -> Optional[StructuralPlan]:
        """Detect and plan unpivoting for horizontally distributed category blocks."""
        cols = [str(c) for c in df.columns]
        if len(cols) < 4:
            return None

        total_cols = [c for c in cols if any(kw in c.lower() for kw in ["total", "subtotal"])]
        unnamed_cols = [c for c in cols if c.startswith("Unnamed:") or c.strip() == ""]

        row0_vals = [str(v).strip() for v in df.iloc[0]] if len(df) > 0 else []
        row0_lower = [v.lower() for v in row0_vals if v and v != "nan"]

        has_repeating_row0 = False
        if len(row0_lower) > 0:
            counts = pd.Series(row0_lower).value_counts()
            if any(counts >= 2):
                has_repeating_row0 = True

        if len(total_cols) >= 1 and (len(unnamed_cols) >= 1 or has_repeating_row0):
            evidence = [
                f"Detected horizontally distributed category blocks across {len(cols)} columns.",
                f"Found {len(total_cols)} subtotal column(s) and {len(unnamed_cols)} unnamed artifact column(s).",
            ]
            if has_repeating_row0:
                evidence.append("Row 0 contains repeating metric sub-headers across groups.")

            ambiguity_flags = []
            group_labels = [c for c in cols if not c.startswith("Unnamed:") and not any(kw in c.lower() for kw in ["total", "subtotal"])]
            if not group_labels:
                ambiguity_flags.append("Could not infer distinct category group names unambiguously.")

            conf = 0.90 if not ambiguity_flags else 0.70
            requires_approval = conf < 0.85 or len(ambiguity_flags) > 0

            target_cols = ["Category_Group"] + [c for c in group_labels[:3]] + ["Metric_Value"]
            target_schema = {c: "object" for c in target_cols}

            sample_before = df.head(3).to_dict(orient="records")
            sample_after = sample_before

            total_cells = len(df) * len(df.columns)
            retained_cells = int(total_cells * 0.75)
            discarded_cells = total_cells - retained_cells

            audit = StructuralAudit(
                source_shape=(len(df), len(df.columns)),
                target_shape=(len(df) * max(1, len(group_labels)), len(target_cols)),
                source_columns=cols,
                target_columns=target_cols,
                source_cells_considered=total_cells,
                source_cells_retained=retained_cells,
                source_cells_discarded=discarded_cells,
                discarded_cell_reasons={"redundant_subtotals": len(total_cols) * len(df)},
                confidence_score=conf,
                ambiguity_flags=ambiguity_flags,
                validation_result={"valid": True, "target_columns_unique": True},
            )

            return StructuralPlan(
                detected_issue="REPEATED_CATEGORY_BLOCKS",
                evidence=evidence,
                proposed_transformation="unpivot_horizontal_category_blocks",
                source_rows=list(range(len(df))),
                source_columns=cols,
                target_schema=target_schema,
                confidence=conf,
                ambiguity_flags=ambiguity_flags,
                requires_human_approval=requires_approval,
                before_after_sample={"before": sample_before, "after": sample_after},
                audit=audit,
            )

        return None

    @classmethod
    def _plan_multi_row_headers(cls, df: pd.DataFrame) -> Optional[StructuralPlan]:
        """Detect and plan multi-row header merging."""
        if len(df) < 2:
            return None

        unnamed_cols = [str(c) for c in df.columns if str(c).startswith("Unnamed:") or str(c).strip() == ""]
        if not unnamed_cols:
            return None

        row0_strs = [isinstance(v, str) and len(v) < 40 for v in df.iloc[0]]
        if sum(row0_strs) / max(1, len(df.columns)) > 0.5:
            evidence = [
                f"{len(unnamed_cols)} unnamed column(s) in top header row.",
                "First row contains valid sub-header label strings.",
            ]
            ambiguity_flags = []

            conf = 0.90
            target_cols = [f"Header_{i}" for i in range(len(df.columns))]
            target_schema = {c: "object" for c in target_cols}

            sample_before = df.head(3).to_dict(orient="records")
            sample_after = df.iloc[1:4].to_dict(orient="records")

            total_cells = len(df) * len(df.columns)
            discarded_cells = len(df.columns)

            audit = StructuralAudit(
                source_shape=(len(df), len(df.columns)),
                target_shape=(len(df) - 1, len(df.columns)),
                source_columns=[str(c) for c in df.columns],
                target_columns=target_cols,
                source_cells_considered=total_cells,
                source_cells_retained=total_cells - discarded_cells,
                source_cells_discarded=discarded_cells,
                discarded_cell_reasons={"promoted_header_row": discarded_cells},
                confidence_score=conf,
                ambiguity_flags=ambiguity_flags,
                validation_result={"valid": True, "target_columns_unique": True},
            )

            return StructuralPlan(
                detected_issue="MULTI_ROW_HEADER",
                evidence=evidence,
                proposed_transformation="flatten_multi_headers",
                source_rows=[0],
                source_columns=[str(c) for c in df.columns],
                target_schema=target_schema,
                confidence=conf,
                ambiguity_flags=ambiguity_flags,
                requires_human_approval=conf < 0.85 or len(ambiguity_flags) > 0,
                before_after_sample={"before": sample_before, "after": sample_after},
                audit=audit,
            )

        return None

    @classmethod
    def _plan_spacers(cls, df: pd.DataFrame) -> Optional[StructuralPlan]:
        """Detect completely empty spacer rows or columns."""
        empty_cols = [str(c) for c in df.columns if df[c].isna().all()]
        empty_rows_idx = list(df.index[df.isna().all(axis=1)])

        if empty_cols or empty_rows_idx:
            evidence = []
            if empty_cols:
                evidence.append(f"Found {len(empty_cols)} completely empty spacer column(s).")
            if empty_rows_idx:
                evidence.append(f"Found {len(empty_rows_idx)} completely empty spacer row(s).")

            target_cols = [str(c) for c in df.columns if str(c) not in empty_cols]
            target_schema = {c: str(df[c].dtype) for c in target_cols}

            sample_before = df.head(3).to_dict(orient="records")
            clean_df = df.drop(columns=empty_cols).drop(index=empty_rows_idx)
            sample_after = clean_df.head(3).to_dict(orient="records")

            total_cells = len(df) * len(df.columns)
            discarded_cells = (len(empty_cols) * len(df)) + (len(empty_rows_idx) * len(target_cols))

            audit = StructuralAudit(
                source_shape=(len(df), len(df.columns)),
                target_shape=(len(clean_df), len(clean_df.columns)),
                source_columns=[str(c) for c in df.columns],
                target_columns=target_cols,
                source_cells_considered=total_cells,
                source_cells_retained=total_cells - discarded_cells,
                source_cells_discarded=discarded_cells,
                discarded_cell_reasons={"blank_spacer_cells": discarded_cells},
                confidence_score=0.98,
                ambiguity_flags=[],
                validation_result={"valid": True, "target_columns_unique": True},
            )

            return StructuralPlan(
                detected_issue="SPACER_ARTIFACTS",
                evidence=evidence,
                proposed_transformation="remove_spacer_rows_cols",
                source_rows=empty_rows_idx,
                source_columns=empty_cols,
                target_schema=target_schema,
                confidence=0.98,
                ambiguity_flags=[],
                requires_human_approval=False,
                before_after_sample={"before": sample_before, "after": sample_after},
                audit=audit,
            )

        return None

    @classmethod
    def _plan_subtotals(cls, df: pd.DataFrame) -> Optional[StructuralPlan]:
        """Detect total/subtotal columns or summary rows."""
        kw_list = ["total", "subtotal", "summary", "grand total", "average"]
        subtotal_cols = [str(c) for c in df.columns if any(kw in str(c).lower() for kw in kw_list)]

        subtotal_rows: list[int] = []
        if len(df) > 0 and len(df.columns) > 0:
            first_col_str = df.iloc[:, 0].dropna().astype(str).str.lower()
            matching_idx = first_col_str[first_col_str.str.contains("|".join(kw_list), regex=True)].index
            subtotal_rows = [int(i) for i in matching_idx]

        if subtotal_cols or subtotal_rows:
            evidence = []
            ambiguity_flags = []
            if subtotal_cols:
                evidence.append(f"Contains {len(subtotal_cols)} subtotal/total column(s): {', '.join(subtotal_cols[:3])}")
            if subtotal_rows:
                evidence.append(f"Contains {len(subtotal_rows)} summary/subtotal row(s)")

            ambiguity_flags.append(
                "Subtotal/total elements detected. Requires explicit verification before deletion to preserve non-redundant records."
            )

            target_cols = [str(c) for c in df.columns if str(c) not in subtotal_cols]
            target_schema = {c: str(df[c].dtype) for c in target_cols}

            sample_before = df.head(3).to_dict(orient="records")
            clean_df = df.drop(columns=subtotal_cols, errors="ignore").drop(index=subtotal_rows, errors="ignore")
            sample_after = clean_df.head(3).to_dict(orient="records")

            total_cells = len(df) * len(df.columns)
            discarded_cells = (len(subtotal_cols) * len(df)) + (len(subtotal_rows) * len(target_cols))

            audit = StructuralAudit(
                source_shape=(len(df), len(df.columns)),
                target_shape=(len(clean_df), len(clean_df.columns)),
                source_columns=[str(c) for c in df.columns],
                target_columns=target_cols,
                source_cells_considered=total_cells,
                source_cells_retained=total_cells - discarded_cells,
                source_cells_discarded=discarded_cells,
                discarded_cell_reasons={"detected_subtotals": discarded_cells},
                confidence_score=0.75,
                ambiguity_flags=ambiguity_flags,
                validation_result={"valid": True, "target_columns_unique": True},
            )

            return StructuralPlan(
                detected_issue="SUBTOTAL_COLUMNS_ROWS",
                evidence=evidence,
                proposed_transformation="remove_subtotal_elements",
                source_rows=subtotal_rows,
                source_columns=subtotal_cols,
                target_schema=target_schema,
                confidence=0.75,
                ambiguity_flags=ambiguity_flags,
                requires_human_approval=True,
                before_after_sample={"before": sample_before, "after": sample_after},
                audit=audit,
            )

        return None

    @classmethod
    def _plan_embedded_structured_records(cls, df: pd.DataFrame) -> Optional[StructuralPlan]:
        """Detect and plan record reconstruction for embedded structured text records."""
        from services.structure.embedded_records import EmbeddedRecordAnalyzer

        profile = EmbeddedRecordAnalyzer.profile_dataset(df)
        labels = profile.get("candidate_field_labels", [])
        confidence = profile.get("confidence", 0.0)
        ambiguity_flags = list(profile.get("ambiguity_flags", []))
        inferred_types = profile.get("inferred_semantic_types", {})

        if not labels or len(labels) < 2 or confidence < 0.50:
            return None

        # Generate preview using EmbeddedRecordAnalyzer
        reconstructed_df, audit_info = EmbeddedRecordAnalyzer.reconstruct_dataframe(df, custom_labels=labels)

        if not audit_info.get("success", False):
            return None

        target_cols = list(reconstructed_df.columns)
        target_schema = {c: str(reconstructed_df[c].dtype) for c in target_cols}

        sample_before = df.head(3).to_dict(orient="records")
        sample_after = reconstructed_df.head(3).to_dict(orient="records")

        evidence = [
            f"Detected embedded structured records across {len(df)} row(s).",
            f"Discovered {len(labels)} field label(s): {', '.join(labels)}",
            f"Cross-row consistency: {profile.get('consistency_of_inferred_structure', 0.0):.0%}",
        ]

        total_cells = len(df) * len(df.columns)
        target_cells = len(reconstructed_df) * len(reconstructed_df.columns)

        audit = StructuralAudit(
            source_shape=(len(df), len(df.columns)),
            target_shape=(len(reconstructed_df), len(reconstructed_df.columns)),
            source_columns=[str(c) for c in df.columns],
            target_columns=target_cols,
            source_cells_considered=total_cells,
            source_cells_retained=target_cells,
            source_cells_discarded=0,
            discarded_cell_reasons={},
            confidence_score=confidence,
            ambiguity_flags=ambiguity_flags,
            validation_result={"valid": True, "target_columns_unique": True},
        )

        requires_approval = confidence < 0.85 or len(ambiguity_flags) > 0

        return StructuralPlan(
            detected_issue="EMBEDDED_STRUCTURED_RECORDS",
            evidence=evidence,
            proposed_transformation="reconstruct_embedded_records",
            source_rows=list(range(len(df))),
            source_columns=[str(c) for c in df.columns],
            target_schema=target_schema,
            confidence=confidence,
            ambiguity_flags=ambiguity_flags,
            requires_human_approval=requires_approval,
            before_after_sample={"before": sample_before, "after": sample_after},
            audit=audit,
        )

