"""Prompts for AI reasoning and planning in AutoDS AI Studio.

Prompts are designed to produce strictly structured JSON responses.
"""

STRUCTURE_REASONING_PROMPT = """
You are an expert Data Scientist and Table Structure Analyst.
Analyze the following compact dataset profile to determine its structural orientation.

Dataset Profile:
{profile_json}

Available Structure Types:
- NORMAL_TABLE: Standard relational table where each row is an entity and each column is an independent variable.
- WIDE_TABLE: Category labels or time periods are spread across columns rather than stacked vertically.
- LONG_TABLE: Tidy/narrow format with explicit key/variable and value columns.
- PIVOT_TABLE: 2D matrix layout with row categories on the left and column categories across headers, containing aggregated values.
- CROSSTAB: Contingency table containing frequency counts.
- MULTI_HEADER: First row(s) contain metadata, titles, or hierarchical header labels.
- TIME_SERIES: Chronological sequence of data indexed by timestamp or date.
- SURVEY: Questionnaire/survey results with many categorical or Likert-scale questions.
- TRANSACTIONAL: Event logs or business transactions with IDs, timestamps, and amounts.
- UNKNOWN: Ambiguous or unidentifiable structure.

Respond ONLY with valid JSON matching this schema:
{{
    "decision": "<STRUCTURE_TYPE>",
    "confidence": <float between 0.0 and 1.0>,
    "reasoning_summary": "<concise 1-3 sentence summary of reasoning>",
    "actions": [
        {{
            "operation": "<explode_multi_value_cells | unpivot | flatten_headers | remove_repeated_headers | remove_metadata_rows | remove_duplicate_rows | rename_columns | drop_empty_rows | drop_empty_columns | convert_types>",
            "target": ["<column_name_1>", ...],
            "parameters": {{}},
            "reason": "<why this action is recommended>"
        }}
    ],
    "warnings": ["<warning_1>", ...]
}}
"""

REPAIR_PLANNING_PROMPT = """
You are an expert Data Cleaning and Preparation Specialist.
Analyze the following dataset profile and quality assessment to create a safe, prioritized repair plan.

Dataset Profile:
{profile_json}

Allowed Operations:
1. "explode_multi_value_cells": Explode synchronized or single multi-value cells into separate relational rows. Parameters: {{"delimiter": "|", "fill_mismatched": true}}
2. "remove_repeated_headers": Drop duplicate header rows embedded inside the data body. Parameters: {{}}
3. "remove_metadata_rows": Drop leading/trailing summary rows and report title banners. Parameters: {{}}
4. "remove_duplicate_rows": Drop identical duplicate rows. Parameters: {{}}
5. "rename_columns": Fix unnamed or duplicate column names. Parameters: {{}}
6. "drop_empty_rows": Remove rows that are entirely null. Parameters: {{}}
7. "drop_empty_columns": Remove columns that are entirely null. Parameters: {{}}
8. "convert_types": Convert columns to target dtype. Parameters: {{"target_type": "numeric" | "datetime"}}, target: list of column names
9. "unpivot": Convert wide table to long format. Parameters: {{"id_columns": [...], "var_name": "...", "value_name": "..."}}
10. "flatten_headers": Promote metadata/header rows. Parameters: {{"header_rows": <int>}}
11. "strip_whitespace": Strip leading/trailing and extra internal whitespace from string columns. Parameters: {{}}
12. "standardize_values": Normalize inconsistent categorical values (case, abbreviations, spelling variants). Parameters: {{}}
13. "auto_dtypes": Auto-detect and convert mistyped object columns to numeric/datetime/boolean. Parameters: {{}}
14. "normalize_dates": Normalize date/timestamp columns to unified datetime format. Parameters: {{"convention": "DMY" | "MDY" | "ISO" | null}}
15. "handle_outliers": Cap extreme statistical outliers in numeric columns. Parameters: {{"method": "iqr" | "zscore" | "mad", "factor": 1.5, "action": "cap"}}
16. "fill_missing": Impute missing cells. Parameters: {{"strategy": "auto" | "median" | "mode" | "mean" | "ffill" | "bfill"}}

IMPORTANT RULES:
- Never recommend operations that cause unsafe data loss.
- Only use allowed operations.
- target columns MUST match exact column names from the profile.
- Return ONLY valid JSON matching this schema:

{{
    "decision": "<brief description of plan>",
    "confidence": <float between 0.0 and 1.0>,
    "reasoning_summary": "<concise rationale for the plan>",
    "actions": [
        {{
            "operation": "<operation_name>",
            "target": ["<column_name>"],
            "parameters": {{}},
            "reason": "<reason>"
        }}
    ],
    "warnings": ["<warning>"]
}}
"""

