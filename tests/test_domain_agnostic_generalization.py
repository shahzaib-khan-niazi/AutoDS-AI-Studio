"""Domain-Agnostic Generalization & Data Integrity Test Suite.

Proves:
1. Complete independence from column names (renaming columns from age/account_balance/customer_name to banana/x17/field_z yields identical results).
2. Unseen domain schema generalization (Meteorology, Logistics, Analytics, Manufacturing, Synthetic).
3. Data preservation over cleaning (zero row/column loss, pd.NA/Unavailable representation).
4. Export blocking on unexplained data loss.
"""

import pandas as pd
import pytest

from services.repair.auto_dtypes import auto_detect_dtypes, _is_identifier_column, _is_boolean_column
from services.repair.outliers import detect_impossible_values, handle_outliers
from services.repair.dates import analyze_date_column, normalize_date_column
from services.repair.missing import fill_missing
from services.repair.export import ExportService
from services.structure.embedded_records import EmbeddedRecordAnalyzer
from services.structure.detector import StructureDetector
from services.repair.structural import reconstruct_embedded_records
from core.exceptions import RepairValidationError


# 1. Numeric & Impossible Value Detection Column-Rename Invariance
def test_numeric_impossible_value_column_rename_invariance():
    # Underlying numeric values with an empirical negative anomaly (-1) in a 90%+ positive series
    vals = [25.0, 34.0, 18.0, 52.0, 40.0, -1.0, 29.0, 61.0, 33.0, 48.0]

    df_original = pd.DataFrame({"age": vals})
    df_renamed_banana = pd.DataFrame({"banana": vals})
    df_renamed_x17 = pd.DataFrame({"x17": vals})

    issues_orig = detect_impossible_values(df_original)
    issues_banana = detect_impossible_values(df_renamed_banana)
    issues_x17 = detect_impossible_values(df_renamed_x17)

    assert "age" in issues_orig
    assert "banana" in issues_banana
    assert "x17" in issues_x17

    assert issues_orig["age"][0]["count"] == 1
    assert issues_banana["banana"][0]["count"] == 1
    assert issues_x17["x17"][0]["count"] == 1


# 2. Date Detection Column-Rename Invariance
def test_date_detection_column_rename_invariance():
    date_strs = ["2026-01-15", "2026-02-20", "2026-03-25", "2026-04-10"]

    df_created_at = pd.DataFrame({"created_at": date_strs})
    df_registered = pd.DataFrame({"registered": date_strs})
    df_banana = pd.DataFrame({"banana": date_strs})

    diag_created = analyze_date_column(df_created_at["created_at"])
    diag_registered = analyze_date_column(df_registered["registered"])
    diag_banana = analyze_date_column(df_banana["banana"])

    assert diag_created.parsed_count == 4
    assert diag_registered.parsed_count == 4
    assert diag_banana.parsed_count == 4

    assert diag_created.inferred_convention == diag_banana.inferred_convention


# 3. Identifier Detection Column-Rename Invariance
def test_identifier_detection_column_rename_invariance():
    zero_padded = pd.Series(["00123", "00456", "00789", "00101"], name="id")
    zero_padded_banana = pd.Series(["00123", "00456", "00789", "00101"], name="banana")

    alpha_codes = pd.Series(["INV-101", "INV-102", "INV-103", "INV-104"], name="code")
    alpha_codes_x17 = pd.Series(["INV-101", "INV-102", "INV-103", "INV-104"], name="x17")

    assert _is_identifier_column(zero_padded) is True
    assert _is_identifier_column(zero_padded_banana) is True

    assert _is_identifier_column(alpha_codes) is True
    assert _is_identifier_column(alpha_codes_x17) is True


# 4. Embedded Record Reconstruction Column-Rename Invariance
def test_embedded_record_column_rename_invariance():
    raw_records = [
        "Name Hussein Hakeem Address Number 22 Fioye Crescent Surulere Lagos Age 17 Gender Male",
        "Name Mary Smith Address 123 Main Street Springfield Age 45 Gender Female",
        "Name John Doe Address 456 Park Avenue New York Age 30 Gender Male",
    ]

    df_original = pd.DataFrame({"Unstructured_Record": raw_records})
    df_renamed = pd.DataFrame({"field_x": raw_records})

    rec_orig, _ = reconstruct_embedded_records(df_original)
    rec_renamed, _ = reconstruct_embedded_records(df_renamed)

    assert set(rec_orig.columns) == set(rec_renamed.columns)
    assert len(rec_orig) == len(rec_renamed) == 3


# 5. Unseen Domain Schema A: Meteorology
def test_unseen_domain_meteorology():
    df = pd.DataFrame({
        "air_pressure": [1013.25, 1011.50, 1009.80, 1014.10],
        "station_code": ["STN-01", "STN-02", "STN-03", "STN-04"],
        "reading": [22.4, 25.1, 19.8, 23.5],
        "recorded_on": ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"],
    })
    res_df, record = auto_detect_dtypes(df)
    assert res_df["air_pressure"].dtype in ("float64", "Float64")
    assert res_df["reading"].dtype in ("float64", "Float64")
    assert pd.api.types.is_datetime64_any_dtype(res_df["recorded_on"])


# 6. Unseen Domain Schema B: Logistics
def test_unseen_domain_logistics():
    df = pd.DataFrame({
        "monthly_cost": ["$1,500.00", "$2,300.50", "$850.75"],
        "region_code": ["REG-NORTH", "REG-SOUTH", "REG-EAST"],
        "units": ["150", "200", "85"],
        "measurement": [4.5, 8.2, 3.1],
    })
    res_df, _ = auto_detect_dtypes(df)
    assert res_df["monthly_cost"].dtype in ("float64", "Float64")
    assert res_df["units"].dtype in ("int64", "Int64")


# 7. Unseen Domain Schema C: Analytics & Scoring
def test_unseen_domain_analytics():
    df = pd.DataFrame({
        "score_value": [88.5, 92.0, 76.4, 95.1],
        "reference_key": ["REF-9001", "REF-9002", "REF-9003", "REF-9004"],
        "event_timestamp": ["2026-08-10 14:30:00", "2026-08-10 15:45:00", "2026-08-10 16:15:00", "2026-08-10 17:00:00"],
        "category_label": ["A", "A", "B", "A"],
    })
    res_df, _ = auto_detect_dtypes(df)
    assert res_df["score_value"].dtype in ("float64", "Float64")
    assert pd.api.types.is_datetime64_any_dtype(res_df["event_timestamp"])


# 8. Unseen Domain Schema D: Manufacturing & Quality
def test_unseen_domain_manufacturing():
    df = pd.DataFrame({
        "weight_measure": [14.2, 14.5, 14.1, 14.3],
        "serial_code": ["001045", "001046", "001047", "001048"],
        "observation": ["PASS", "PASS", "FAIL", "PASS"],
        "status_flag": ["true", "true", "false", "true"],
    })
    res_df, _ = auto_detect_dtypes(df)
    assert res_df["serial_code"].dtype == "object"  # Preserved as string ID
    assert str(res_df["status_flag"].dtype) == "boolean"


# 9. Synthetic Randomized Column Names
def test_synthetic_randomized_column_names():
    df = pd.DataFrame({
        "feature_a": [10, 20, 30, 40],
        "x17": [1.1, 2.2, 3.3, 4.4],
        "metric_03": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        "field_z": ["CA-01", "CA-02", "CA-03", "CA-04"],
        "value_9": ["yes", "no", "yes", "no"],
    })
    res_df, _ = auto_detect_dtypes(df)
    assert res_df["feature_a"].dtype in ("int64", "Int64")
    assert res_df["x17"].dtype in ("float64", "Float64")
    assert pd.api.types.is_datetime64_any_dtype(res_df["metric_03"])
    assert str(res_df["value_9"].dtype) == "boolean"


# 10. Data Preservation: Zero Silent Row Loss
def test_zero_silent_row_loss():
    # 500 rows with some missing cells
    df = pd.DataFrame({
        "feature_x": [i for i in range(500)],
        "value_y": [float(i) if i % 10 != 0 else None for i in range(500)],
    })

    repaired_df, record = fill_missing(df, strategy="auto")
    assert len(repaired_df) == 500  # Exactly 500 rows preserved


# 11. Data Preservation: Unrepairable Cells Become pd.NA (Unavailable)
def test_unrepairable_cells_become_unavailable():
    df = pd.DataFrame({
        "id_code": ["ID-01", "ID-02", None, "ID-04"],
        "score": [10.0, None, None, 40.0],
    })

    # High missingness column or identifier column -> Auto imputation leaves as pd.NA
    repaired_df, _ = fill_missing(df, strategy="auto")
    assert len(repaired_df) == 4
    assert pd.isna(repaired_df.loc[2, "id_code"])
    assert pd.isna(repaired_df.loc[1, "score"])


# 12. Export Integrity Blocking on Unexplained Data Loss
def test_unexplained_data_loss_blocks_export():
    original_df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
    empty_df = pd.DataFrame()

    with pytest.raises(RepairValidationError) as excinfo:
        ExportService.export_csv(empty_df, original_df=original_df)

    assert "Export blocked" in str(excinfo.value)
