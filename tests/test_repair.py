"""Tests for dataset repair operations and validator."""

import numpy as np
import pandas as pd
import pytest

from models.repair import RepairAction, RepairOperation
from services.repair.service import RepairService
from services.repair.duplicates import remove_duplicate_rows
from services.repair.columns import rename_columns
from services.repair.missing import drop_empty_rows, drop_empty_columns
from services.repair.structural import unpivot, flatten_headers
from services.repair.validator import validate_repair


def test_remove_duplicate_rows() -> None:
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    repaired, record = remove_duplicate_rows(df)

    assert len(repaired) == 3
    assert record.rows_before == 4
    assert record.rows_after == 3
    assert record.details["duplicates_removed"] == 1
    # Original should be unmodified
    assert len(df) == 4


def test_rename_columns_deterministic() -> None:
    # Test unnamed, duplicates, whitespace
    df = pd.DataFrame(
        [[1, 2, 3, 4]],
        columns=["Unnamed: 0", "Revenue", "Revenue", "  TrimMe  "],
    )
    repaired, record = rename_columns(df)

    expected_cols = ["unnamed_0", "Revenue", "Revenue_1", "TrimMe"]
    assert list(repaired.columns) == expected_cols
    assert record.success is True


def test_drop_empty_rows_and_cols() -> None:
    df = pd.DataFrame({
        "a": [1, np.nan, 3],
        "b": [np.nan, np.nan, np.nan],
        "c": [4, np.nan, 6],
    })

    # Drop empty rows
    r_rows, _ = drop_empty_rows(df)
    assert len(r_rows) == 2  # Middle row dropped

    # Drop empty columns
    r_cols, _ = drop_empty_columns(df)
    assert list(r_cols.columns) == ["a", "c"]


def test_unpivot_wide_to_long() -> None:
    df = pd.DataFrame({
        "id": [1, 2],
        "Q1": [100, 200],
        "Q2": [110, 210],
    })

    melted, record = unpivot(df, id_columns=["id"], value_columns=["Q1", "Q2"])
    assert len(melted) == 4
    assert set(melted.columns) == {"id", "variable", "value"}


def test_flatten_headers() -> None:
    df = pd.DataFrame([
        ["Header_A", "Header_B"],
        ["Val_1", "Val_2"],
        ["Val_3", "Val_4"],
    ])

    flattened, record = flatten_headers(df, header_rows=1)
    assert list(flattened.columns) == ["Header_A", "Header_B"]
    assert len(flattened) == 2


def test_repair_validator_catastrophic_loss() -> None:
    df_orig = pd.DataFrame({"a": range(100)})
    df_broken = pd.DataFrame({"a": range(10)})  # 90% row loss

    val = validate_repair(df_orig, df_broken, "custom_op")
    assert val.valid is False
    assert len(val.errors) > 0


def test_full_repair_service_pipeline() -> None:
    df = pd.DataFrame({
        "Unnamed: 0": [1, 1, 2],
        "Score": [10, 10, 20],
        "Empty": [np.nan, np.nan, np.nan],
    })

    actions = [
        RepairAction(operation=RepairOperation.REMOVE_DUPLICATE_ROWS),
        RepairAction(operation=RepairOperation.RENAME_COLUMNS),
        RepairAction(operation=RepairOperation.DROP_EMPTY_COLUMNS),
    ]

    repaired, result = RepairService.repair(df, actions)
    assert result.success is True
    assert len(repaired) == 2
    assert "Empty" not in repaired.columns
    assert "unnamed_0" in repaired.columns
