"""Unit test suite for granular row and column removal feature."""

import pandas as pd
import pytest
from models.repair import RepairAction, RepairOperation
from core.exceptions import RepairValidationError
from services.repair.columns import (
    remove_specific_columns,
    remove_specific_rows,
    remove_rows_and_columns,
)
from services.repair.dispatcher import dispatch


def test_remove_specific_columns_single():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    repaired, record = remove_specific_columns(df, columns=["B"])

    assert "B" not in repaired.columns
    assert list(repaired.columns) == ["A", "C"]
    assert len(df.columns) == 3  # Original not mutated
    assert record.columns_before == 3
    assert record.columns_after == 2
    assert record.details["removed_column_names"] == ["B"]
    assert record.details["original_shape"] == (3, 3)
    assert record.details["resulting_shape"] == (3, 2)
    assert record.details["user_approved"] is True


def test_remove_specific_columns_multiple():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]})
    repaired, record = remove_specific_columns(df, columns=["A", "C"])

    assert list(repaired.columns) == ["B", "D"]
    assert record.details["removed_column_names"] == ["A", "C"]


def test_remove_specific_columns_invalid():
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(RepairValidationError, match="Columns not found"):
        remove_specific_columns(df, columns=["NON_EXISTENT"])

    with pytest.raises(RepairValidationError, match="No columns specified"):
        remove_specific_columns(df, columns=[])


def test_remove_specific_rows_by_index():
    df = pd.DataFrame({"A": [10, 20, 30, 40, 50], "B": ["a", "b", "c", "d", "e"]})
    repaired, record = remove_specific_rows(df, row_indices=[1, 3])

    assert len(repaired) == 3
    assert repaired["A"].tolist() == [10, 30, 50]
    assert len(df) == 5  # Original not mutated
    assert record.rows_before == 5
    assert record.rows_after == 3
    assert record.details["removed_row_indices"] == [1, 3]
    assert record.details["original_shape"] == (5, 2)
    assert record.details["resulting_shape"] == (3, 2)
    assert record.details["user_approved"] is True


def test_remove_specific_rows_invalid():
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(RepairValidationError, match="out of range"):
        remove_specific_rows(df, row_indices=[10])

    with pytest.raises(RepairValidationError, match="No row indices specified"):
        remove_specific_rows(df, row_indices=[])


def test_remove_rows_and_columns_combined():
    df = pd.DataFrame({
        "Col1": [1, 2, 3, 4],
        "Col2": [10, 20, 30, 40],
        "Col3": [100, 200, 300, 400],
    })

    repaired, record = remove_rows_and_columns(
        df,
        row_indices=[0, 2],
        columns=["Col2"],
        condition_desc="Rows 0, 2 and Col2",
        reason="Manual user test",
    )

    assert len(repaired) == 2
    assert list(repaired.columns) == ["Col1", "Col3"]
    assert repaired["Col1"].tolist() == [2, 4]
    assert record.details["removed_row_indices"] == [0, 2]
    assert record.details["removed_column_names"] == ["Col2"]
    assert record.details["original_shape"] == (4, 3)
    assert record.details["resulting_shape"] == (2, 2)


def test_dispatcher_remove_rows_and_columns():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    action = RepairAction(
        operation=RepairOperation.REMOVE_ROWS_AND_COLUMNS,
        target=["C"],
        parameters={"row_indices": [1]},
        reason="Test dispatch removal",
    )

    repaired, record = dispatch(df, action)

    assert list(repaired.columns) == ["A", "B"]
    assert len(repaired) == 2
    assert repaired["A"].tolist() == [1, 3]
    assert record.details["removed_column_names"] == ["C"]
    assert record.details["removed_row_indices"] == [1]
