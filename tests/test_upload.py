"""Tests for upload service."""

import io
import pandas as pd
import pytest

from core.exceptions import DatasetLoadError, DatasetValidationError
from services.upload.service import UploadService


def test_upload_csv_success() -> None:
    csv_content = b"col_a,col_b,col_c\n1,foo,10.5\n2,bar,20.5\n3,baz,30.5\n"
    file_obj = io.BytesIO(csv_content)

    df, metadata = UploadService.load(file_obj, "sample.csv")

    assert len(df) == 3
    assert len(df.columns) == 3
    assert metadata.filename == "sample.csv"
    assert metadata.rows == 3
    assert metadata.columns == 3


def test_upload_parquet_success() -> None:
    df_orig = pd.DataFrame({"x": [10, 20], "y": ["alpha", "beta"]})
    file_obj = io.BytesIO()
    df_orig.to_parquet(file_obj)
    file_obj.seek(0)

    df, metadata = UploadService.load(file_obj, "sample.parquet")
    assert len(df) == 2
    assert metadata.columns == 2


def test_upload_unsupported_format() -> None:
    file_obj = io.BytesIO(b"some content")
    with pytest.raises(DatasetLoadError):
        UploadService.load(file_obj, "sample.txt")


def test_upload_empty_csv() -> None:
    file_obj = io.BytesIO(b"")
    with pytest.raises(DatasetLoadError):
        UploadService.load(file_obj, "empty.csv")


def test_upload_zero_rows_with_header() -> None:
    file_obj = io.BytesIO(b"col_a,col_b\n")
    with pytest.raises(DatasetValidationError):
        UploadService.load(file_obj, "zero_rows.csv")
