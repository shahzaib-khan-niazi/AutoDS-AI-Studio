"""Upload service for loading datasets from files.

Handles CSV, Excel, and Parquet files.
Validates files before loading. Archives originals to data/raw/.
DOES NOT clean or modify the data during loading.
"""

import os
from typing import Any, BinaryIO, Optional
import pandas as pd

from core.exceptions import DatasetLoadError, DatasetValidationError
from core.logging import logger
from models.dataset import DatasetMetadata
from utils.files import (
    get_file_extension,
    is_supported_extension,
    generate_timestamped_filename,
    ensure_directory,
)


class UploadService:
    """Service for uploading and loading dataset files."""

    RAW_DATA_DIR = os.path.join("data", "raw")

    @classmethod
    def load(cls, file: BinaryIO, filename: str) -> tuple[pd.DataFrame, DatasetMetadata]:
        """Load a dataset from an uploaded file.

        Args:
            file: File-like object from Streamlit uploader.
            filename: Original filename.

        Returns:
            Tuple of (DataFrame, DatasetMetadata).

        Raises:
            DatasetLoadError: If the file cannot be loaded.
            DatasetValidationError: If the loaded data is invalid.
        """
        logger.info("Loading file: {}", filename)

        # Validate file extension
        if not is_supported_extension(filename):
            ext = get_file_extension(filename)
            raise DatasetLoadError(
                message=f"Unsupported file type: '{ext}'",
                details=f"Supported types: .csv, .xlsx, .xls, .parquet",
            )

        # Load into DataFrame based on extension
        ext = get_file_extension(filename)
        try:
            df = cls._read_file(file, ext)
        except DatasetLoadError:
            raise
        except Exception as e:
            raise DatasetLoadError(
                message=f"Failed to read file: {filename}",
                details=str(e),
            )

        # Validate the loaded DataFrame
        cls._validate_dataframe(df, filename)

        # Create metadata
        metadata = cls._create_metadata(df, filename, file=file)

        # Archive the original file
        cls._archive_file(file, filename)

        logger.info(
            "Dataset loaded: {} ({} rows × {} cols, {})",
            filename,
            metadata.rows,
            metadata.columns,
            f"{metadata.memory_usage_mb:.2f} MB",
        )

        return df, metadata

    @staticmethod
    def _read_file(file: BinaryIO, ext: str) -> pd.DataFrame:
        """Read a file into a DataFrame based on extension.

        Args:
            file: File-like object.
            ext: Lowercase file extension.

        Returns:
            Loaded DataFrame.

        Raises:
            DatasetLoadError: If reading fails.
        """
        try:
            if ext == ".csv":
                return pd.read_csv(file)
            elif ext in (".xlsx", ".xls"):
                return pd.read_excel(file, engine="openpyxl")
            elif ext == ".parquet":
                return pd.read_parquet(file)
            else:
                raise DatasetLoadError(
                    message=f"Unsupported extension: {ext}",
                )
        except DatasetLoadError:
            raise
        except pd.errors.EmptyDataError:
            raise DatasetLoadError(
                message="The file is empty or contains no parseable data.",
            )
        except Exception as e:
            raise DatasetLoadError(
                message=f"Error reading {ext} file",
                details=str(e),
            )

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, filename: str) -> None:
        """Validate that the loaded DataFrame is usable.

        Args:
            df: Loaded DataFrame.
            filename: Original filename for error messages.

        Raises:
            DatasetValidationError: If the DataFrame is invalid.
        """
        if not isinstance(df, pd.DataFrame):
            raise DatasetValidationError(
                message="The file did not produce a valid dataset.",
            )

        if df.empty:
            raise DatasetValidationError(
                message="The dataset is empty (zero rows).",
                details=f"File: {filename}",
            )

        if len(df.columns) == 0:
            raise DatasetValidationError(
                message="The dataset has no columns.",
                details=f"File: {filename}",
            )

    @classmethod
    def _extract_excel_metadata(cls, file: BinaryIO) -> tuple[list[str], dict[str, Any]]:
        """Extract merged cell ranges and sheet metadata using openpyxl if available."""
        merged_cells: list[str] = []
        sheet_meta: dict[str, Any] = {}
        try:
            file.seek(0)
            import openpyxl
            wb = openpyxl.load_workbook(file, data_only=True, read_only=False)
            sheet = wb.active
            if sheet is not None:
                sheet_meta["title"] = sheet.title
                sheet_meta["max_row"] = sheet.max_row
                sheet_meta["max_column"] = sheet.max_column
                if hasattr(sheet, "merged_cells") and sheet.merged_cells:
                    merged_cells = [str(r) for r in sheet.merged_cells.ranges]
            file.seek(0)
        except Exception as e:
            logger.debug("Merged cell extraction skipped: {}", str(e))
            try:
                file.seek(0)
            except Exception:
                pass
        return merged_cells, sheet_meta

    @classmethod
    def _create_metadata(cls, df: pd.DataFrame, filename: str, file: Optional[BinaryIO] = None) -> DatasetMetadata:
        """Create metadata from a loaded DataFrame.

        Args:
            df: Loaded DataFrame.
            filename: Original filename.
            file: Optional file handle.

        Returns:
            DatasetMetadata instance.
        """
        memory_bytes = int(df.memory_usage(deep=True).sum())
        dtypes_dict: dict[str, str] = {}
        for col in df.columns:
            dtypes_dict[str(col)] = str(df[col].dtype)

        merged_cells: list[str] = []
        sheet_meta: dict[str, Any] = {}
        ext = get_file_extension(filename)
        if file is not None and ext in (".xlsx", ".xls"):
            merged_cells, sheet_meta = cls._extract_excel_metadata(file)

        return DatasetMetadata(
            filename=filename,
            rows=len(df),
            columns=len(df.columns),
            memory_usage_bytes=memory_bytes,
            column_names=[str(c) for c in df.columns],
            dtypes=dtypes_dict,
            merged_cells=merged_cells,
            sheet_metadata=sheet_meta,
        )

    @classmethod
    def _archive_file(cls, file: BinaryIO, filename: str) -> None:
        """Archive the uploaded file to data/raw/ with a timestamp.

        Args:
            file: File-like object (seeked back to start).
            filename: Original filename.
        """
        try:
            ensure_directory(cls.RAW_DATA_DIR)
            archived_name = generate_timestamped_filename(filename)
            archive_path = os.path.join(cls.RAW_DATA_DIR, archived_name)

            file.seek(0)
            content = file.read()
            with open(archive_path, "wb") as f:
                f.write(content)
            file.seek(0)

            logger.debug("File archived to: {}", archive_path)
        except Exception as e:
            # Archive failure is non-fatal — log and continue
            logger.warning("Failed to archive file: {}", str(e))
