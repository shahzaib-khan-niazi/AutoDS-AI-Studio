"""Dataset Export Service for AutoDS AI Studio.

Ensures export cleanliness:
- Only repaired data is exported
- Unique, sanitized column names (no Unnamed: artifacts)
- No artificial index column (e.g. Unnamed: 0)
- CSV export uses index=False
- Excel export uses index=False
- Internal datetime columns preserved as datetime64[ns]; formatted to DD-MM-YY on export
"""

import io
from typing import BinaryIO, Optional
import pandas as pd


from core.exceptions import RepairValidationError


class ExportService:
    """Service for exporting cleaned datasets cleanly and safely."""

    @classmethod
    def validate_export_integrity(cls, df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None) -> None:
        """Verify dataset integrity prior to export. Blocks export on unexplained data loss.

        Args:
            df: Cleaned DataFrame to export.
            original_df: Optional original DataFrame before repair.

        Raises:
            RepairValidationError: If critical data loss or corruption is detected.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise RepairValidationError("Export blocked: Invalid DataFrame object.")

        if df.empty and (original_df is None or not original_df.empty):
            raise RepairValidationError("Export blocked: Dataset is empty or catastrophic row loss occurred.")

        if original_df is not None and len(original_df) > 0 and len(df) == 0:
            raise RepairValidationError(
                f"Export blocked due to unexplained data loss: Original dataset contained {len(original_df)} rows, but exported dataset is empty."
            )

    @classmethod
    def clean_df_for_export(cls, df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Sanitize column names and strip Unnamed artifacts prior to export.

        Args:
            df: Source DataFrame (never modified).
            original_df: Optional original DataFrame before repair.

        Returns:
            Sanitized DataFrame copy ready for export.
        """
        cls.validate_export_integrity(df, original_df)
        export_df = df.copy()

        # Remove artificial index columns if present
        index_cols = [c for c in export_df.columns if str(c).strip().lower() in ("unnamed: 0", "index")]
        if index_cols and len(export_df.columns) > len(index_cols):
            export_df = export_df.drop(columns=index_cols)

        # Sanitize column names (strip whitespace, fill generic Unnamed: names)
        new_cols: list[str] = []
        seen: dict[str, int] = {}
        for idx, col in enumerate(export_df.columns):
            name = str(col).strip()
            if name.startswith("Unnamed:") or name == "" or name == "nan":
                name = f"col_{idx + 1}"

            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0

            new_cols.append(name)

        export_df.columns = pd.Index(new_cols)
        return export_df

    @classmethod
    def export_csv(
        cls,
        df: pd.DataFrame,
        original_df: Optional[pd.DataFrame] = None,
        date_format: str = "%d-%m-%y",
    ) -> bytes:
        """Export dataset to CSV bytes with index=False.

        Args:
            df: Source DataFrame.
            original_df: Optional original DataFrame for validation.
            date_format: Date format for datetime columns (default DD-MM-YY).

        Returns:
            UTF-8 encoded CSV bytes with index=False.
        """
        export_df = cls.clean_df_for_export(df, original_df=original_df)
        buffer = io.StringIO()
        export_df.to_csv(buffer, index=False, date_format=date_format)
        return buffer.getvalue().encode("utf-8")

    @classmethod
    def export_excel(
        cls,
        df: pd.DataFrame,
        original_df: Optional[pd.DataFrame] = None,
        date_format: str = "%d-%m-%y",
    ) -> bytes:
        """Export dataset to Excel (.xlsx) bytes with index=False.

        Args:
            df: Source DataFrame.
            original_df: Optional original DataFrame for validation.
            date_format: Date format for datetime columns.

        Returns:
            Excel file bytes with index=False.
        """
        export_df = cls.clean_df_for_export(df, original_df=original_df)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl", date_format=date_format) as writer:
            export_df.to_excel(writer, index=False, sheet_name="Cleaned Data")
        return buffer.getvalue()
