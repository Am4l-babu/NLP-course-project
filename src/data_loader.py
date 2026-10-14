"""Reading and writing CSV data safely."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

import config
from src.validation import validate_training_df, validate_upload_file


class DataLoadError(Exception):
    """Raised with a user-readable message when a file cannot be read."""


def read_csv_bytes(data: bytes, filename: str = "upload.csv") -> pd.DataFrame:
    """Parse uploaded CSV bytes into an all-string DataFrame.

    * checks extension and size first,
    * decodes UTF-8 (with or without BOM), falling back to Windows-1252,
    * keeps every cell as the original text (``dtype=str``, no NA conversion) so
      the export can reproduce the input faithfully.
    """
    problems = validate_upload_file(filename, len(data) if data is not None else 0)
    if problems:
        raise DataLoadError(" ".join(problems))

    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DataLoadError("The file is not valid UTF-8 or Windows-1252 text. Save it as CSV (UTF-8).")

    try:
        df = pd.read_csv(
            io.StringIO(text),
            dtype=str,
            keep_default_na=False,
            nrows=config.MAX_UPLOAD_ROWS + 1,
        )
    except pd.errors.EmptyDataError:
        raise DataLoadError("The CSV file is empty.")
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"The file could not be parsed as CSV: {exc}")
    except Exception as exc:  # defensive: surface a readable message, never a stack trace
        raise DataLoadError(f"The file could not be read: {exc}")

    if len(df) > config.MAX_UPLOAD_ROWS:
        raise DataLoadError(f"Too many rows; the limit is {config.MAX_UPLOAD_ROWS:,}.")
    return df


def load_demo_data() -> pd.DataFrame:
    """Load and validate the built-in synthetic demo dataset."""
    path = Path(config.DEMO_DATA_PATH)
    if not path.exists():
        raise DataLoadError(
            f"Demo dataset not found at {path}. Run: python scripts/build_demo_dataset.py"
        )
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    result = validate_training_df(raw)
    if not result.ok:
        raise DataLoadError("Demo dataset failed validation: " + " ".join(result.errors))
    return result.df


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def _neutralise_formula(value):
    """Defuse CSV/Excel formula injection: a text cell starting with = @ or a
    non-numeric +/- gets a leading apostrophe so spreadsheets show it as text."""
    if not isinstance(value, str) or not value:
        return value
    first = value[0]
    if first in ("=", "@"):
        return "'" + value
    if first in ("+", "-"):
        try:
            float(value.replace(",", ""))
            return value
        except ValueError:
            return "'" + value
    return value


def to_csv_bytes(df: pd.DataFrame, neutralise_formulas: bool = True) -> bytes:
    """Serialise ``df`` to UTF-8 CSV (with BOM so Excel detects the encoding).

    Row order and columns are exactly those of ``df``. With
    ``neutralise_formulas`` (default) text cells that a spreadsheet could execute
    as formulas get a leading apostrophe; the README documents this.
    """
    out = df.copy()
    if neutralise_formulas:
        for col in out.columns:
            if out[col].dtype == object or str(out[col].dtype).startswith("str"):
                out[col] = out[col].map(_neutralise_formula)
    buffer = io.StringIO()
    out.to_csv(buffer, index=False)
    return ("﻿" + buffer.getvalue()).encode("utf-8")
