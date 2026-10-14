"""Validation of uploaded files and DataFrames.

Principle: **nothing is silently discarded**. Every problem is either
  * an *error* (blocks training / classification, with a fix-it hint), or
  * a *warning* that states exactly what was done (e.g. "3 exact duplicate rows
    were removed: rows 12, 40, 77").
Row numbers are CSV line numbers (header = line 1, first record = line 2).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import config
from src.preprocessing import STATUS_EMPTY, STATUS_INVALID, STATUS_TRUNCATED, prepare_text


@dataclass
class ValidationResult:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    df: Optional[pd.DataFrame] = None
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rows_str(index_values, limit: int = 8) -> str:
    """Format CSV line numbers, e.g. '5, 9, 14 (+3 more)'."""
    lines = [int(i) + 2 for i in index_values]
    shown = ", ".join(str(n) for n in lines[:limit])
    extra = len(lines) - limit
    return shown + (f" (+{extra} more)" if extra > 0 else "")


def _label_key(label: str) -> str:
    s = str(label).lower().replace("&", " and ")
    return re.sub(r"[^a-z]", "", s)


_SENTIMENT_LOOKUP = {_label_key(s): s for s in config.SENTIMENTS}
_CATEGORY_LOOKUP = {_label_key(c): c for c in config.CATEGORIES}


def canonical_sentiment(value) -> Optional[str]:
    return _SENTIMENT_LOOKUP.get(_label_key(value)) if isinstance(value, str) else None


def canonical_category(value) -> Optional[str]:
    return _CATEGORY_LOOKUP.get(_label_key(value)) if isinstance(value, str) else None


# --------------------------------------------------------------------------- #
# file level
# --------------------------------------------------------------------------- #
def validate_upload_file(filename: str, size_bytes: int) -> list:
    """Return a list of error strings (empty list = acceptable)."""
    errors = []
    base = os.path.basename(filename or "")
    ext = os.path.splitext(base)[1].lower()
    if not base:
        errors.append("The uploaded file has no name.")
    elif ext not in config.ALLOWED_UPLOAD_EXTENSIONS:
        errors.append(
            f"Unsupported file type '{ext or 'none'}'. Please upload a "
            f"{', '.join(config.ALLOWED_UPLOAD_EXTENSIONS)} file."
        )
    if size_bytes is None or size_bytes <= 0:
        errors.append("The uploaded file is empty.")
    elif size_bytes > config.MAX_UPLOAD_MB * 1024 * 1024:
        errors.append(
            f"The file is {size_bytes / 1024 / 1024:.1f} MB; the limit is "
            f"{config.MAX_UPLOAD_MB} MB. Split it into smaller files."
        )
    return errors


# --------------------------------------------------------------------------- #
# labelled training data
# --------------------------------------------------------------------------- #
def validate_training_df(df: pd.DataFrame) -> ValidationResult:
    """Validate (and tidy) labelled training data.

    On success ``result.df`` has columns feedback_id, feedback_text, sentiment,
    category, created_at with canonical labels. Exact duplicate rows are removed
    *and reported*; conflicting duplicates (same text, different labels) are an error.
    """
    errors: list = []
    warnings: list = []
    info: dict = {}

    if df is None or len(df.columns) == 0:
        return ValidationResult(False, ["The file has no columns."], info=info)

    work = df.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]
    if work.columns.duplicated().any():
        dups = sorted(set(work.columns[work.columns.duplicated()]))
        errors.append(f"Duplicate column names after trimming/lower-casing: {dups}.")

    missing = [c for c in config.REQUIRED_TRAINING_COLUMNS if c not in work.columns]
    if missing:
        errors.append(
            f"Missing required column(s): {missing}. The CSV header must contain "
            f"{config.REQUIRED_TRAINING_COLUMNS} (optional: feedback_id, created_at)."
        )
    if len(work) == 0:
        errors.append("The file contains a header but no data rows.")
    if errors:
        return ValidationResult(False, errors, warnings, info=info)

    if len(work) > config.MAX_UPLOAD_ROWS:
        return ValidationResult(
            False,
            [f"Too many rows ({len(work):,}); the limit is {config.MAX_UPLOAD_ROWS:,}."],
        )

    # Text columns as clean strings (NaN -> "").
    for col in config.REQUIRED_TRAINING_COLUMNS:
        work[col] = work[col].fillna("").astype(str).str.strip()
    info["rows_in_file"] = len(work)

    # --- empty / unusable text -------------------------------------------- #
    prepared = work[config.TEXT_COLUMN].map(prepare_text)
    status = prepared.map(lambda p: p.status)
    bad_text = status.isin([STATUS_EMPTY, STATUS_INVALID])
    if bad_text.any():
        errors.append(
            f"{int(bad_text.sum())} row(s) have empty or unusable feedback_text "
            f"(CSV lines {_rows_str(work.index[bad_text])}). Fill in or delete these rows."
        )
    long_text = status == STATUS_TRUNCATED
    if long_text.any():
        warnings.append(
            f"{int(long_text.sum())} feedback text(s) exceed {config.MAX_TEXT_CHARS:,} "
            f"characters and will be truncated for training (CSV lines "
            f"{_rows_str(work.index[long_text])})."
        )

    # --- labels ------------------------------------------------------------ #
    sent = work[config.SENTIMENT_COLUMN].map(canonical_sentiment)
    cat = work[config.CATEGORY_COLUMN].map(canonical_category)
    bad_sent = sent.isna()
    bad_cat = cat.isna()
    if bad_sent.any():
        bad_vals = sorted(set(work.loc[bad_sent, config.SENTIMENT_COLUMN]))[:6]
        errors.append(
            f"{int(bad_sent.sum())} row(s) have an invalid sentiment {bad_vals} "
            f"(CSV lines {_rows_str(work.index[bad_sent])}). "
            f"Allowed values: {config.SENTIMENTS}."
        )
    if bad_cat.any():
        bad_vals = sorted(set(work.loc[bad_cat, config.CATEGORY_COLUMN]))[:6]
        errors.append(
            f"{int(bad_cat.sum())} row(s) have an invalid category {bad_vals} "
            f"(CSV lines {_rows_str(work.index[bad_cat])}). "
            f"Allowed values: {config.CATEGORIES}. Use 'Other' for anything else."
        )
    if errors:
        return ValidationResult(False, errors, warnings, info=info)

    work[config.SENTIMENT_COLUMN] = sent
    work[config.CATEGORY_COLUMN] = cat
    work[config.TEXT_COLUMN] = prepared.map(lambda p: p.text)

    # --- duplicates --------------------------------------------------------- #
    key = work[config.TEXT_COLUMN].str.lower().str.replace(r"\s+", " ", regex=True)
    conflict_groups = (
        work.assign(_k=key)
        .groupby("_k")[[config.SENTIMENT_COLUMN, config.CATEGORY_COLUMN]]
        .nunique()
    )
    conflicting_keys = conflict_groups[
        (conflict_groups[config.SENTIMENT_COLUMN] > 1)
        | (conflict_groups[config.CATEGORY_COLUMN] > 1)
    ].index
    if len(conflicting_keys):
        rows = work.index[key.isin(conflicting_keys)]
        errors.append(
            f"{len(conflicting_keys)} feedback text(s) appear with conflicting labels "
            f"(CSV lines {_rows_str(rows)}). Make the labels consistent."
        )
        return ValidationResult(False, errors, warnings, info=info)

    dup_mask = work.assign(_k=key).duplicated(
        subset=["_k", config.SENTIMENT_COLUMN, config.CATEGORY_COLUMN], keep="first"
    )
    if dup_mask.any():
        warnings.append(
            f"{int(dup_mask.sum())} exact duplicate row(s) were removed before training "
            f"(CSV lines {_rows_str(work.index[dup_mask])}). Duplicates would leak "
            f"between the training and test sets."
        )
        work = work[~dup_mask]
    info["duplicates_removed"] = int(dup_mask.sum())

    # --- optional columns --------------------------------------------------- #
    if config.ID_COLUMN not in work.columns:
        work[config.ID_COLUMN] = [f"ROW-{i + 1:05d}" for i in range(len(work))]
    else:
        work[config.ID_COLUMN] = work[config.ID_COLUMN].fillna("").astype(str)
    if config.DATE_COLUMN in work.columns:
        parsed = pd.to_datetime(work[config.DATE_COLUMN], errors="coerce", format="mixed")
        n_bad = int(parsed.isna().sum() - (work[config.DATE_COLUMN].astype(str).str.strip() == "").sum())
        if n_bad > 0:
            warnings.append(
                f"{n_bad} created_at value(s) could not be parsed as dates and will be "
                f"ignored in time-based charts."
            )
        work[config.DATE_COLUMN] = parsed
    else:
        work[config.DATE_COLUMN] = pd.NaT

    # --- sizes and class counts -------------------------------------------- #
    n = len(work)
    if n < config.MIN_TOTAL_TRAINING_ROWS:
        errors.append(
            f"Only {n} usable row(s); at least {config.MIN_TOTAL_TRAINING_ROWS} are needed."
        )

    s_counts = work[config.SENTIMENT_COLUMN].value_counts()
    for label in config.SENTIMENTS:
        have = int(s_counts.get(label, 0))
        if have < config.MIN_EXAMPLES_PER_CLASS:
            errors.append(
                f"Sentiment '{label}' has {have} example(s); at least "
                f"{config.MIN_EXAMPLES_PER_CLASS} are needed for every sentiment."
            )

    c_counts = work[config.CATEGORY_COLUMN].value_counts()
    if len(c_counts) < 2:
        errors.append("At least two different categories are needed to train a topic classifier.")
    thin = {k: int(v) for k, v in c_counts.items() if v < config.MIN_EXAMPLES_PER_CLASS}
    if thin:
        errors.append(
            f"Categories with fewer than {config.MIN_EXAMPLES_PER_CLASS} examples: {thin}. "
            f"Add more examples or relabel them as 'Other'."
        )

    info["rows_usable"] = n
    info["sentiment_counts"] = {k: int(v) for k, v in s_counts.items()}
    info["category_counts"] = {k: int(v) for k, v in c_counts.items()}

    cols = [config.ID_COLUMN, config.TEXT_COLUMN, config.SENTIMENT_COLUMN,
            config.CATEGORY_COLUMN, config.DATE_COLUMN]
    clean = work[cols].reset_index(drop=True)
    return ValidationResult(not errors, errors, warnings, clean if not errors else None, info)


# --------------------------------------------------------------------------- #
# unlabelled data for batch classification
# --------------------------------------------------------------------------- #
def find_text_column(df: pd.DataFrame, preferred: str = config.TEXT_COLUMN) -> Optional[str]:
    """Exact match first, then case-insensitive match on the preferred name."""
    if preferred in df.columns:
        return preferred
    for col in df.columns:
        if str(col).strip().lower() == preferred:
            return col
    return None


def validate_unlabeled_df(df: pd.DataFrame, text_column: Optional[str] = None) -> ValidationResult:
    """Validate a batch of unlabelled feedback.

    The data frame is *not* modified or filtered: rows with unusable text stay in
    the file (and in the export) and are marked as skipped.
    """
    errors: list = []
    warnings: list = []
    info: dict = {}

    if df is None or len(df.columns) == 0:
        return ValidationResult(False, ["The file has no columns."], info=info)
    if len(df) == 0:
        return ValidationResult(
            False, ["The file contains a header but no data rows (empty batch)."], info=info
        )
    if len(df) > config.MAX_UPLOAD_ROWS:
        return ValidationResult(
            False, [f"Too many rows ({len(df):,}); the limit is {config.MAX_UPLOAD_ROWS:,}."]
        )

    text_column = text_column or find_text_column(df)
    if text_column is None or text_column not in df.columns:
        errors.append(
            f"No '{config.TEXT_COLUMN}' column found. Columns present: {list(df.columns)}. "
            f"Rename the column holding the feedback to '{config.TEXT_COLUMN}' or choose it manually."
        )
        return ValidationResult(False, errors, info=info)

    clash = [c for c in config.PREDICTION_COLUMNS if c in df.columns]
    if clash:
        errors.append(
            f"The file already contains column(s) reserved for predictions: {clash}. "
            f"Rename or remove them so predictions are not confused with your own labels."
        )
        return ValidationResult(False, errors, info=info)

    prepared = df[text_column].map(prepare_text)
    status = prepared.map(lambda p: p.status)
    n_empty = int((status == STATUS_EMPTY).sum())
    n_invalid = int((status == STATUS_INVALID).sum())
    n_trunc = int((status == STATUS_TRUNCATED).sum())
    n_usable = int(prepared.map(lambda p: p.usable).sum())

    if n_usable == 0:
        errors.append("None of the rows contain usable feedback text; nothing to classify.")
    if n_empty + n_invalid:
        bad = status.isin([STATUS_EMPTY, STATUS_INVALID])
        warnings.append(
            f"{n_empty + n_invalid} row(s) have empty or unusable text and will be marked "
            f"'skipped' in the output (CSV lines {_rows_str(df.index[bad])})."
        )
    if n_trunc:
        warnings.append(
            f"{n_trunc} row(s) exceed {config.MAX_TEXT_CHARS:,} characters; only the first "
            f"{config.MAX_TEXT_CHARS:,} characters are classified."
        )
    info.update(
        text_column=text_column,
        rows=len(df),
        usable=n_usable,
        skipped=n_empty + n_invalid,
        truncated=n_trunc,
    )
    return ValidationResult(not errors, errors, warnings, df, info)
