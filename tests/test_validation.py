import pandas as pd
import pytest

import config
from src.data_loader import DataLoadError, read_csv_bytes
from src.validation import (
    canonical_category, canonical_sentiment, validate_training_df,
    validate_unlabeled_df, validate_upload_file,
)
from tests.conftest import make_training_frame


# ---- upload files ------------------------------------------------------------ #
def test_wrong_extension_rejected():
    assert validate_upload_file("data.xlsx", 100)
    assert validate_upload_file("../../etc/passwd", 100)  # no extension -> rejected
    assert not validate_upload_file("good.csv", 100)


def test_empty_and_oversized_files_rejected():
    assert validate_upload_file("a.csv", 0)
    assert validate_upload_file("a.csv", config.MAX_UPLOAD_MB * 1024 * 1024 + 1)


def test_path_components_are_ignored_in_filename_check():
    assert not validate_upload_file("..\\..\\evil\\good.csv", 10)


def test_read_csv_bytes_keeps_cells_verbatim():
    df = read_csv_bytes(b"feedback_text,score\n  hello ,007\n,NA\n", "x.csv")
    assert df.loc[0, "score"] == "007" and df.loc[1, "score"] == "NA"
    assert df.loc[0, "feedback_text"] == "  hello "


def test_read_csv_bytes_handles_bom_and_cp1252():
    assert list(read_csv_bytes("﻿feedback_text\nhi\n".encode("utf-8"), "a.csv").columns) == ["feedback_text"]
    assert read_csv_bytes("feedback_text\ncaf\xe9\n".encode("cp1252"), "a.csv").loc[0, "feedback_text"] == "caf\xe9"


@pytest.mark.parametrize("payload", [b"", b"\n\n"])
def test_read_csv_bytes_rejects_empty(payload):
    with pytest.raises(DataLoadError):
        read_csv_bytes(payload, "a.csv")


def test_read_csv_bytes_rejects_bad_extension():
    with pytest.raises(DataLoadError):
        read_csv_bytes(b"a,b\n1,2\n", "a.exe")


# ---- labelled training data ------------------------------------------------------ #
def test_valid_training_data_passes():
    res = validate_training_df(make_training_frame(n_per_cell=5))
    assert res.ok, res.errors
    assert list(res.df.columns) == ["feedback_id", "feedback_text", "sentiment", "category", "created_at"]


def test_demo_dataset_is_valid_and_balanced(demo_df):
    assert len(demo_df) == 300
    assert demo_df["sentiment"].value_counts().nunique() == 1
    assert demo_df["feedback_id"].str.startswith("DEMO-").all()


@pytest.mark.parametrize("missing", config.REQUIRED_TRAINING_COLUMNS)
def test_missing_required_column_is_reported(missing):
    df = make_training_frame(5).drop(columns=[missing])
    res = validate_training_df(df)
    assert not res.ok and any(missing in e for e in res.errors)


def test_header_only_file_rejected():
    res = validate_training_df(make_training_frame(5).iloc[0:0])
    assert not res.ok and any("no data rows" in e for e in res.errors)


def test_invalid_sentiment_label_reported_with_row_numbers():
    df = make_training_frame(5)
    df.loc[3, "sentiment"] = "Happy"
    res = validate_training_df(df)
    assert not res.ok
    msg = " ".join(res.errors)
    assert "Happy" in msg and "5" in msg  # index 3 -> CSV line 5


def test_invalid_category_label_reported():
    df = make_training_frame(5)
    df.loc[0, "category"] = "Parking"
    res = validate_training_df(df)
    assert not res.ok and any("Parking" in e for e in res.errors)


def test_labels_are_case_and_spacing_insensitive():
    assert canonical_sentiment(" positive ") == "Positive"
    assert canonical_category("internet & wi-fi") == "Internet and Wi-Fi"
    assert canonical_category("INTERNET AND WIFI") == "Internet and Wi-Fi"
    assert canonical_category("Parking") is None and canonical_sentiment(None) is None


def test_empty_feedback_text_is_an_error_not_silently_dropped():
    df = make_training_frame(5)
    df.loc[2, "feedback_text"] = "   "
    res = validate_training_df(df)
    assert not res.ok and any("empty" in e for e in res.errors)


def test_insufficient_examples_in_a_sentiment_class():
    df = make_training_frame(5)
    df = df[~((df.sentiment == "Neutral") & (df.index % 2 == 0))]  # leave too few neutrals
    df = pd.concat([df[df.sentiment != "Neutral"], df[df.sentiment == "Neutral"].head(2)])
    res = validate_training_df(df)
    assert not res.ok and any("Neutral" in e for e in res.errors)


def test_missing_sentiment_class_entirely():
    df = make_training_frame(5)
    res = validate_training_df(df[df.sentiment != "Negative"])
    assert not res.ok and any("Negative" in e and "0 example" in e for e in res.errors)


def test_category_with_too_few_examples():
    df = make_training_frame(5)
    extra = pd.DataFrame({"feedback_text": ["only one transport message"], "sentiment": ["Positive"], "category": ["Transport"]})
    res = validate_training_df(pd.concat([df, extra], ignore_index=True))
    assert not res.ok and any("Transport" in e for e in res.errors)


def test_exact_duplicates_are_removed_and_reported():
    df = make_training_frame(5)
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    res = validate_training_df(df)
    assert res.ok
    assert len(res.df) == len(df) - 2
    assert any("duplicate" in w for w in res.warnings)
    assert res.info["duplicates_removed"] == 2


def test_conflicting_duplicates_are_an_error():
    df = make_training_frame(5)
    clash = df.iloc[[0]].copy()
    clash["sentiment"] = "Negative" if df.loc[0, "sentiment"] != "Negative" else "Positive"
    res = validate_training_df(pd.concat([df, clash], ignore_index=True))
    assert not res.ok and any("conflicting" in e for e in res.errors)


def test_too_few_rows_rejected():
    res = validate_training_df(make_training_frame(1))
    assert not res.ok


def test_invalid_dates_warn_but_do_not_block():
    df = make_training_frame(5)
    df["created_at"] = "2026-01-05"
    df.loc[0, "created_at"] = "not a date"
    res = validate_training_df(df)
    assert res.ok and any("created_at" in w for w in res.warnings)


# ---- unlabelled batches --------------------------------------------------------------- #
def test_empty_batch_rejected():
    res = validate_unlabeled_df(pd.DataFrame(columns=["feedback_text"]))
    assert not res.ok and any("empty batch" in e for e in res.errors)


def test_batch_without_text_column_rejected():
    res = validate_unlabeled_df(pd.DataFrame({"comment": ["hi"]}))
    assert not res.ok and any("feedback_text" in e for e in res.errors)


def test_batch_text_column_lookup_is_case_insensitive():
    res = validate_unlabeled_df(pd.DataFrame({"Feedback_Text": ["hello there"]}))
    assert res.ok and res.info["text_column"] == "Feedback_Text"


def test_batch_with_reserved_prediction_columns_rejected():
    res = validate_unlabeled_df(pd.DataFrame({"feedback_text": ["hi"], "predicted_sentiment": ["Positive"]}))
    assert not res.ok and any("reserved" in e for e in res.errors)


def test_batch_with_only_blank_rows_rejected():
    res = validate_unlabeled_df(pd.DataFrame({"feedback_text": ["", "  ", "!!!"]}))
    assert not res.ok


def test_batch_with_some_blank_rows_warns_and_keeps_all_rows():
    df = pd.DataFrame({"feedback_text": ["good", "", "bad"]})
    res = validate_unlabeled_df(df)
    assert res.ok and res.info["skipped"] == 1 and len(res.df) == 3
    assert any("skipped" in w for w in res.warnings)
