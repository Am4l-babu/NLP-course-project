import io
import json

import numpy as np
import pandas as pd
import pytest

import config
from src.data_loader import to_csv_bytes
from src.predict import FeedbackClassifier, ModelLoadError, PredictionError
from src.train import TrainingError, train_models
from tests.conftest import make_training_frame


# ---- training ------------------------------------------------------------------------ #
def test_training_produces_evaluated_models(training_result):
    r = training_result
    assert r.status == "success"
    for key in ("sentiment", "category"):
        task = r.tasks[key]
        assert {c.name for c in task.candidates} == {"Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"}
        best = max(c.cv_f1_macro_mean for c in task.candidates)
        assert task.selected.cv_f1_macro_mean == pytest.approx(best)  # selected by CV macro-F1
        m = task.selected.test_metrics
        assert 0 <= m["f1_macro"] <= 1 and 0 <= m["accuracy"] <= 1
        assert np.array(m["confusion_matrix"]).sum() == m["n_test"]


def test_split_is_stratified_and_leak_free(training_result):
    sp = training_result.split
    assert sp["stratified_by"] == "sentiment+category"
    assert sp["n_train"] + sp["n_test"] == training_result.dataset["n_rows"]


def test_training_is_reproducible(demo_df, training_result):
    again = train_models(demo_df, dataset_name="test demo data")
    for key in ("sentiment", "category"):
        a, b = training_result.tasks[key], again.tasks[key]
        assert a.selected_name == b.selected_name
        assert a.selected.test_metrics["f1_macro"] == pytest.approx(b.selected.test_metrics["f1_macro"])


def test_training_refuses_invalid_data():
    bad = make_training_frame(5).drop(columns=["sentiment"])
    with pytest.raises(TrainingError):
        train_models(bad)


def test_training_refuses_missing_class():
    df = make_training_frame(5)
    with pytest.raises(TrainingError):
        train_models(df[df.sentiment != "Neutral"])


def test_unsuccessful_result_cannot_become_a_classifier(training_result):
    import copy
    broken = copy.copy(training_result)
    broken.status = "failed"
    with pytest.raises(ModelLoadError):
        FeedbackClassifier.from_training_result(broken)


# ---- single prediction ------------------------------------------------------------------ #
@pytest.mark.parametrize("text,sentiment", [
    ("The faculty explained the concepts very clearly.", "Positive"),
    ("The classroom projector is not working.", "Negative"),
    ("The laboratory is open until 5 PM.", "Neutral"),
])
def test_requirement_examples(classifier, text, sentiment):
    assert classifier.predict_one(text).sentiment == sentiment


def test_unseen_feedback_gets_valid_labels_and_scores(classifier):
    p = classifier.predict_one("The hostel corridor lights flicker every night and nobody repairs them.")
    assert p.ok
    assert p.sentiment in config.SENTIMENTS and p.category in config.CATEGORIES and p.priority in config.PRIORITIES
    assert set(p.sentiment_scores) == set(config.SENTIMENTS)
    assert sum(p.sentiment_scores.values()) == pytest.approx(1.0)
    assert 0 < p.sentiment_score <= 1


@pytest.mark.parametrize("bad", ["", "   ", None, "!!!", np.nan])
def test_bad_single_inputs_do_not_raise(classifier, bad):
    p = classifier.predict_one(bad)
    assert not p.ok and p.sentiment is None and p.message


def test_very_long_and_odd_text_is_handled(classifier):
    assert classifier.predict_one("The wifi is slow. " * 1000).status == "truncated"
    assert classifier.predict_one("ഫാക്കൽറ്റി നല്ലതാണ് \U0001F600 \x00").ok


# ---- batch prediction ---------------------------------------------------------------------- #
def _batch():
    return pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "feedback_text": ["The canteen food is cold and tasteless.", "", "The library opens at 8 AM.", "Excellent lab instructors!"],
        "extra": ["x", "y", "z", "w"],
    })


def test_batch_preserves_input_columns_and_order(classifier):
    df = _batch()
    out, summary = classifier.predict_batch(df)
    assert list(out.columns[: len(df.columns)]) == list(df.columns)
    assert out[["id", "feedback_text", "extra"]].equals(df)
    assert list(out.columns[len(df.columns):]) == config.PREDICTION_COLUMNS
    assert summary["uploaded"] == 4 and summary["classified"] == 3 and summary["skipped"] == 1


def test_batch_marks_skipped_rows_and_does_not_drop_them(classifier):
    out, _ = classifier.predict_batch(_batch())
    assert len(out) == 4
    assert out.loc[1, "predicted_sentiment"] == "" and out.loc[1, "classification_status"].startswith("skipped")
    assert out.loc[0, "classification_status"] == "ok"
    assert out.loc[0, "classification_timestamp"].endswith("+00:00")


def test_batch_does_not_mutate_input(classifier):
    df = _batch()
    before = df.copy(deep=True)
    classifier.predict_batch(df)
    assert df.equals(before)


def test_empty_batch_raises_readable_error(classifier):
    with pytest.raises(PredictionError, match="empty batch"):
        classifier.predict_batch(pd.DataFrame(columns=["feedback_text"]))


def test_batch_missing_text_column_raises(classifier):
    with pytest.raises(PredictionError):
        classifier.predict_batch(pd.DataFrame({"comment": ["x"]}))


def test_batch_matches_single_predictions(classifier):
    df = _batch()
    out, _ = classifier.predict_batch(df)
    single = classifier.predict_one(df.loc[0, "feedback_text"])
    assert out.loc[0, "predicted_sentiment"] == single.sentiment
    assert out.loc[0, "predicted_category"] == single.category


# ---- CSV export ------------------------------------------------------------------------------ #
def test_csv_export_roundtrip_preserves_records(classifier):
    out, _ = classifier.predict_batch(_batch())
    data = to_csv_bytes(out)
    back = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(back.columns) == list(out.columns)
    assert back["id"].tolist() == ["a", "b", "c", "d"]
    assert back["feedback_text"].tolist() == out["feedback_text"].tolist()
    assert back["predicted_sentiment"].tolist() == out["predicted_sentiment"].tolist()


def test_csv_export_quotes_commas_and_newlines():
    df = pd.DataFrame({"feedback_text": ['says "hi", then\nleaves']})
    back = pd.read_csv(io.BytesIO(to_csv_bytes(df)), dtype=str, encoding="utf-8-sig")
    assert back.loc[0, "feedback_text"] == 'says "hi", then\nleaves'


def test_csv_export_neutralises_spreadsheet_formulas():
    df = pd.DataFrame({"feedback_text": ["=HYPERLINK(\"http://x\")", "@cmd", "+1+1", "-5", "normal text"]})
    back = pd.read_csv(io.BytesIO(to_csv_bytes(df)), dtype=str, encoding="utf-8-sig")["feedback_text"].tolist()
    assert back[0].startswith("'=") and back[1].startswith("'@") and back[2].startswith("'+")
    assert back[3] == "-5" and back[4] == "normal text"


# ---- persistence ------------------------------------------------------------------------------ #
def test_save_and_load_roundtrip(classifier, tmp_path):
    classifier.save(tmp_path)
    loaded = FeedbackClassifier.load(tmp_path)
    text = "The bus was late again today."
    a, b = classifier.predict_one(text), loaded.predict_one(text)
    assert (a.sentiment, a.category) == (b.sentiment, b.category)
    assert a.sentiment_score == pytest.approx(b.sentiment_score)


def test_loading_from_missing_folder_gives_readable_error(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        FeedbackClassifier.load(tmp_path / "nope")


def test_missing_single_model_file(classifier, tmp_path):
    classifier.save(tmp_path)
    (tmp_path / config.CATEGORY_MODEL_FILE).unlink()
    with pytest.raises(ModelLoadError, match="not found"):
        FeedbackClassifier.load(tmp_path)


def test_corrupted_model_file_is_detected_before_unpickling(classifier, tmp_path):
    classifier.save(tmp_path)
    path = tmp_path / config.SENTIMENT_MODEL_FILE
    path.write_bytes(path.read_bytes()[:200] + b"garbage")
    with pytest.raises(ModelLoadError, match="checksum"):
        FeedbackClassifier.load(tmp_path)


def test_swapped_model_file_is_rejected(classifier, tmp_path):
    import pickle
    classifier.save(tmp_path)
    (tmp_path / config.SENTIMENT_MODEL_FILE).write_bytes(pickle.dumps({"not": "a model"}))
    with pytest.raises(ModelLoadError):
        FeedbackClassifier.load(tmp_path)


def test_unreadable_metadata_is_rejected(classifier, tmp_path):
    classifier.save(tmp_path)
    (tmp_path / config.MODEL_METADATA_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelLoadError, match="metadata"):
        FeedbackClassifier.load(tmp_path)


def test_matching_checksum_but_wrong_object_type_is_rejected(classifier, tmp_path):
    """Even if metadata is consistent, a non-Pipeline object must not be accepted."""
    import hashlib, joblib
    classifier.save(tmp_path)
    joblib.dump({"not": "a pipeline"}, tmp_path / config.SENTIMENT_MODEL_FILE)
    meta_path = tmp_path / config.MODEL_METADATA_FILE
    meta = json.loads(meta_path.read_text())
    meta["files"]["sentiment"]["sha256"] = hashlib.sha256((tmp_path / config.SENTIMENT_MODEL_FILE).read_bytes()).hexdigest()
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ModelLoadError, match="Pipeline"):
        FeedbackClassifier.load(tmp_path)


def test_metadata_contains_no_feedback_text(classifier, tmp_path):
    classifier.save(tmp_path)
    blob = (tmp_path / config.MODEL_METADATA_FILE).read_text()
    assert "projector" not in blob and "faculty explained" not in blob
