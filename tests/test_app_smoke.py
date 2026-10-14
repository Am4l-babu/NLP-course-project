"""Headless smoke tests: every page renders without an exception (Streamlit AppTest)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

TIMEOUT = 120


@pytest.fixture(autouse=True)
def _no_saved_model(tmp_path, monkeypatch):
    """Make the app fall back to the in-memory demo model, independent of models/."""
    import config
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)


def test_dashboard_renders():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert len(at.metric) == 6
    assert any("Dashboard" in t.value for t in at.title)


def test_analyze_page_single_entry_flow():
    at = AppTest.from_file(str(ROOT / "pages/1_Analyze_Feedback.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    at.text_area[0].set_value("The classroom projector is not working.")
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("Latest result" in s.value for s in at.subheader)


def test_analyze_page_empty_input_is_handled():
    at = AppTest.from_file(str(ROOT / "pages/1_Analyze_Feedback.py"), default_timeout=TIMEOUT).run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert len(at.warning) >= 1


def test_bulk_page_renders_empty_state():
    at = AppTest.from_file(str(ROOT / "pages/2_Bulk_Classification.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert any("Waiting for a CSV" in i.value for i in at.info)


def test_training_page_trains_on_demo_data():
    at = AppTest.from_file(str(ROOT / "pages/3_Model_Training.py"), default_timeout=300).run()
    assert not at.exception, at.exception
    train_buttons = [b for b in at.button if b.label == "Train models"]
    assert train_buttons
    train_buttons[0].click().run()
    assert not at.exception, at.exception
    assert any("evaluation succeeded" in s.value for s in at.success)
    assert len(at.tabs) >= 2


def test_training_page_requires_passcode_when_configured(monkeypatch):
    monkeypatch.setenv("SFC_ADMIN_PASSCODE", "s3cret")
    at = AppTest.from_file(str(ROOT / "pages/3_Model_Training.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert not [b for b in at.button if b.label == "Train models"]  # gated server-side
    at.text_input[0].set_value("wrong")
    at.button[0].click().run()
    assert any("Incorrect" in e.value for e in at.error)
    assert not [b for b in at.button if b.label == "Train models"]
    at.text_input[0].set_value("s3cret")
    at.button[0].click().run()
    assert [b for b in at.button if b.label == "Train models"]


def test_explorer_renders():
    at = AppTest.from_file(str(ROOT / "pages/4_Feedback_Explorer.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, at.exception
    assert any("Common words" in s.value for s in at.subheader)


# ---- upload flows -------------------------------------------------------------------- #
def _bulk_app():
    return AppTest.from_file(str(ROOT / "pages/2_Bulk_Classification.py"), default_timeout=300).run()


def test_bulk_upload_classifies_and_reports_skipped_rows():
    at = _bulk_app()
    csv = (ROOT / "samples" / "sample_input.csv").read_bytes()
    at.file_uploader[0].upload("sample_input.csv", csv, "text/csv").run()
    assert not at.exception, at.exception
    assert any("skipped" in w.value for w in at.warning)  # blank rows are announced, not hidden
    [b for b in at.button if b.label == "Classify feedback"][0].click().run()
    assert not at.exception, at.exception
    assert any("Classified 22 of 24" in s.value for s in at.success)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Uploaded records"] == "24" and metrics["Successfully classified"] == "22"
    assert metrics["Invalid / skipped"] == "2"


def test_bulk_upload_header_only_csv_is_rejected_as_empty_batch():
    at = _bulk_app()
    at.file_uploader[0].upload("empty.csv", b"feedback_text\n", "text/csv").run()
    assert not at.exception, at.exception
    assert any("empty batch" in e.value for e in at.error)


def test_training_page_shows_validation_errors_and_blocks_training():
    at = AppTest.from_file(str(ROOT / "pages/3_Model_Training.py"), default_timeout=TIMEOUT).run()
    at.radio[0].set_value("Upload my labelled CSV").run()
    bad = b"feedback_text,sentiment\nhello there,Happy\n"
    at.file_uploader[0].upload("bad.csv", bad, "text/csv").run()
    assert not at.exception, at.exception
    errors = " ".join(e.value for e in at.error)
    assert "Missing required column" in errors and "Training is disabled" in errors
    assert not [b for b in at.button if b.label == "Train models"]
