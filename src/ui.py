"""Streamlit helpers shared by every page (kept out of the core modules so those
stay importable and testable without Streamlit)."""
from __future__ import annotations

import hmac
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import config
from src.analytics import standardize_results
from src.data_loader import DataLoadError, load_demo_data
from src.predict import FeedbackClassifier, ModelLoadError
from src.train import TrainingError, train_models

DEMO_LABEL = "Demo dataset (synthetic, predicted by the active model)"
BULK_LABEL = "Latest bulk classification (this session)"
SINGLE_LABEL = "Entries analysed this session"

_CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1200px;}
.sfc-badge {display:inline-block; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.85rem; color:#fff;}
.sfc-note {border-left:4px solid #457B9D; padding:0.5rem 0.9rem; background:rgba(69,123,157,0.08); border-radius:4px; font-size:0.9rem;}
.sfc-warn {border-left:4px solid #E9A23B; padding:0.5rem 0.9rem; background:rgba(233,162,59,0.10); border-radius:4px; font-size:0.9rem;}
div[data-testid="stMetric"] {background:rgba(128,128,128,0.08); padding:0.7rem 0.9rem; border-radius:8px;}
</style>
"""


def setup_page(title: str, subtitle: str = "") -> None:
    """Inject shared styling and render the page header + privacy note."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    with st.sidebar:
        st.markdown("**Privacy**")
        st.caption(
            "Runs locally. Feedback text is processed in memory and is not logged or "
            "stored unless you explicitly save a model or download a file. No names, "
            "e-mail addresses or phone numbers are needed - please do not enter them."
        )


def badge(label: str, kind: str = "sentiment") -> str:
    colors = config.SENTIMENT_COLORS if kind == "sentiment" else config.PRIORITY_COLORS
    color = colors.get(label, "#6C757D")
    return f'<span class="sfc-badge" style="background:{color}">{label}</span>'


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="{"sfc-warn" if warn else "sfc-note"}">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Model access
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training a demo model on the built-in dataset (first run only)...")
def _demo_classifier() -> FeedbackClassifier:
    result = train_models(load_demo_data(), dataset_name="built-in synthetic demo dataset")
    return FeedbackClassifier.from_training_result(result)


def get_classifier() -> FeedbackClassifier:
    """Active classifier for this browser session: trained this session > saved
    on disk > in-memory demo model (so the app always works out of the box)."""
    if "classifier" in st.session_state:
        return st.session_state["classifier"]
    source, problem = "saved", None
    try:
        clf = FeedbackClassifier.load()
    except ModelLoadError as exc:
        metadata_exists = (config.MODELS_DIR / config.MODEL_METADATA_FILE).exists()
        problem = str(exc) if metadata_exists else None
        try:
            clf = _demo_classifier()
        except (TrainingError, DataLoadError) as exc2:
            st.error(f"No usable model and the demo model could not be trained: {exc2}")
            st.stop()
        source = "demo (in memory)"
    set_classifier(clf, source, problem)
    return clf


def set_classifier(clf: FeedbackClassifier, source: str, problem: str = None) -> None:
    st.session_state["classifier"] = clf
    st.session_state["classifier_source"] = source
    st.session_state["classifier_problem"] = problem


def model_banner() -> FeedbackClassifier:
    """Load the model and show a one-line status; returns the classifier."""
    clf = get_classifier()
    if st.session_state.get("classifier_problem"):
        note(f"A saved model could not be used and was ignored: {st.session_state['classifier_problem']} "
             "Using an in-memory demo model instead.", warn=True)
    meta = clf.metadata
    ds = meta.get("dataset", {}).get("name", "unknown data")
    sent = meta.get("tasks", {}).get("sentiment", {}).get("selected", "?")
    cat = meta.get("tasks", {}).get("category", {}).get("selected", "?")
    st.caption(
        f"Active model: **{st.session_state.get('classifier_source')}** - trained on {ds} - "
        f"sentiment: {sent}; topic: {cat}"
    )
    if "demo" in str(ds).lower() or "demo" in st.session_state.get("classifier_source", ""):
        st.caption(
            "The demo data is a small **synthetic** set for demonstration only. Reliable "
            "real-world performance needs a larger, representative, accurately labelled dataset."
        )
    return clf


# --------------------------------------------------------------------------- #
# Admin gate for training
# --------------------------------------------------------------------------- #
def admin_gate() -> bool:
    """True when the user may train models.

    If ``SFC_ADMIN_PASSCODE`` is set, a passcode is required and compared on the
    server in constant time (this is a real check, not UI hiding). If it is not
    set the page runs in open local mode and says so.
    """
    expected = os.environ.get(config.ADMIN_PASSCODE_ENV, "")
    if not expected:
        note("Local course-project mode: <b>no administrator passcode is configured</b>, so anyone who can "
             f"open this app can retrain models. Set the <code>{config.ADMIN_PASSCODE_ENV}</code> environment "
             "variable to require one. A production deployment needs proper authentication, access "
             "control and secure storage.", warn=True)
        return True
    if st.session_state.get("admin_ok"):
        return True
    with st.form("admin_login"):
        code = st.text_input("Administrator passcode", type="password")
        submitted = st.form_submit_button("Unlock training")
    if submitted:
        if hmac.compare_digest(code.encode(), expected.encode()):
            st.session_state["admin_ok"] = True
            st.rerun()
        st.error("Incorrect passcode.")
    return False


# --------------------------------------------------------------------------- #
# Session datasets (what the Dashboard / Explorer analyse)
# --------------------------------------------------------------------------- #
def register_bulk_results(df: pd.DataFrame, text_column: str) -> None:
    st.session_state["bulk_results"] = standardize_results(df, text_column)


def add_single_result(prediction) -> None:
    rows = st.session_state.setdefault("single_results", [])
    rows.append(
        {
            "feedback_text": prediction.original_text,
            "predicted_sentiment": prediction.sentiment,
            "predicted_category": prediction.category,
            "predicted_priority": prediction.priority,
            "sentiment_confidence": prediction.sentiment_score,
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        }
    )


def _demo_predictions(clf: FeedbackClassifier) -> pd.DataFrame:
    key = f"demo_pred::{clf.model_id}"
    if key not in st.session_state:
        raw = load_demo_data().copy()
        raw["created_at"] = raw["created_at"].dt.strftime("%Y-%m-%d")
        out, _ = clf.predict_batch(raw, config.TEXT_COLUMN)
        st.session_state[key] = standardize_results(out, config.TEXT_COLUMN)
    return st.session_state[key]


def dataset_picker(clf: FeedbackClassifier, key: str):
    """Sidebar-free radio to choose which classified data to analyse.
    Returns ``(label, DataFrame or None)``."""
    options = [DEMO_LABEL]
    if st.session_state.get("bulk_results") is not None and len(st.session_state["bulk_results"]):
        options.insert(0, BULK_LABEL)
    if st.session_state.get("single_results"):
        options.append(SINGLE_LABEL)
    choice = st.radio("Data to analyse", options, key=key, horizontal=True)
    if choice == BULK_LABEL:
        return choice, st.session_state["bulk_results"]
    if choice == SINGLE_LABEL:
        return choice, standardize_results(pd.DataFrame(st.session_state["single_results"]))
    try:
        return choice, _demo_predictions(clf)
    except DataLoadError as exc:
        st.error(str(exc))
        return choice, None


def source_caveat(label: str) -> None:
    if label == DEMO_LABEL:
        note("These are <b>model predictions on the synthetic demo data</b>, which the model was also "
             "trained on, so they look better than they would on genuinely new feedback. "
             "Upload your own file on the <i>Bulk Classification</i> page to analyse real data.", warn=True)
    else:
        note("All sentiments, topics and priorities shown are <b>model predictions</b>, not ground-truth labels.")
