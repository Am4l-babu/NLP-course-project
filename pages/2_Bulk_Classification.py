"""Page: classify a CSV of unlabelled feedback and download the results."""
import streamlit as st

import config
from src import analytics, ui, visualizations as viz
from src.data_loader import DataLoadError, read_csv_bytes, to_csv_bytes
from src.predict import PredictionError
from src.validation import find_text_column, validate_unlabeled_df

ui.setup_page("Bulk Classification", "Upload a CSV of feedback and classify every row.")
clf = ui.model_banner()

st.markdown(
    f"Upload a **CSV** (max {config.MAX_UPLOAD_MB} MB, {config.MAX_UPLOAD_ROWS:,} rows) with a "
    f"`{config.TEXT_COLUMN}` column. Other columns are kept untouched and in the same order. "
    "The output adds *predicted_* columns - these are **predictions, not ground-truth labels**."
)
with st.expander("Example of the expected format"):
    st.code("feedback_id,feedback_text,created_at\nS-001,The Wi-Fi in the library is very slow,2026-02-03\n"
            "S-002,Lab assistants were very helpful,2026-02-04", language="csv")

file = st.file_uploader("Feedback CSV", type=["csv"], accept_multiple_files=False)
if file is None:
    st.info("Waiting for a CSV file. Nothing is stored: the file is processed in memory only.")
    prev = st.session_state.get("bulk_output")
    if prev is None:
        st.stop()
    st.caption("Showing the previous result from this session.")
else:
    try:
        raw = read_csv_bytes(file.getvalue(), file.name)
    except DataLoadError as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()

    text_col = find_text_column(raw)
    if text_col is None:
        st.warning(f"No `{config.TEXT_COLUMN}` column found. Choose the column that contains the feedback text:")
        text_col = st.selectbox("Text column", list(raw.columns))

    check = validate_unlabeled_df(raw, text_col)
    for e in check.errors:
        st.error(e)
    for w in check.warnings:
        st.warning(w)
    if not check.ok:
        st.stop()

    if st.button("Classify feedback", type="primary"):
        with st.spinner("Classifying..."):
            try:
                out, summary = clf.predict_batch(raw, text_col)
            except PredictionError as exc:
                st.error(f"Classification failed: {exc}")
                st.stop()
        st.session_state["bulk_output"] = (out, summary, file.name)
        ui.register_bulk_results(out, text_col)
        st.success(f"Classified {summary['classified']:,} of {summary['uploaded']:,} records.")
    elif "bulk_output" not in st.session_state:
        st.info("Press **Classify feedback** to run the model.")
        st.stop()

stored = st.session_state.get("bulk_output")
if stored is None:
    st.stop()
out, summary, fname = stored

c1, c2, c3 = st.columns(3)
c1.metric("Uploaded records", f"{summary['uploaded']:,}")
c2.metric("Successfully classified", f"{summary['classified']:,}")
c3.metric("Invalid / skipped", f"{summary['skipped']:,}", help="Rows with empty or unusable text are kept in the output and marked 'skipped'.")
if summary["skipped"]:
    st.warning(f"{summary['skipped']} record(s) were skipped (see `classification_status`). They are still present in the download.")

ok = analytics.standardize_results(out, summary["text_column"])
if not ok.empty:
    s1, s2 = st.columns(2)
    s1.plotly_chart(viz.sentiment_distribution(ok), width="stretch")
    s2.plotly_chart(viz.topic_distribution(ok), width="stretch")

st.subheader("Predictions")
f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
query = f1.text_input("Search feedback text")
sent_f = f2.multiselect("Sentiment", config.SENTIMENTS)
cat_f = f3.multiselect("Category", config.CATEGORIES)
pri_f = f4.multiselect("Priority", config.PRIORITIES)
view = out
if query.strip():
    view = view[view[summary["text_column"]].astype(str).str.contains(query.strip(), case=False, regex=False)]
if sent_f:
    view = view[view["predicted_sentiment"].isin(sent_f)]
if cat_f:
    view = view[view["predicted_category"].isin(cat_f)]
if pri_f:
    view = view[view["predicted_priority"].isin(pri_f)]
st.caption(f"Showing {len(view):,} of {len(out):,} rows. Columns starting with `predicted_` are model output; "
           "`*_confidence` is the raw, uncalibrated model probability.")
st.dataframe(view, hide_index=False)

st.download_button(
    "Download classified CSV",
    data=to_csv_bytes(out),
    file_name=f"classified_{fname.rsplit('.', 1)[0]}.csv",
    mime="text/csv",
    type="primary",
    help="All original columns and rows in original order, plus the predicted_* columns.",
)
st.caption("Text cells beginning with = + - @ get a leading apostrophe in the download so spreadsheets cannot run them as formulas.")
