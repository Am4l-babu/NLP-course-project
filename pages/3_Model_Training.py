"""Page: train, compare, evaluate and save the models."""
import json

import pandas as pd
import streamlit as st

import config
from src import ui, visualizations as viz
from src.data_loader import DataLoadError, load_demo_data, read_csv_bytes
from src.predict import FeedbackClassifier
from src.train import TrainingError, train_models
from src.validation import validate_training_df

ui.setup_page("Model Training and Evaluation", "Train the sentiment and topic classifiers and inspect how well they work.")

with st.expander("How it works (TF-IDF, algorithms, methodology)"):
    st.markdown(
        """
**TF-IDF** turns each feedback into a vector of numbers. *Term frequency* counts how often a word or two-word
phrase appears in the text; *inverse document frequency* down-weights terms that appear in almost every
feedback (so "the" counts for little and "projector" or "not working" counts for more). We use word
**unigrams and bigrams**, so "not working" is a feature on its own.

**Algorithms compared** (separately for sentiment and for topic):
- **Logistic Regression** - learns one weight per term per class and turns the weighted sum into class probabilities.
- **Multinomial Naive Bayes** - assumes terms are independent given the class and compares class-conditional term frequencies; very fast, strong on small text data.
- **Linear SVM** - finds the maximum-margin linear boundary between classes; no probability output.

**Method.** One stratified train/test split (fixed seed). Models are compared by stratified k-fold cross-validation
on the *training split only* (the vectoriser is re-fitted in each fold, so there is no leakage). The best mean
**macro F1** is selected - macro F1 weights every class equally, so a rare class cannot hide behind a big one.
The test split is used once, to report accuracy, precision, recall, macro-F1 and confusion matrices.
        """
    )

if not ui.admin_gate():
    st.stop()

# ---- data source ---------------------------------------------------------- #
st.subheader("1. Choose labelled training data")
source = st.radio("Data source", ["Built-in synthetic demo dataset", "Upload my labelled CSV"], horizontal=True)
df, dataset_name = None, None
if source.startswith("Built-in"):
    try:
        df, dataset_name = load_demo_data(), "built-in synthetic demo dataset"
    except DataLoadError as exc:
        st.error(str(exc))
        st.stop()
    ui.note("<b>Synthetic demonstration data</b> (IDs start with <code>DEMO-</code>) - not genuine student or customer "
            "feedback. It is small and only suitable for demos and smoke tests.", warn=True)
else:
    st.markdown(f"Required columns: `{', '.join(config.REQUIRED_TRAINING_COLUMNS)}` (optional: `feedback_id`, `created_at`). "
                f"Sentiment must be one of {config.SENTIMENTS}; category one of {config.CATEGORIES}. "
                f"At least {config.MIN_EXAMPLES_PER_CLASS} examples per sentiment and per category used. "
                "The file is kept in memory only and is **not** saved to disk.")
    up = st.file_uploader("Labelled training CSV", type=["csv"], key="train_upload")
    if up is None:
        st.info("Upload a CSV to continue.")
        st.stop()
    try:
        df, dataset_name = read_csv_bytes(up.getvalue(), up.name), up.name
    except DataLoadError as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()

check = validate_training_df(df)
for w in check.warnings:
    st.warning(w)
if not check.ok:
    for e in check.errors:
        st.error(e)
    st.error("Training is disabled until the problems above are fixed.")
    st.stop()
st.success(f"Validation passed: {check.info['rows_usable']} usable labelled rows.")

d1, d2 = st.columns(2)
d1.markdown("**Sentiment distribution**")
d1.dataframe(pd.Series(check.info["sentiment_counts"], name="rows").rename_axis("sentiment").reset_index(), hide_index=True)
d2.markdown("**Category distribution**")
d2.dataframe(pd.Series(check.info["category_counts"], name="rows").rename_axis("category").reset_index(), hide_index=True)

# ---- training ------------------------------------------------------------- #
st.subheader("2. Train and evaluate")
if st.button("Train models", type="primary"):
    with st.spinner("Training 3 algorithms x 2 tasks with cross-validation..."):
        try:
            result = train_models(df, dataset_name=dataset_name)
        except TrainingError as exc:
            st.session_state.pop("training_result", None)
            st.error(f"Training failed - no model was produced. {exc}")
            st.stop()
    st.session_state["training_result"] = result
    ui.set_classifier(FeedbackClassifier.from_training_result(result), f"trained this session on {dataset_name}")
    st.success("Training finished and evaluation succeeded. The new models are now active for this session "
               "(not yet saved to disk).")

result = st.session_state.get("training_result")
if result is None:
    st.info("Press **Train models** to start.")
    st.stop()

summary = result.to_dict()
sp = summary["split"]
st.markdown(f"**Split:** {sp['n_train']} training / {sp['n_test']} test rows (test fraction {sp['test_size']}, "
            f"stratified by `{sp['stratified_by']}`, seed {summary['seed']}). Trained {summary['trained_at']} UTC.")
for w in summary["warnings"]:
    st.warning(w)

tab_s, tab_c = st.tabs(["Sentiment", "Topic / category"])
for tab, key in ((tab_s, "sentiment"), (tab_c, "category")):
    task, tsum = result.tasks[key], summary["tasks"][key]
    with tab:
        sel = task.selected
        st.success(f"**Selected: {task.selected_name}** - mean cross-validation macro-F1 "
                   f"{sel.cv_f1_macro_mean:.3f} (+/- {sel.cv_f1_macro_std:.3f}, {sel.cv_folds}-fold). "
                   f"Selection metric: {config.SELECTION_METRIC}.")
        rows = []
        for c in tsum["candidates"]:
            m = c["test_metrics"]
            rows.append({"Model": c["name"], "CV macro-F1": round(c["cv_f1_macro_mean"], 3),
                         "Test accuracy": round(m["accuracy"], 3), "Test precision (macro)": round(m["precision_macro"], 3),
                         "Test recall (macro)": round(m["recall_macro"], 3), "Test macro-F1": round(m["f1_macro"], 3),
                         "Probabilities": "yes (uncalibrated)" if c["supports_probabilities"] else "no"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.plotly_chart(viz.model_comparison_fig(tsum["candidates"], f"{key.title()}: model comparison"), width="stretch")
        sm = tsum["candidates"][[c['name'] for c in tsum['candidates']].index(task.selected_name)]["test_metrics"]
        st.plotly_chart(viz.confusion_matrix_fig(sm["confusion_matrix"], sm["labels"],
                                                 f"{key.title()}: confusion matrix of {task.selected_name} (test set, n={sm['n_test']})"),
                        width="stretch")
        st.markdown("**Per-class results (selected model, test set)**")
        st.dataframe(pd.DataFrame(sm["per_class"]).T.round(3).rename_axis("class"))
        st.caption(f"Test set has only {sm['n_test']} rows, so per-class numbers rest on very few examples and are noisy.")

        st.markdown("**Influential terms (diagnostic)**")
        feats = task.features
        st.dataframe(pd.DataFrame({cls: [f"{t} ({w:+.2f})" for t, w in items[:6]] for cls, items in feats.items()}), hide_index=True)
        st.caption("Terms with the largest weights for each class. This shows what the model associates with a class in the "
                   "training data; it does not prove those words *cause* the sentiment or topic.")

        ex = task.examples
        e1, e2 = st.columns(2)
        e1.markdown(f"**Correct test predictions** ({ex['n_correct']} total)")
        e1.dataframe(ex["correct"], hide_index=True)
        e2.markdown(f"**Incorrect test predictions** ({ex['n_incorrect']} total)")
        if ex["n_incorrect"]:
            e2.dataframe(ex["incorrect"], hide_index=True)
        else:
            e2.info("No errors on this small test set - which does not mean the model is perfect.")

# ---- save / download -------------------------------------------------------- #
st.subheader("3. Save or download")
st.caption("Nothing is written to disk until you press **Save**. Saved models go to the local `models/` folder together "
           "with a checksum file; the app only ever loads models from that folder, never from uploads.")
b1, b2, b3 = st.columns(3)
if b1.button("Save models to models/ folder"):
    try:
        ui.get_classifier().save()
        st.success(f"Saved to {config.MODELS_DIR}")
    except OSError as exc:
        st.error(f"Could not save the models: {exc}")
clf = ui.get_classifier()
b2.download_button("Download sentiment model (.joblib)", clf.to_joblib_bytes("sentiment"), config.SENTIMENT_MODEL_FILE)
b3.download_button("Download topic model (.joblib)", clf.to_joblib_bytes("category"), config.CATEGORY_MODEL_FILE)
st.download_button("Download evaluation metrics (JSON)", json.dumps(summary, indent=2), "evaluation_metrics.json", "application/json")
st.caption("Joblib files are Python pickles - only load files that you created yourself.")
