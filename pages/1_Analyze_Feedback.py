"""Page: analyse individual feedback entries."""
import pandas as pd
import streamlit as st

import config
from src import ui, visualizations as viz
from src.predict import PredictionError

ui.setup_page("Analyze Feedback", "Classify one piece of feedback at a time.")
clf = ui.model_banner()

if "history" not in st.session_state:
    st.session_state["history"] = []

with st.form("analyze_form"):
    text = st.text_area(
        "Feedback text",
        height=140,
        max_chars=config.MAX_TEXT_CHARS * 2,
        placeholder="e.g. The classroom projector is not working.",
        help=f"Plain text. Longer than {config.MAX_TEXT_CHARS:,} characters is truncated.",
    )
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted:
    try:
        pred = clf.predict_one(text)
    except PredictionError as exc:
        st.error(f"Could not analyse this feedback: {exc}")
    else:
        if not pred.ok:
            st.warning(pred.message + " Please enter some feedback text.")
        else:
            if pred.message:
                st.info(pred.message)
            st.session_state["history"].insert(0, pred)
            ui.add_single_result(pred)

history = st.session_state["history"]
if not history:
    st.info("Enter feedback above and press **Analyze**. Results from this session will be listed here.")
else:
    latest = history[0]
    st.subheader("Latest result")
    st.markdown(f"> {latest.original_text}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Sentiment**")
        st.markdown(ui.badge(latest.sentiment), unsafe_allow_html=True)
        if latest.sentiment_score is not None:
            st.caption(f"Model probability: {latest.sentiment_score:.0%} (uncalibrated)")
    with c2:
        st.markdown("**Topic**")
        st.markdown(f"### {latest.category}")
        if latest.category_score is not None:
            st.caption(f"Model probability: {latest.category_score:.0%} (uncalibrated)")
    with c3:
        st.markdown("**Priority estimate**")
        st.markdown(ui.badge(latest.priority, "priority"), unsafe_allow_html=True)
        st.caption("Rule-based; a person must verify High items.")

    low = [name for name, s in (("sentiment", latest.sentiment_score), ("topic", latest.category_score))
           if s is not None and s < config.LOW_CONFIDENCE_THRESHOLD]
    if low:
        ui.note(f"The model's score for the {' and '.join(low)} prediction is low "
                f"(below {config.LOW_CONFIDENCE_THRESHOLD:.0%}). Treat it as a suggestion and review manually.", warn=True)
    if latest.sentiment_scores:
        d1, d2 = st.columns(2)
        d1.plotly_chart(viz.probability_bar(latest.sentiment_scores, "Sentiment: raw model probabilities", config.SENTIMENT_COLORS), width="stretch")
        d2.plotly_chart(viz.probability_bar(latest.category_scores, "Topic: raw model probabilities"), width="stretch")
    with st.expander("Why this priority?"):
        for reason in latest.priority_reasons:
            st.markdown(f"- {reason}")
        st.caption("Priority = sentiment + keyword cues (safety, urgency, repeated problems, things not working). "
                   "It is a triage aid, not a validated emergency detector.")

    with st.expander("How to read this result - limitations", expanded=True):
        st.markdown(
            "- The classifier learned word patterns from a **small labelled dataset**; it does not understand meaning.\n"
            "- **Probabilities are raw model outputs and are not calibrated** - 80% does not mean it is right 80% of the time.\n"
            "- **Sarcasm** (\"Great, another delay\"), **mixed sentiment** (\"good teaching but terrible Wi-Fi\") and "
            "**spelling mistakes** are often misclassified.\n"
            "- **Malayalam-English code-mixing** and campus slang are only handled if they appeared in the training data.\n"
            "- Neutral text can still describe an important problem, and negative text is not automatically urgent.\n"
            "- Results are decision *support* for staff; they must not be used to make decisions about individuals."
        )

    if len(history) > 1:
        st.subheader(f"Earlier results this session ({len(history) - 1})")
        rows = [{"Feedback": h.original_text[:140], "Sentiment": h.sentiment, "Topic": h.category, "Priority": h.priority}
                for h in history[1:]]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    if st.button("Clear session results"):
        st.session_state["history"] = []
        st.session_state.pop("single_results", None)
        st.rerun()
