"""Smart Feedback Classification System - Streamlit entry point + Dashboard page.

Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import config
from src import analytics, ui, visualizations as viz

st.set_page_config(page_title="Smart Feedback Classification", page_icon=":speech_balloon:", layout="wide")


def dashboard() -> None:
    ui.setup_page(
        "Feedback Dashboard",
        "Overview of classified feedback: sentiment, topics, priorities and trends.",
    )
    clf = ui.model_banner()
    label, data = ui.dataset_picker(clf, key="dash_source")
    if data is None or data.empty:
        st.info("No classified feedback yet. Use **Analyze Feedback** or **Bulk Classification** first.")
        return
    ui.source_caveat(label)

    # ---- filters ---------------------------------------------------------- #
    with st.expander("Filters", expanded=False):
        c1, c2, c3 = st.columns(3)
        sentiments = c1.multiselect("Sentiment", config.SENTIMENTS, default=config.SENTIMENTS)
        cats = sorted(data[analytics.CAT].unique())
        categories = c2.multiselect("Category", cats, default=cats)
        priorities = c3.multiselect("Priority", config.PRIORITIES, default=config.PRIORITIES)
        date_range = None
        if analytics.has_usable_dates(data):
            lo, hi = data[config.DATE_COLUMN].min().date(), data[config.DATE_COLUMN].max().date()
            picked = st.date_input("Date range (created_at)", value=(lo, hi), min_value=lo, max_value=hi)
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                date_range = picked
        else:
            st.caption("Date filter and trend chart are unavailable: this data has no usable `created_at` dates.")

    view = analytics.filter_results(data, sentiments, categories, priorities, date_range)
    if view.empty:
        st.warning("No feedback matches the current filters.")
        return

    # ---- KPI cards -------------------------------------------------------- #
    m = analytics.summary_metrics(view)
    k = st.columns(4)
    k[0].metric("Feedback analysed", f"{m['total']:,}")
    k[1].metric("Positive", f"{m['positive_pct']:.0f}%")
    k[2].metric("Neutral", f"{m['neutral_pct']:.0f}%")
    k[3].metric("Negative", f"{m['negative_pct']:.0f}%")
    k2 = st.columns(2)
    k2[0].metric("Most frequently mentioned topic", m["top_topic"], f"{m['top_topic_count']} items", delta_color="off")
    k2[1].metric("Potentially high-priority items", f"{m['high_priority']:,}",
                 help="Rule-based indicator, not a validated emergency detector. A person must review these.")

    # ---- charts ----------------------------------------------------------- #
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.sentiment_distribution(view), width="stretch")
    c2.plotly_chart(viz.topic_distribution(view), width="stretch")
    c3, c4 = st.columns(2)
    c3.plotly_chart(viz.sentiment_by_category(view), width="stretch")
    c4.plotly_chart(viz.priority_distribution(view), width="stretch")
    if analytics.has_usable_dates(view):
        st.plotly_chart(viz.trend_over_time(view), width="stretch")

    # ---- recent table ----------------------------------------------------- #
    st.subheader("Recent feedback and model predictions")
    cols = [c for c in (config.DATE_COLUMN, config.TEXT_COLUMN, analytics.SENT, analytics.CAT, analytics.PRI) if c in view.columns]
    recent = view.sort_values(config.DATE_COLUMN, ascending=False, na_position="last") if analytics.has_usable_dates(view) else view
    table = recent[cols].head(25).copy()
    table[config.TEXT_COLUMN] = table[config.TEXT_COLUMN].map(lambda t: t if len(t) <= 160 else t[:157] + "...")
    st.dataframe(table, hide_index=True)
    st.caption("Predicted values are model output, not verified labels. High-priority items need human verification.")


pages = [
    st.Page(dashboard, title="Dashboard", icon=":material/dashboard:", default=True, url_path="dashboard"),
    st.Page("pages/1_Analyze_Feedback.py", title="Analyze Feedback", icon=":material/edit_note:"),
    st.Page("pages/2_Bulk_Classification.py", title="Bulk Classification", icon=":material/upload_file:"),
    st.Page("pages/3_Model_Training.py", title="Model Training", icon=":material/model_training:"),
    st.Page("pages/4_Feedback_Explorer.py", title="Feedback Explorer", icon=":material/search:"),
]
st.navigation(pages).run()
