"""Page: search and explore classified feedback."""
import streamlit as st

import config
from src import analytics, ui, visualizations as viz
from src.preprocessing import mask_pii

ui.setup_page("Feedback Explorer", "Search, filter and explore classified feedback.")
clf = ui.model_banner()
label, data = ui.dataset_picker(clf, key="explorer_source")
if data is None or data.empty:
    st.info("No classified feedback yet. Use **Analyze Feedback** or **Bulk Classification** first.")
    st.stop()
ui.source_caveat(label)

# ---- filters ----------------------------------------------------------------- #
f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
query = f1.text_input("Search text", placeholder="e.g. projector")
sentiments = f2.multiselect("Sentiment", config.SENTIMENTS, default=config.SENTIMENTS)
topics_all = sorted(data[analytics.CAT].unique())
topics = f3.multiselect("Topic", topics_all, default=topics_all)
priorities = f4.multiselect("Priority", config.PRIORITIES, default=config.PRIORITIES)
date_range = None
if analytics.has_usable_dates(data):
    lo, hi = data[config.DATE_COLUMN].min().date(), data[config.DATE_COLUMN].max().date()
    picked = st.date_input("Date range (created_at)", value=(lo, hi), min_value=lo, max_value=hi)
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        date_range = picked
view = analytics.filter_results(data, sentiments, topics, priorities, date_range, query)
st.markdown(f"**{len(view):,}** of {len(data):,} items match.")
if view.empty:
    st.warning("Nothing matches these filters. Try removing some.")
    st.stop()

# ---- table ---------------------------------------------------------------------- #
show = view[[c for c in (config.DATE_COLUMN, config.TEXT_COLUMN, analytics.SENT, analytics.CAT, analytics.PRI) if c in view.columns]].copy()
show[config.TEXT_COLUMN] = show[config.TEXT_COLUMN].map(mask_pii)  # contact details are masked on screen
st.dataframe(show, hide_index=True)
st.caption("E-mail addresses, URLs and phone-like numbers are masked in this view. Sentiment, topic and priority are model predictions.")

# ---- terms ----------------------------------------------------------------------- #
st.subheader("Common words and phrases")
ui.note("<b>Exploratory view.</b> These charts count how often words appear in the filtered feedback (common stop words "
        "removed, negations such as 'not' kept). Frequency shows <i>what</i> people mention, not <i>why</i> they feel a certain way.")
t1, t2 = st.columns(2)
n_terms = st.slider("Number of terms", 5, 25, 12)
t1.plotly_chart(viz.top_terms_fig(analytics.top_terms(view[config.TEXT_COLUMN], n_terms, (1, 1)), "Most frequent words"), width="stretch")
t2.plotly_chart(viz.top_terms_fig(analytics.top_terms(view[config.TEXT_COLUMN], n_terms, (2, 2)), "Most frequent two-word phrases"), width="stretch")

# ---- examples -------------------------------------------------------------------- #
st.subheader("Representative examples")
st.caption("A couple of examples per sentiment (highest model score first). Contact details are masked.")
st.dataframe(analytics.representative_examples(view), hide_index=True)
