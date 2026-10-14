"""Aggregations and filters over classified feedback (no Streamlit imports)."""
from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import config
from src.preprocessing import STOP_WORDS, mask_pii, normalize_text
from src.validation import find_text_column

SENT = "predicted_sentiment"
CAT = "predicted_category"
PRI = "predicted_priority"


def standardize_results(df: pd.DataFrame, text_column: Optional[str] = None) -> pd.DataFrame:
    """Keep only successfully classified rows and add normalised helper columns.

    Result columns: ``feedback_text``, ``predicted_*``, ``created_at`` (datetime or
    NaT), plus every other original column. Row order is preserved.
    """
    out = df.copy()
    text_column = text_column or find_text_column(out) or config.TEXT_COLUMN
    if text_column != config.TEXT_COLUMN and config.TEXT_COLUMN not in out.columns:
        out = out.rename(columns={text_column: config.TEXT_COLUMN})
    for col in (SENT, CAT, PRI):
        if col not in out.columns:
            raise ValueError(f"Results are missing the '{col}' column.")
    out = out[out[SENT].astype(str).str.len() > 0].copy()
    date_col = next((c for c in out.columns if str(c).strip().lower() == config.DATE_COLUMN), None)
    if date_col is not None:
        out[config.DATE_COLUMN] = pd.to_datetime(out[date_col], errors="coerce", format="mixed")
    else:
        out[config.DATE_COLUMN] = pd.NaT
    return out.reset_index(drop=True)


def has_usable_dates(df: pd.DataFrame) -> bool:
    """Only draw time-based charts when there is a genuine time axis."""
    if config.DATE_COLUMN not in df.columns:
        return False
    return df[config.DATE_COLUMN].dropna().dt.normalize().nunique() >= 2


def filter_results(
    df: pd.DataFrame,
    sentiments: Optional[list] = None,
    categories: Optional[list] = None,
    priorities: Optional[list] = None,
    date_range: Optional[tuple] = None,
    query: str = "",
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if sentiments is not None:
        mask &= df[SENT].isin(sentiments)
    if categories is not None:
        mask &= df[CAT].isin(categories)
    if priorities is not None:
        mask &= df[PRI].isin(priorities)
    if date_range and len(date_range) == 2 and has_usable_dates(df):
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        dates = df[config.DATE_COLUMN]
        mask &= dates.notna() & (dates >= start) & (dates < end + pd.Timedelta(days=1))
    if query and query.strip():
        mask &= df[config.TEXT_COLUMN].astype(str).str.contains(query.strip(), case=False, regex=False)
    return df[mask]


def summary_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {"total": 0, "positive_pct": 0.0, "neutral_pct": 0.0, "negative_pct": 0.0,
                "top_topic": None, "top_topic_count": 0, "high_priority": 0}
    counts = df[SENT].value_counts()
    topics = df[CAT].value_counts()
    return {
        "total": total,
        "positive_pct": 100 * counts.get("Positive", 0) / total,
        "neutral_pct": 100 * counts.get("Neutral", 0) / total,
        "negative_pct": 100 * counts.get("Negative", 0) / total,
        "top_topic": topics.index[0],
        "top_topic_count": int(topics.iloc[0]),
        "high_priority": int((df[PRI] == "High").sum()),
    }


def top_terms(texts, n: int = 15, ngram_range: tuple = (1, 1)) -> pd.DataFrame:
    """Most frequent words / phrases (stop words removed, negations kept).

    Exploratory only: frequency says what is mentioned often, not why.
    """
    texts = [t for t in pd.Series(texts).dropna().astype(str).tolist() if t.strip()]
    if not texts:
        return pd.DataFrame({"term": [], "count": []})
    vec = CountVectorizer(
        preprocessor=normalize_text,
        lowercase=False,
        stop_words=STOP_WORDS,
        token_pattern=r"[^\W\d_]{2,}",
        ngram_range=ngram_range,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            matrix = vec.fit_transform(texts)
        except ValueError:  # empty vocabulary
            return pd.DataFrame({"term": [], "count": []})
    counts = matrix.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    out = pd.DataFrame({"term": terms, "count": counts}).sort_values(
        ["count", "term"], ascending=[False, True]
    )
    return out.head(n).reset_index(drop=True)


def representative_examples(df: pd.DataFrame, per_sentiment: int = 2) -> pd.DataFrame:
    """A few readable examples per sentiment (PII masked). Picks the highest
    model-score rows when scores exist, otherwise the first rows."""
    parts = []
    for sentiment in config.SENTIMENTS:
        sub = df[df[SENT] == sentiment]
        if "sentiment_confidence" in sub.columns:
            score = pd.to_numeric(sub["sentiment_confidence"], errors="coerce")
            sub = sub.assign(_s=score).sort_values("_s", ascending=False, na_position="last")
        parts.append(sub.head(per_sentiment))
    out = pd.concat(parts) if parts else df.head(0)
    out = out[[config.TEXT_COLUMN, SENT, CAT, PRI]].copy()
    out[config.TEXT_COLUMN] = out[config.TEXT_COLUMN].map(mask_pii)
    return out.reset_index(drop=True)
