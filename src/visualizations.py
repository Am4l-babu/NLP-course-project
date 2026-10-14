"""Plotly figures. Sentiment colours are consistent everywhere and every mark
also carries a text label, so meaning never depends on colour alone."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from src.analytics import CAT, PRI, SENT

_MARGIN = dict(l=10, r=10, t=50, b=10)


def _finish(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=16)), margin=_MARGIN, height=height,
                      legend=dict(orientation="h", y=-0.2, title_text=""))
    return fig


def sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df[SENT].value_counts().reindex(config.SENTIMENTS, fill_value=0).reset_index()
    counts.columns = ["Sentiment", "Feedback count"]
    total = max(int(counts["Feedback count"].sum()), 1)
    counts["label"] = counts["Feedback count"].map(lambda v: f"{v} ({100 * v / total:.0f}%)")
    fig = px.bar(counts, x="Sentiment", y="Feedback count", color="Sentiment", text="label",
                 color_discrete_map=config.SENTIMENT_COLORS, category_orders={"Sentiment": config.SENTIMENTS})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, yaxis_title="Number of feedback items", xaxis_title="Predicted sentiment")
    return _finish(fig, "Sentiment distribution")


def topic_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df[CAT].value_counts().reset_index()
    counts.columns = ["Topic", "Feedback count"]
    fig = px.bar(counts.sort_values("Feedback count"), x="Feedback count", y="Topic", orientation="h",
                 text="Feedback count", color_discrete_sequence=["#457B9D"])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Number of feedback items", yaxis_title="Predicted topic")
    return _finish(fig, "Topic distribution", height=max(300, 34 * len(counts) + 100))


def sentiment_by_category(df: pd.DataFrame) -> go.Figure:
    tab = df.groupby([CAT, SENT]).size().reset_index(name="Feedback count")
    tab.columns = ["Topic", "Sentiment", "Feedback count"]
    order = df[CAT].value_counts().index.tolist()
    fig = px.bar(tab, y="Topic", x="Feedback count", color="Sentiment", orientation="h", text="Feedback count",
                 color_discrete_map=config.SENTIMENT_COLORS,
                 category_orders={"Topic": order[::-1], "Sentiment": config.SENTIMENTS})
    fig.update_traces(textposition="inside")
    fig.update_layout(barmode="stack", xaxis_title="Number of feedback items", yaxis_title="Predicted topic")
    return _finish(fig, "Sentiment by topic", height=max(320, 36 * len(order) + 120))


def priority_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df[PRI].value_counts().reindex(config.PRIORITIES, fill_value=0).reset_index()
    counts.columns = ["Priority", "Feedback count"]
    fig = px.bar(counts, x="Priority", y="Feedback count", color="Priority", text="Feedback count",
                 color_discrete_map=config.PRIORITY_COLORS, category_orders={"Priority": config.PRIORITIES})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title="Estimated priority (rule-based)", yaxis_title="Number of feedback items")
    return _finish(fig, "Priority estimate (needs human review)")


def trend_over_time(df: pd.DataFrame, freq: str = "W") -> go.Figure:
    d = df.dropna(subset=[config.DATE_COLUMN]).copy()
    d["Period"] = d[config.DATE_COLUMN].dt.to_period(freq).dt.start_time
    tab = d.groupby(["Period", SENT]).size().reset_index(name="Feedback count")
    tab.columns = ["Period", "Sentiment", "Feedback count"]
    fig = px.line(tab, x="Period", y="Feedback count", color="Sentiment", markers=True,
                  color_discrete_map=config.SENTIMENT_COLORS, category_orders={"Sentiment": config.SENTIMENTS})
    label = "week" if freq == "W" else "month"
    fig.update_layout(xaxis_title=f"Start of {label} (from created_at)", yaxis_title="Number of feedback items")
    return _finish(fig, f"Feedback volume per {label} by predicted sentiment")


def confusion_matrix_fig(matrix, labels: list, title: str) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(z=matrix, x=labels, y=labels, colorscale="Blues", showscale=False,
                   text=matrix, texttemplate="%{text}", hovertemplate="True %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>")
    )
    fig.update_layout(xaxis_title="Predicted label", yaxis_title="True label", yaxis_autorange="reversed")
    fig.update_xaxes(side="bottom", tickangle=-35)
    return _finish(fig, title, height=max(340, 38 * len(labels) + 160))


def model_comparison_fig(candidates: list, title: str) -> go.Figure:
    rows = []
    for c in candidates:
        rows.append({"Model": c["name"], "Measure": "Cross-validation macro-F1 (training split)", "Score": c["cv_f1_macro_mean"]})
        rows.append({"Model": c["name"], "Measure": "Test-set macro-F1", "Score": c["test_metrics"]["f1_macro"]})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Model", y="Score", color="Measure", barmode="group", text=df["Score"].map(lambda v: f"{v:.2f}"),
                 color_discrete_sequence=["#457B9D", "#A8DADC"])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis=dict(range=[0, 1.05], title="Macro F1-score"), xaxis_title="")
    return _finish(fig, title)


def top_terms_fig(terms: pd.DataFrame, title: str) -> go.Figure:
    fig = px.bar(terms.sort_values("count"), x="count", y="term", orientation="h", text="count",
                 color_discrete_sequence=["#6C757D"])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Occurrences in the selected feedback", yaxis_title="")
    return _finish(fig, title, height=max(300, 26 * len(terms) + 110))


def probability_bar(scores: dict, title: str, color_map: dict = None) -> go.Figure:
    items = sorted(scores.items(), key=lambda kv: kv[1])
    df = pd.DataFrame(items, columns=["Label", "Model probability"])
    fig = px.bar(df, x="Model probability", y="Label", orientation="h", color="Label",
                 color_discrete_map=color_map or {}, text=df["Model probability"].map(lambda v: f"{v:.0%}"))
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1.1], tickformat=".0%", title="Raw model probability (uncalibrated)"), yaxis_title="")
    return _finish(fig, title, height=max(180, 34 * len(df) + 90))
